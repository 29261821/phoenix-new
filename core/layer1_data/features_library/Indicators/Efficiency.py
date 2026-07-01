# ==============================================================================
# § 指標 | 效率比率 (Efficiency Ratio)
# 核心職責: 計算 Kaufman's Efficiency Ratio (ER)，衡量價格移動的訊噪比與趨勢平滑度。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| source        | H & G | str  | -        | 無 (必填)       | 價格來源 (如 'close') |
| period        | H & G | int  | 10 ~ 30  | 無 (必填)       | ER 效率比率的基礎計算週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | 21              | 用於 Bias (宏觀乖離) 計算的長線 EMA 週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (加速度) 計算的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | 34              | 用於 Volatility (政權混沌度) 的滾動標準差週期 |

【特徵工程說明】
- 原始 ER 為 0~1 的絕對比例，G 接口將其轉換為無尺度連續特徵。
- 透過 adapt_macro_p 觀察 ER 相對於歷史均值的乖離，捕捉趨勢衰竭。
- 透過 adapt_vol_p 衡量市場效率的穩定度，識別單邊政權或頻繁震盪。
"""
from typing import Dict

import polars as pl

from src.features.functions.abs import calculate as abs_val
from src.features.functions.shift import calculate as prev
from src.features.functions.sum import calculate as rolling_sum


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算考夫曼效率比率 (ER)。
    保留原始 0.0 到 1.0 的絕對比例數值。
    確保舊有基於 ER 作為趨勢啟動濾網 (如 ER > 0.3) 的策略，或是 KAMA 自適應均線
    的平滑常數引擎可以無縫調用。

    契約：
    - df 必須包含 params['source'] 指定的欄位。
    - params 必須包含 'source', 'period' 鍵。
    """
    source_col = params["source"]
    period = params["period"]
    source = pl.col(source_col)
    epsilon = 1e-9

    direction = abs_val(series=(source - prev(series=source, period=period)))
    volatility = rolling_sum(
        series=abs_val(series=(source - prev(series=source, period=1))), period=period
    )
    er = direction / (volatility + epsilon)

    return {"type": "scalar", "values": {"ER": er}}


def adapt_Efficiency(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將原本理論值介於 [0, 1] 的效率比率，進行高階時序特徵萃取。
    正交分解為：絕對效率水位 (Position)、效率宏觀乖離 (Bias)、效率加速度 (Momentum) 與 政權混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端趨勢爆發) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，所有動能、乖離與變異數週期全面可由 YAML 配置。
    """
    er = h_output["values"]["ER"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    # 保留原 ML 專家調校的預設值，同時開放 YAML 覆寫
    adapt_macro_p = params.get("adapt_macro_p", 21)
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    # 防禦性常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (市場效率絕對水位): 當前市場是處於趨勢還是震盪雜訊
    # 語意補值: 0.0 (代表極度無效率，純粹的隨機漫步與雜訊)
    # 理論上 ER 必落於 [0, 1] 之間
    # ---------------------------------------------------------
    # Stable 版：嚴格約束於 [0.0, 1.0]
    feat_er_position_stable = (
        er.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束防禦異常計算越界，[-0.1, 1.1]
    feat_er_position_sensitive = (
        er.fill_nan(0.0).fill_null(0.0).clip(-0.1, 1.1).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (市場效率宏觀乖離): ER 相對於長線均線的偏離
    # 語意補值: 0.0 (代表當前效率與歷史基準一致)
    # ---------------------------------------------------------
    er_ema_macro = er.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = er - er_ema_macro

    # Stable 版：約束於 [-0.2, 0.2]，過濾掉過度極端的效率偏移，關注微觀趨勢衰退
    feat_er_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.8, 0.8]，捕捉市場瞬間由死水轉向史詩級單邊趨勢的極端乖離
    feat_er_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.8, 0.8).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (市場效率加速度): 趨勢平滑度的變化速度 (一階導數正規化)
    # 語意補值: 0.0 (市場效率維持等速，無突然的加速或減速)
    # 降共線性處理: 減去短線 EMA 並進行自適應標準化
    # ---------------------------------------------------------
    er_ema_micro = er.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (er - er_ema_micro) / (er_ema_micro.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_er_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯趨勢突破瞬間的強大爆發力
    feat_er_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (政權混沌度 / 效率波動): ER 自身的歷史變異數
    # 語意補值: 0.0 (市場處於非常穩定的政權狀態，無論是穩定盤整或穩定單邊)
    # 防禦處理: 強制套用 log1p 平滑極端的變異數爆炸
    # ---------------------------------------------------------
    er_volatility = er.rolling_std(window_size=adapt_vol_p)
    log_er_vol = er_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.2]
    feat_er_volatility_stable = (
        log_er_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.5]，保留市場在單邊與震盪中瘋狂切換的高維混沌特徵
    feat_er_volatility_sensitive = (
        log_er_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )

    return {
        "feat_er_position_stable": feat_er_position_stable,
        "feat_er_position_sensitive": feat_er_position_sensitive,
        "feat_er_bias_stable": feat_er_bias_stable,
        "feat_er_bias_sensitive": feat_er_bias_sensitive,
        "feat_er_momentum_stable": feat_er_momentum_stable,
        "feat_er_momentum_sensitive": feat_er_momentum_sensitive,
        "feat_er_volatility_stable": feat_er_volatility_stable,
        "feat_er_volatility_sensitive": feat_er_volatility_sensitive,
    }
