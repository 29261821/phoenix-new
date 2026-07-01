# ==============================================================================
# § 指標 | 相對強弱指數 (Relative Strength Index)
# 核心職責: 衡量價格變動的速度與變動幅度，經典的超買超賣動能振盪器。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| source        | H & G | str  | -        | 無 (必填)       | 價格來源 (如 'close') |
| period        | H & G | int  | 7 ~ 21   | 無 (必填)       | RSI 的計算週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | period 參數值   | 用於 Bias (動能宏觀乖離) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (超買賣翻轉加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 13 ~ 34  | period 參數值   | 用於 Volatility (動能混沌度) 的滾動標準差週期 |

【特徵工程說明】
- RSI 原始輸出為 0~100 的指標。G 接口將其中心化並縮放至 [-1.0, 1.0] 以符合神經網路胃口。
- 透過 adapt_macro_p 觀察 RSI 偏離其近期歷史中樞的程度，捕捉動能的頂底背離。
- 透過 adapt_micro_p 的加速度特徵，提早識別動能的衰竭與反向爆發。
"""
from typing import Dict

import polars as pl

from src.features.functions.shift import calculate as prev
from src.features.functions.wma import calculate as wma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的 RSI 數值 (0~100)。
    供傳統量化腳本作為過濾器 (如 RSI < 30 視為超賣) 調用。

    契約：
    - df 必須包含 params['source'] 指定的欄位。
    - params 必須包含 'source', 'period' 鍵。
    """
    source_col = params["source"]
    period = params["period"]
    source = pl.col(source_col)
    epsilon = 1e-9

    delta = source - prev(series=source, period=1)
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)

    avg_gain = wma(series=gain, length=period)
    avg_loss = wma(series=loss, length=period)

    rs = avg_gain / (avg_loss + epsilon)
    rsi_val = 100 - (100 / (1 + rs))

    return {"type": "scalar", "values": {"RSI": rsi_val}}


def adapt_RSI(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將 [0, 100] 的經典動能指標轉換為 [-1, 1] 的無量綱時空特徵。
    正交分解為：動能絕對水位 (Position)、動能宏觀乖離 (Bias)、翻轉加速度 (Momentum)、動能混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    rsi_val = h_output["values"]["RSI"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["period"])
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", params["period"])

    epsilon = 1e-6

    # 【核心：中心化與縮放】
    # 將 0~100 映射為 -1.0 ~ 1.0 的對稱空間，0 代表多空平衡的 50
    centered_rsi = (rsi_val - 50.0) / 50.0

    # ---------------------------------------------------------
    # (A) Position (動能絕對水位): RSI 的相對位置
    # 語意補值: 0.0 (代表動能處於多空平衡的中立區)
    # ---------------------------------------------------------
    # Stable 版：嚴格約束於 [-1.0, 1.0]
    feat_rsi_position_stable = (
        centered_rsi.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：略微放寬至 [-1.2, 1.2]，包容數值微小抖動
    feat_rsi_position_sensitive = (
        centered_rsi.fill_nan(0.0).fill_null(0.0).clip(-1.2, 1.2).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (動能宏觀乖離): 動能相對於其長線政權的背離
    # 語意補值: 0.0 (當前動能與近期宏觀動能一致)
    # ---------------------------------------------------------
    rsi_ema_macro = centered_rsi.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = centered_rsi - rsi_ema_macro

    # Stable 版：約束於 [-1.0, 1.0]
    feat_rsi_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-2.0, 2.0]，捕捉動能瞬間斷層式逆轉的極端偏離
    feat_rsi_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (超買賣翻轉加速度): 動能信號的變化速度 (一階導數)
    # 語意補值: 0.0 (動能推進維持等速，或陷入鈍化死水)
    # ---------------------------------------------------------
    ema_centered_rsi = centered_rsi.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (centered_rsi - ema_centered_rsi) / (ema_centered_rsi.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0]
    feat_rsi_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極度凸顯 V 轉或 A 轉瞬間強大的反轉加速度
    feat_rsi_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (動能混沌度): RSI 信號的歷史變異數
    # 語意補值: 0.0 (動能維持單一方向的平穩推進，或處於絕對鈍化區)
    # ---------------------------------------------------------
    rsi_vol = centered_rsi.rolling_std(window_size=adapt_vol_p)
    log_rsi_vol = rsi_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_rsi_volatility_stable = (
        log_rsi_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留多空拉鋸導致動能反覆震盪的混沌狀態
    feat_rsi_volatility_sensitive = (
        log_rsi_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_rsi_position_stable": feat_rsi_position_stable,
        "feat_rsi_position_sensitive": feat_rsi_position_sensitive,
        "feat_rsi_bias_stable": feat_rsi_bias_stable,
        "feat_rsi_bias_sensitive": feat_rsi_bias_sensitive,
        "feat_rsi_momentum_stable": feat_rsi_momentum_stable,
        "feat_rsi_momentum_sensitive": feat_rsi_momentum_sensitive,
        "feat_rsi_volatility_stable": feat_rsi_volatility_stable,
        "feat_rsi_volatility_sensitive": feat_rsi_volatility_sensitive,
    }
