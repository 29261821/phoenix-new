# ==============================================================================
# § 指標 | 平滑異同移動平均線 (MACD)
# 核心職責: 透過雙均線差值，捕捉趨勢的方向、強度與動能翻轉。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口無尺度特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| source        | H & G | str  | -        | 無 (必填)       | 價格來源 (如 'close') |
| fast_period   | H & G | int  | 8 ~ 15   | 無 (必填)       | 快線 EMA 週期 (通常 12) |
| slow_period   | H & G | int  | 21 ~ 34  | 無 (必填)       | 慢線 EMA 週期 (通常 26) |
| signal_period | H & G | int  | 5 ~ 13   | 無 (必填)       | 訊號線 EMA 週期 (通常 9) |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | slow_period 參數| (未直接使用，保留做系統對齊) |
| adapt_micro_p | G 專用| int  | 5 ~ 13   | signal_period 參數| 用於 Momentum (柱狀圖加速度) 的短線平滑週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | slow_period 參數| 用於 Volatility (動能混沌度) 的滾動標準差週期 |

【特徵工程說明】
- 原始 MACD 帶有絕對價格尺度，G 接口將其除以 close 轉換為無量綱特徵 (百分比尺度)。
- 透過 adapt_micro_p 計算柱狀圖的加速度，極致敏銳地捕捉動能衰竭。
- 透過 adapt_vol_p 計算 MACD 波動率，識別大行情啟動前的動能收斂。
"""
from typing import Dict

import polars as pl

from src.features.functions.ema import calculate as ema


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的 MACD 指標值，不進行尺度縮放。
    確保傳統量化策略 (如 MACD 零軸穿越、黃金交叉/死亡交叉) 能無縫對接。

    契約：
    - df 必須包含 params['source'] 指定的欄位。
    - params 必須包含 'source', 'fast_period', 'slow_period', 'signal_period' 鍵。
    """
    source_col = params["source"]
    fast_period = params["fast_period"]
    slow_period = params["slow_period"]
    signal_period = params["signal_period"]
    source = pl.col(source_col)

    fast_ema = ema(series=source, length=fast_period)
    slow_ema = ema(series=source, length=slow_period)
    line = fast_ema - slow_ema
    sig = ema(series=line, length=signal_period)
    hist = line - sig

    return {"type": "vector", "values": {"Line": line, "Signal": sig, "Hist": hist}}


def adapt_MACD(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將具有絕對價格尺度的 MACD 轉換為 DL/ML 必須的無尺度連續特徵。
    正交分解為：動能歷史水位 (Position)、柱狀圖發散乖離 (Bias)、動能翻轉加速度 (Momentum)、動能混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端趨勢) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，動能與變異數週期全面可由 YAML 配置。
    """
    macd_line = h_output["values"]["Line"]
    macd_hist = h_output["values"]["Hist"]

    # 1. 提取基礎參數
    slow_period = params["slow_period"]
    signal_period = params["signal_period"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_micro_p = params.get("adapt_micro_p", signal_period)
    adapt_vol_p = params.get("adapt_vol_p", slow_period)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # 【核心：消除絕對 Scale 污染】
    # 將 MACD line 與 Hist 轉換為佔股價的百分比
    norm_macd = macd_line / (close + epsilon)
    norm_hist = macd_hist / (close + epsilon)

    # ---------------------------------------------------------
    # (A) Position (動能歷史水位): MACD 雙均線差值的相對強度
    # 語意補值: 0.0 (長短均線重合，無明顯動能方向)
    # ---------------------------------------------------------
    # Stable 版：約束於 [-0.05, 0.05]，代表最多關注 5% 的雙線偏離
    feat_macd_position_stable = (
        norm_macd.fill_nan(0.0).fill_null(0.0).clip(-0.05, 0.05).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-0.2, 0.2]，捕捉暴跌時 20% 的極端脫離
    feat_macd_position_sensitive = (
        norm_macd.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (柱狀圖發散乖離): MACD 線相對於其信號線的乖離 (即 Hist)
    # 語意補值: 0.0 (MACD 線與信號線完美貼合，可能即將交叉)
    # ---------------------------------------------------------
    # Stable 版：約束於 [-0.02, 0.02]
    feat_macd_bias_stable = (
        norm_hist.fill_nan(0.0).fill_null(0.0).clip(-0.02, 0.02).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.1, 0.1]，捕捉極端急拉急殺時的主升/主跌段背離
    feat_macd_bias_sensitive = (
        norm_hist.fill_nan(0.0).fill_null(0.0).clip(-0.1, 0.1).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (動能翻轉加速度): 柱狀圖 (Hist) 的加速度
    # 語意補值: 0.0 (動能維持等速發展)
    # 降共線性處理: 減去自身的 EMA 並自適應標準化，極致敏銳地捕捉綠柱/紅柱縮短的反轉瞬間
    # ---------------------------------------------------------
    ema_norm_hist = norm_hist.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (norm_hist - ema_norm_hist) / (ema_norm_hist.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_macd_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯變盤瞬間的強大加速度
    feat_macd_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (動能混沌度): MACD (動能強度) 的歷史變異數
    # 語意補值: 0.0 (動能極度平穩或如死水般收斂)
    # 防禦處理: 強制套用 log1p 平滑
    # 若數值飆高，代表市場趨勢動能在多空之間進行劇烈的拉扯
    # ---------------------------------------------------------
    macd_volatility = norm_macd.rolling_std(window_size=adapt_vol_p)
    log_macd_vol = macd_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.1]
    feat_macd_volatility_stable = (
        log_macd_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.3]，保留市場瘋狂狀態下的極端特徵
    feat_macd_volatility_sensitive = (
        log_macd_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.3).cast(pl.Float64)
    )

    return {
        "feat_macd_position_stable": feat_macd_position_stable,
        "feat_macd_position_sensitive": feat_macd_position_sensitive,
        "feat_macd_bias_stable": feat_macd_bias_stable,
        "feat_macd_bias_sensitive": feat_macd_bias_sensitive,
        "feat_macd_momentum_stable": feat_macd_momentum_stable,
        "feat_macd_momentum_sensitive": feat_macd_momentum_sensitive,
        "feat_macd_volatility_stable": feat_macd_volatility_stable,
        "feat_macd_volatility_sensitive": feat_macd_volatility_sensitive,
    }
