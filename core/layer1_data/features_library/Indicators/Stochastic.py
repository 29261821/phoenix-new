# ==============================================================================
# § 指標 | 隨機指標 (Stochastic Oscillator)
# 核心職責: 衡量收盤價在近期價格區間的相對位置，經典的超買超賣動能振盪器。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| source        | H & G | str  | -        | 無 (必填)       | 價格來源 (通常為 'close') |
| k_period      | H & G | int  | 5 ~ 21   | 無 (必填)       | 基礎 %K 的觀察週期 |
| d_period      | H & G | int  | 3 ~ 10   | 無 (必填)       | %D 的 SMA 平滑週期 |
| smooth_k_period| H & G| int  | 1 ~ 5    | 無 (必填)       | %K 本身的 SMA 內部平滑週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | k_period 參數值 | 用於 Bias (交叉乖離) 的長線衰減觀察週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | d_period 參數值 | 用於 Momentum (翻轉加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 13 ~ 34  | k_period 參數值 | 用於 Volatility (動能混沌度) 的滾動標準差週期 |

【特徵工程說明】
- 原始 %K 與 %D 為 0~100 的指標。G 接口將其中心化並縮放至 [-1.0, 1.0] 以符合神經網路胃口。
- Bias 透過計算 %K 與 %D 的差值，精確捕捉兩線即將交叉或發散的 MACD-like 訊號。
- 透過 adapt_micro_p 的加速度特徵，提早識別 Stochastic 在超買/超賣區的彎折與衰竭。
"""
from typing import Dict

import polars as pl

from src.features.functions.highest import calculate as highest
from src.features.functions.lowest import calculate as lowest
from src.features.functions.sma import calculate as sma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的 KD 數值 (0~100)。
    供傳統量化腳本作為過濾器 (如 K < 20 且 K 向上穿越 D 視為作多信號) 調用。

    契約：
    - df 必須包含 params['source'], 'high', 'low' 指定的欄位。
    - params 必須包含 'source', 'k_period', 'd_period', 'smooth_k_period' 鍵。
    """
    source_col, k_period, d_period, smooth_k_period = (
        params["source"],
        params["k_period"],
        params["d_period"],
        params["smooth_k_period"],
    )
    source = pl.col(source_col)
    h, l = pl.col("high"), pl.col("low")
    epsilon = 1e-9

    low_k = lowest(series=l, period=k_period)
    high_k = highest(series=h, period=k_period)

    k_val = 100 * ((source - low_k) / (high_k - low_k + epsilon))
    k_smooth = sma(series=k_val, length=smooth_k_period)
    d_val = sma(series=k_smooth, length=d_period)

    return {"type": "vector", "values": {"K": k_smooth, "D": d_val}}


def adapt_Stochastic(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將 [0, 100] 的雙線動能指標轉換為 [-1, 1] 的無量綱時空特徵。
    正交分解為：動能絕對水位 (Position)、快慢線交叉乖離 (Bias)、翻轉加速度 (Momentum)、動能混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    k_val = h_output["values"]["K"].cast(pl.Float64)
    d_val = h_output["values"]["D"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["k_period"])
    adapt_micro_p = params.get("adapt_micro_p", params["d_period"])
    adapt_vol_p = params.get("adapt_vol_p", params["k_period"])

    epsilon = 1e-6

    # 【核心：中心化與縮放】
    # 將 0~100 映射為 -1.0 ~ 1.0 的對稱空間，0 代表多空平衡的 50
    centered_k = (k_val - 50.0) / 50.0
    centered_d = (d_val - 50.0) / 50.0

    # ---------------------------------------------------------
    # (A) Position (動能絕對水位): 核心 %K 線的相對位置
    # 語意補值: 0.0 (代表動能處於多空平衡的中立區)
    # ---------------------------------------------------------
    # Stable 版：嚴格約束於 [-1.0, 1.0]
    feat_stoch_position_stable = (
        centered_k.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：略微放寬至 [-1.2, 1.2]，包容極值區的數值微小抖動
    feat_stoch_position_sensitive = (
        centered_k.fill_nan(0.0).fill_null(0.0).clip(-1.2, 1.2).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (快慢線交叉乖離): %K 與 %D 的差值
    # 語意補值: 0.0 (兩線完美貼合或剛剛交叉)
    # 數值為正代表多頭動能擴張，數值為負代表空頭動能擴張
    # ---------------------------------------------------------
    bias = centered_k - centered_d

    # Stable 版：約束於 [-0.5, 0.5]
    feat_stoch_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]，捕捉暴拉/暴跌產生的極大雙線張口
    feat_stoch_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (超買賣翻轉加速度): 核心 %K 的變化速度 (一階導數)
    # 語意補值: 0.0 (動能維持等速或在頂底陷入絕對鈍化)
    # 降共線性處理: 減去短線 EMA 並自適應標準化，提早捕捉 K 線的彎折
    # ---------------------------------------------------------
    ema_centered_k = centered_k.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (centered_k - ema_centered_k) / (ema_centered_k.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0]
    feat_stoch_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極度凸顯 V 轉或 A 轉瞬間強大的反轉加速度
    feat_stoch_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (動能混沌度): %K 信號的歷史變異數
    # 語意補值: 0.0 (動能維持單向推進，或處於絕對的平頂/平底鈍化)
    # ---------------------------------------------------------
    stoch_vol = centered_k.rolling_std(window_size=adapt_vol_p)
    log_stoch_vol = stoch_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_stoch_volatility_stable = (
        log_stoch_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留多空反覆爭奪的混沌狀態
    feat_stoch_volatility_sensitive = (
        log_stoch_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_stoch_position_stable": feat_stoch_position_stable,
        "feat_stoch_position_sensitive": feat_stoch_position_sensitive,
        "feat_stoch_bias_stable": feat_stoch_bias_stable,
        "feat_stoch_bias_sensitive": feat_stoch_bias_sensitive,
        "feat_stoch_momentum_stable": feat_stoch_momentum_stable,
        "feat_stoch_momentum_sensitive": feat_stoch_momentum_sensitive,
        "feat_stoch_volatility_stable": feat_stoch_volatility_stable,
        "feat_stoch_volatility_sensitive": feat_stoch_volatility_sensitive,
    }
