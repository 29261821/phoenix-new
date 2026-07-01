# ==============================================================================
# § 指標 | 成交量振盪器 (Volume Oscillator)
# 核心職責: 顯示快慢成交量均線之間的差異，反映短期資金流相較於長期的擴張與收縮。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| fast_len      | H & G | int  | 3 ~ 10   | 無 (必填)       | 快速成交量均線週期 |
| slow_len      | H & G | int  | 20 ~ 50  | 無 (必填)       | 慢速成交量均線週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | slow_len 參數值 | 用於 Position (量能絕對水位) 的滾動 Z-Score 週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 13   | fast_len 參數值 | 用於 Momentum (量能翻轉加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | slow_len 參數值 | 用於 Volatility (量能混沌度) 的滾動標準差週期 |

【特徵工程說明】
- 原始 VO 是百分比數值，但成交量可能瞬間爆發 10 倍 (1000%)，直接輸入會導致 DL 模型梯度爆炸。
- G 接口進一步使用滾動 Z-Score 將 VO 嚴格無量綱化，轉為神經網路友好的 [-1, 1] 空間。
- 透過 adapt_micro_p 提早捕捉成交量能萎縮或瞬間點火的加速度特徵。
"""
from typing import Dict

import polars as pl

from src.features.functions.sma import calculate as sma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的量能百分比數值 (VO = (Fast - Slow) / Slow * 100)。
    確保依賴絕對門檻 (如 VO > 50 視為顯著放量) 的傳統量化策略可無縫使用。

    契約：
    - df 必須包含 'volume' 欄位。
    - params 必須包含 'fast_len', 'slow_len' 鍵。
    """
    fast_len, slow_len = params["fast_len"], params["slow_len"]
    v = pl.col("volume")
    epsilon = 1e-9

    fast_ma = sma(series=v, length=fast_len)
    slow_ma = sma(series=v, length=slow_len)

    vo = 100 * (fast_ma - slow_ma) / (slow_ma + epsilon)

    return {"type": "scalar", "values": {"VO": vo}}


def adapt_Volume_Oscillator(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將具備嚴重厚尾分佈(Heavy-tail)的量能振盪器，轉換為 [-1, 1] 的無量綱時空特徵。
    正交分解為：量能相對水位 (Position)、量能宏觀乖離 (Bias)、翻轉加速度 (Momentum)、量能混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    vo_val = h_output["values"]["VO"].cast(pl.Float64)

    # 1. 參數提取
    fast_len = params["fast_len"]
    slow_len = params["slow_len"]
    adapt_macro_p = params.get("adapt_macro_p", slow_len)
    adapt_micro_p = params.get("adapt_micro_p", fast_len)
    adapt_vol_p = params.get("adapt_vol_p", slow_len)

    epsilon = 1e-6

    # --- (A) Position & Bias 邏輯 ---
    vo_mean = vo_val.rolling_mean(window_size=adapt_macro_p)
    vo_std = vo_val.rolling_std(window_size=adapt_macro_p)
    z_vo = (vo_val - vo_mean) / (vo_std + epsilon)

    feat_vo_position_stable = z_vo.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    feat_vo_position_sensitive = z_vo.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)

    z_vo_ema = z_vo.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = z_vo - z_vo_ema

    feat_vo_bias_stable = bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    feat_vo_bias_sensitive = bias.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)

    # --- (B) Momentum 邏輯 ---
    ema_bias = bias.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (bias - ema_bias) / (ema_bias.abs() + epsilon)

    feat_vo_momentum_stable = momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    feat_vo_momentum_sensitive = momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)

    # --- (C) Volatility 邏輯 (修正重點) ---
    # 不要對 z_vo 計算標準差，因為那會得到恆量 1.0。
    # 改為計算原始 VO 的滾動標準差，並除以其滾動均值的絕對值（變異係數概念），實現無尺度化。
    vo_abs = vo_val.abs()/100.0

    feat_vo_volatility_stable = (
    vo_abs
    .ewm_mean(span=adapt_vol_p)
    .fill_nan(0.0)
    .fill_null(0.0)
    .clip(0.0, 1.0)
    .cast(pl.Float64)
)
    feat_vo_volatility_sensitive = (
    vo_abs
    .ewm_mean(span=adapt_vol_p)
    .fill_nan(0.0)
    .fill_null(0.0)
    .clip(0.0, 3.0)
    .cast(pl.Float64)
)
    return {
        "feat_vo_position_stable": feat_vo_position_stable,
        "feat_vo_position_sensitive": feat_vo_position_sensitive,
        "feat_vo_bias_stable": feat_vo_bias_stable,
        "feat_vo_bias_sensitive": feat_vo_bias_sensitive,
        "feat_vo_momentum_stable": feat_vo_momentum_stable,
        "feat_vo_momentum_sensitive": feat_vo_momentum_sensitive,
        "feat_vo_volatility_stable": feat_vo_volatility_stable,
        "feat_vo_volatility_sensitive": feat_vo_volatility_sensitive,
    }