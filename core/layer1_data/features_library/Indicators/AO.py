# ==============================================================================
# § 指標 | 威廉斯震盪指標 (Awesome Oscillator)
# 核心職責: 計算快慢速移動平均線之間的差值，衡量市場動能。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| fast          | H & G | int  | 3 ~ 10   | 無 (必填)       | 快速 SMA 週期 |
| slow          | H & G | int  | 20 ~ 50  | 無 (必填)       | 慢速 SMA 週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 100 | slow 參數的值   | 用於計算 Position (Z-Score) 的宏觀歷史觀察期 |
| adapt_micro_p | G 專用| int  | 3 ~ 13   | fast 參數的值   | 用於 Momentum (動量) 計算時的 EMA 平滑週期，隔離共線性 |
| adapt_vol_p   | G 專用| int  | 20 ~ 55  | adapt_macro_p   | 用於 Volatility (波動率) 滾動標準差的觀察週期 |

【特徵工程說明】
- AO 本身無尺度邊界，必須透過 G 接口的 Z-Score 與百分比進行無量綱化。
- 透過 adapt_macro_p 決定模型觀察歷史動能極值的視角廣度。
- 透過 adapt_micro_p 決定模型對動能反轉 (紅綠柱切換) 的敏感度。
"""
from typing import Dict

import polars as pl

from src.features.functions.sma import calculate as sma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留原始絕對動能差值，不做任何無量綱化或 Clip，
    確保舊有腳本、自然語言策略可無縫判斷零軸穿越與紅綠柱變化。

    契約：
    - df 必須包含 'high', 'low' 欄位。
    - params 必須包含 'fast', 'slow' 鍵。
    """
    fast_period = params["fast"]
    slow_period = params["slow"]

    hl2 = (pl.col("high") + pl.col("low")) / 2.0
    fast_ma = sma(series=hl2, length=fast_period)
    slow_ma = sma(series=hl2, length=slow_period)
    ao_val = fast_ma - slow_ma

    return {"type": "scalar", "values": {"AO": ao_val}}


def adapt_AO(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將絕對動能 (AO) 轉換為供 DL/ML 使用的無尺度、穩定特徵。
    將單一震盪器正交分解為：偏離 (Bias)、位置 (Position)、動量 (Momentum)、波動 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端行情) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，所有滾動週期全面可由 YAML 配置。
    """
    ao_val = h_output["values"]["AO"]

    # 1. 提取基礎參數
    fast_period = params["fast"]
    slow_period = params["slow"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", slow_period)
    adapt_micro_p = params.get("adapt_micro_p", fast_period)
    adapt_vol_p = params.get("adapt_vol_p", adapt_macro_p)

    # 防禦性常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Bias (偏離特徵): 動能佔價格的百分比
    # 語意補值: 0.0 (無動能)
    # ---------------------------------------------------------
    bias = ao_val / (close + epsilon)

    # Stable 版：約束於 [-0.1, 0.1]，代表最多只關注 ±10% 的動能波動
    feat_ao_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.1, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.5, 0.5]，允許捕捉極端行情下的超大動能
    feat_ao_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Position (位置特徵): AO 的滾動 Z-Score (使用 adapt_macro_p)
    # 語意補值: 0.0 (處於歷史常態平均)
    # ---------------------------------------------------------
    ao_rolling_mean = ao_val.rolling_mean(window_size=adapt_macro_p)
    ao_rolling_std = ao_val.rolling_std(window_size=adapt_macro_p)
    z_score = (ao_val - ao_rolling_mean) / (ao_rolling_std + epsilon)

    # Stable 版：嚴格約束 [-3.0, 3.0] 的常態分佈範圍
    feat_ao_position_stable = (
        z_score.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-5.0, 5.0]，捕捉 5 個標準差之外的極端爆發
    feat_ao_position_sensitive = (
        z_score.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (動量特徵): Bias 的加速度/二階導數 (使用 adapt_micro_p)
    # 語意補值: 0.0 (動能無加速現象)
    # 降共線性處理: 減去自身的 EMA 並進行自適應標準化
    # ---------------------------------------------------------
    ema_bias = bias.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (bias - ema_bias) / (ema_bias.abs() + epsilon)

    feat_ao_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    feat_ao_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (波動特徵): Bias 的滾動標準差 (使用 adapt_vol_p)
    # 語意補值: 0.0 (無動能波動)
    # 防禦處理: 強制套用 log1p 平滑可能出現的右偏長尾
    # ---------------------------------------------------------
    volatility = bias.rolling_std(window_size=adapt_vol_p)
    log_volatility = volatility.log1p()

    # Stable 版：約束於 [0.0, 0.1]
    feat_ao_volatility_stable = (
        log_volatility.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.5]
    feat_ao_volatility_sensitive = (
        log_volatility.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )

    return {
        "feat_ao_bias_stable": feat_ao_bias_stable,
        "feat_ao_bias_sensitive": feat_ao_bias_sensitive,
        "feat_ao_position_stable": feat_ao_position_stable,
        "feat_ao_position_sensitive": feat_ao_position_sensitive,
        "feat_ao_momentum_stable": feat_ao_momentum_stable,
        "feat_ao_momentum_sensitive": feat_ao_momentum_sensitive,
        "feat_ao_volatility_stable": feat_ao_volatility_stable,
        "feat_ao_volatility_sensitive": feat_ao_volatility_sensitive,
    }
