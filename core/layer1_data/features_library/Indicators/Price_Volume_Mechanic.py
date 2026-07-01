# ==============================================================================
# § 指標 | 價量力學引擎 v3.1 (150分典範版)
# 核心職責: 根據【第一邊：微觀物理】作戰計畫，實現對內在動能的分析。
# v3.1 更新: [架構升級] 導入 H 接口合約標準與 G 接口極致抗尺度特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型  | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|-------|----------|-----------------|------|
| volume_ma_period| H & G| int   | 10 ~ 50  | 無 (必填)       | 判斷「高成交量」的動態基準均線週期 |
| volume_multiplier| H   | float | 1.0 ~ 3.0| 無 (必填)       | 認定為有效動能的高量乘數 |
| adapt_macro_p   | G 專用| int   | 21 ~ 55  | volume_ma_period| 用於 Bias (力道宏觀乖離) 計算的長線 EMA 衰減週期 |
| adapt_micro_p   | G 專用| int   | 3 ~ 13   | 5               | 用於 Momentum (力道翻轉加速度) 計算的短線 EMA 週期 |
| adapt_vol_p     | G 專用| int   | 13 ~ 34  | 21              | 用於 Volatility (力竭混沌度) 的平滑週期 |

【特徵工程說明】
- 原始的 Force 與 Exhaustion 帶有絕對的成交量 Scale，對神經網路來說是致命污染。
- G 接口將其轉化為無量綱的 `淨力道比率 (Net Force Ratio)` 與 `力竭比率 (Exhaustion Ratio)`。
- 透過 adapt_macro_p 觀察淨力道相對於長線均值的背離，捕捉量價背離。
- 透過力竭比率直接作為 Volatility，完美刻畫多空交戰卻無進展的市場高熵 (Entropy) 狀態。
"""
from typing import Any, Dict

import polars as pl

from src.features.functions.sma import calculate as sma


def calculate(df: pl.DataFrame, params: Dict[str, Any], **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對成交量尺度的看漲力、看跌力與力竭向量。
    確保舊有依賴絕對成交量門檻的傳統策略可以無縫對接。

    契約:
    - df: pl.DataFrame, 必須包含 'open', 'high', 'low', 'close', 'volume' 欄位。
    - params: Dict, 必須包含 'volume_ma_period', 'volume_multiplier' 鍵。
    """
    # --- 1. 契約驗證與參數提取 ---
    volume_ma_period: int = params.get("volume_ma_period")
    volume_multiplier: float = params.get("volume_multiplier")

    if not volume_ma_period or not volume_multiplier:
        raise ValueError(
            "Price_Volume_Mechanics_Engine 的參數 'volume_ma_period' 和 'volume_multiplier' 必須被提供。"
        )

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(
                f"輸入 DataFrame 缺少 Price_Volume_Mechanic 所需的欄位: {col}"
            )

    epsilon = 1e-9
    o, h, l, c, v = (
        pl.col("open"),
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        pl.col("volume"),
    )

    # --- 2. 基礎 K 線形態計算 (Candle Morphology) ---
    total_range = h - l
    body_size = (c - o).abs()

    # --- 3. 成交量基準計算 (Volume Baseline) ---
    volume_ma = sma(series=v, length=volume_ma_period)
    is_high_volume = v > (volume_ma * volume_multiplier)

    # --- 4. 核心力學分析 (Core Mechanics Analysis) ---
    bullish_force = (
        pl.when(is_high_volume & (c > o))
        .then(v * (body_size / (total_range + epsilon)))
        .otherwise(0)
        .fill_null(0)
    )

    bearish_force = (
        pl.when(is_high_volume & (c < o))
        .then(v * (body_size / (total_range + epsilon)))
        .otherwise(0)
        .fill_null(0)
    )

    is_exhaustion_candle = body_size < (total_range * 0.2)
    exhaustion = (
        pl.when(is_high_volume & is_exhaustion_candle)
        .then(v * (1 - (body_size / (total_range + epsilon))))
        .otherwise(0)
        .fill_null(0)
    )

    return {
        "type": "vector",
        "values": {
            "Bullish_Force": bullish_force,
            "Bearish_Force": bearish_force,
            "Exhaustion": exhaustion,
        },
    }


def adapt_Price_Volume_Mechanic(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將絕對成交量污染的力學向量轉換為無量綱的 DL/ML 特徵。
    正交分解為：多空力道水位 (Position)、力道宏觀乖離 (Bias)、力道翻轉加速度 (Momentum)、力竭混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    bull = h_output["values"]["Bullish_Force"]
    bear = h_output["values"]["Bearish_Force"]
    exh = h_output["values"]["Exhaustion"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["volume_ma_period"])
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", 21)

    epsilon = 1e-6

    # 【核心：消除絕對 Scale 污染】
    total_force = bull + bear + exh + epsilon

    # 淨力道比率 (Net Force Ratio) [-1.0, 1.0]
    net_force_ratio = (bull - bear) / total_force
    # 力竭比率 (Exhaustion Ratio) [0.0, 1.0]
    exhaustion_ratio = exh / total_force

    # ---------------------------------------------------------
    # (A) Position (多空力道水位): 當下 K 棒的純粹多空方向佔比
    # 語意補值: 0.0 (代表無量、或多空完美抵銷)
    # ---------------------------------------------------------
    # Stable 版與 Sensitive 版：先天約束於 [-1.0, 1.0]
    feat_pvm_position_stable = (
        net_force_ratio.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    feat_pvm_position_sensitive = (
        net_force_ratio.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (力道宏觀乖離): 淨力道相對於長線均線的背離
    # 語意補值: 0.0 (當前力道與歷史環境一致)
    # ---------------------------------------------------------
    force_ema_macro = net_force_ratio.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = net_force_ratio - force_ema_macro

    # Stable 版：約束於 [-1.0, 1.0]
    feat_pvm_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-2.0, 2.0]，捕捉瞬間主力瘋狂倒貨/搶籌造成的極端偏離
    feat_pvm_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (力道翻轉加速度): 淨力道的變化速度
    # 語意補值: 0.0 (力道維持等速)
    # 降共線性處理: 減去自身的短線 EMA 並自適應標準化
    # ---------------------------------------------------------
    ema_net_force = net_force_ratio.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (net_force_ratio - ema_net_force) / (ema_net_force.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0]
    feat_pvm_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯變盤瞬間強大加速度
    feat_pvm_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (力竭混沌度): 採用物理學意義上的「力竭比率」作為市場熵的度量
    # 語意補值: 0.0 (趨勢純粹且順暢，無交戰留下的長影線)
    # ---------------------------------------------------------
    smoothed_exhaustion = exhaustion_ratio.ewm_mean(span=adapt_vol_p, ignore_nulls=True)

    # Stable 版：約束於 [0.0, 0.5] (過濾掉極端十字星雜訊)
    feat_pvm_volatility_stable = (
        smoothed_exhaustion.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留主力在高檔激烈交戰(爆量十字星)的完整特徵
    feat_pvm_volatility_sensitive = (
        smoothed_exhaustion.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_pvm_position_stable": feat_pvm_position_stable,
        "feat_pvm_position_sensitive": feat_pvm_position_sensitive,
        "feat_pvm_bias_stable": feat_pvm_bias_stable,
        "feat_pvm_bias_sensitive": feat_pvm_bias_sensitive,
        "feat_pvm_momentum_stable": feat_pvm_momentum_stable,
        "feat_pvm_momentum_sensitive": feat_pvm_momentum_sensitive,
        "feat_pvm_volatility_stable": feat_pvm_volatility_stable,
        "feat_pvm_volatility_sensitive": feat_pvm_volatility_sensitive,
    }
