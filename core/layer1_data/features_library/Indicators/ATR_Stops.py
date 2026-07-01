# ==============================================================================
# § 指標 | ATR 止損帶 (ATR Stops)
# 核心職責: 基於市場波動率動態計算多頭支撐與空頭壓力位，反映市場流動性邊界。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型  | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|-------|----------|-----------------|------|
| atr_len       | H & G | int   | 10 ~ 21  | 無 (必填)       | 計算 ATR 的基礎週期 |
| multiplier    | H     | float | 1.5 ~ 3.0| 無 (必填)       | ATR 的乘數，決定止損帶寬度 |
| adapt_micro_p | G 專用| int   | 5 ~ 14   | atr_len 參數的值| 用於計算 Momentum (動能) 時的 EMA 平滑週期，隔離共線性 |

【特徵工程說明】
- 原始 ATR Stops 為絕對價格位準，G 接口將其轉換為無量綱的相對位置 (%Stop)、頻寬與乖離。
- 透過 adapt_micro_p 決定模型對「撞擊止損帶加速度」的敏感度。
"""
from typing import Dict

import polars as pl

from src.features.functions.atr import calculate as atr


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算基於 ATR 的動態止損位。
    保留絕對價格尺度的止損線，供傳統量化腳本或交易執行模組直接掛單使用。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含 'atr_len', 'multiplier' 鍵。
    """
    atr_len = params["atr_len"]
    multiplier = params["multiplier"]

    atr_val = atr(df=df, period=atr_len)

    long_stop = pl.col("high") - atr_val * multiplier
    short_stop = pl.col("low") + atr_val * multiplier

    return {"type": "level", "values": {"Long": long_stop, "Short": short_stop}}


def adapt_ATR_Stops(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將絕對價格止損線轉換為供 DL/ML 使用的無尺度、穩定特徵。
    將止損帶正交分解為：位置逼近率 (Position)、止損寬度 (Volatility)、中樞乖離 (Bias)、逼近動能 (Momentum)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉止損獵殺) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，動能平滑週期可由 YAML 配置。
    """
    long_stop = h_output["values"]["Long"]
    short_stop = h_output["values"]["Short"]

    # 1. 提取基礎參數
    atr_len = params["atr_len"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_micro_p = params.get("adapt_micro_p", atr_len)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (位置特徵): 價格在多空止損帶之間的相對座標 (%Stop)
    # 語意補值: 0.5 (代表處於多空止損帶的正中央，安全地帶)
    # ---------------------------------------------------------
    pct_stop = (close - long_stop) / (short_stop - long_stop + epsilon)

    # Stable 版：嚴格約束於 [0.0, 1.0] 內，防止模型 Activation 偏移
    feat_atr_stops_position_stable = (
        pct_stop.fill_nan(0.5).fill_null(0.5).clip(0.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-1.0, 2.0]，捕捉價格跌破或突破止損線的 Stop Hunt (流動性掠奪)
    feat_atr_stops_position_sensitive = (
        pct_stop.fill_nan(0.5).fill_null(0.5).clip(-1.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Volatility (通道寬度特徵): 止損帶寬度佔價格的百分比
    # 語意補值: 0.0 (代表極度收斂、無止損空間)
    # 防禦處理: 強制套用 log1p 平滑極端波動率擴張
    # ---------------------------------------------------------
    bandwidth = (short_stop - long_stop) / (close + epsilon)
    log_bandwidth = bandwidth.log1p()

    # Stable 版：約束於 [0.0, 0.2] (容許最多 20% 的動態止損空間)
    feat_atr_stops_volatility_stable = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.5] (捕捉黑天鵝級別的止損帶擴張)
    feat_atr_stops_volatility_sensitive = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Bias (中樞乖離特徵): 價格相對於止損帶虛擬中樞的偏離
    # 語意補值: 0.0 (代表完美貼合多空平衡中樞)
    # ---------------------------------------------------------
    midpoint = (long_stop + short_stop) / 2.0
    bias = (close / (midpoint + epsilon)) - 1.0

    # Stable 版：約束於 [-0.1, 0.1]，代表最多 10% 的中樞偏離
    feat_atr_stops_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.1, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.3, 0.3]，保留高波動資產的超限偏離資訊
    feat_atr_stops_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.3, 0.3).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Momentum (止損逼近動能): 位置逼近率 (%Stop) 的加速度
    # 語意補值: 0.0 (代表與止損線的距離保持穩定，無逼近動能)
    # 降共線性處理: 減去自身的 EMA 並標準化，凸顯極速殺跌或暴拉的動能
    # ---------------------------------------------------------
    ema_pct_stop = pct_stop.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    pct_stop_osc = (pct_stop - ema_pct_stop) / (ema_pct_stop.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_atr_stops_momentum_stable = (
        pct_stop_osc.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束，捕捉極端暴力洗盤時的瞬間加速度
    feat_atr_stops_momentum_sensitive = (
        pct_stop_osc.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    return {
        "feat_atr_stops_position_stable": feat_atr_stops_position_stable,
        "feat_atr_stops_position_sensitive": feat_atr_stops_position_sensitive,
        "feat_atr_stops_volatility_stable": feat_atr_stops_volatility_stable,
        "feat_atr_stops_volatility_sensitive": feat_atr_stops_volatility_sensitive,
        "feat_atr_stops_bias_stable": feat_atr_stops_bias_stable,
        "feat_atr_stops_bias_sensitive": feat_atr_stops_bias_sensitive,
        "feat_atr_stops_momentum_stable": feat_atr_stops_momentum_stable,
        "feat_atr_stops_momentum_sensitive": feat_atr_stops_momentum_sensitive,
    }
