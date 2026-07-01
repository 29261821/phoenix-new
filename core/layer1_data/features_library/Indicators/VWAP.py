# ==============================================================================
# § 指標 | 成交量加權平均價 (Volume Weighted Average Price, VWAP)
# 核心職責: 計算日內成交量加權平均價及其標準差帶。
# v3.2 更新: [API 簽名修正] 徹底移除呼叫底層函數時的 kwargs 污染，回歸位置參數。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| stds          | H     | list | -        | [1.0, 1.5, 2.0] | 要計算的標準差通道倍數列表 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | 34              | 用於 Bias (日內成本長線乖離) 的 EMA 週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 13   | 5               | 用於 Momentum (穿透成本加速度) 的短線 EMA 週期 |

【特徵工程說明】
- VWAP 為極強的機構日內防守線。G 接口將價格與 VWAP 的關係轉換為無量綱的 Z-Score (Position)。
- 透過 adapt_micro_p 計算價格穿越 VWAP 或其標準差帶時的加速度。
- 導出日內的 Normalized Volatility，反映當日交投的狂熱程度。
"""
from typing import Dict, List

import polars as pl

from src.features.functions.cumsum import calculate as cumsum
from src.features.functions.pow import calculate as power
from src.features.functions.sqrt import calculate as sqrt


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算 VWAP 及其多個標準差通道。
    將結果封裝為 type: level，供傳統量化腳本作為日內動態支撐/壓力位使用。
    並額外導出 StdDev 供 G 接口使用。

    契約：
    - df 必須包含 'timestamp', 'high', 'low', 'close', 'volume' 欄位。
    - params 可選包含 'stds' (一個浮點數列表)。
    """
    stds: List[float] = params.get("stds", [1.0, 1.5, 2.0, 2.5, 3.0])
    epsilon = 1e-9

    # 依賴 timestamp 取出日期，確保每日(Session)重新計算
    sid = pl.col("timestamp").dt.date()
    h, l, c, v = pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")

    tp = (h + l + c) / 3.0
    tp_vol = tp * v

    # [v3.2 修正] 移除 series= 關鍵字，確保底層函數相容性
    cum_tp_vol = cumsum(tp_vol).over(sid)
    cum_vol = cumsum(v).over(sid)
    vwap_val = cum_tp_vol / (cum_vol + epsilon)

    # 日內變異數與標準差計算
    sq_dev_sum = cumsum(tp * tp * v).over(sid)
    mean_sq = sq_dev_sum / (cum_vol + epsilon)

    # [v3.2 修正] 移除 base= 與 exponent= 關鍵字
    var_val = mean_sq - power(vwap_val, 2)
    std_dev = sqrt(var_val.abs())  # 確保數值為正

    outputs = {"VWAP": vwap_val, "StdDev": std_dev}
    for i, std_mult in enumerate(stds, 1):
        outputs[f"Upper{i}"] = vwap_val + (std_mult * std_dev)
        outputs[f"Lower{i}"] = vwap_val - (std_mult * std_dev)

    return {"type": "level", "values": outputs}


def adapt_VWAP(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將絕對價格尺度的日內平均成本線，轉換為無量綱的 DL/ML 特徵。
    正交分解為：日內成本相對座標 (Position)、價格乖離率 (Bias)、逼近成本加速度 (Momentum)、日內波動頻寬 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端單邊) 雙版本。
    """
    vwap = h_output["values"]["VWAP"]
    std_dev = h_output["values"]["StdDev"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", 34)
    adapt_micro_p = params.get("adapt_micro_p", 5)

    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (日內成本相對座標): 價格相對於 VWAP 的即時 Z-Score
    # 語意補值: 0.0 (代表價格剛好落在 VWAP 上，多空成本平衡)
    # ---------------------------------------------------------
    z_vwap = (close - vwap) / (std_dev + epsilon)

    feat_vwap_position_stable = (
        z_vwap.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )
    feat_vwap_position_sensitive = (
        z_vwap.fill_nan(0.0).fill_null(0.0).clip(-6.0, 6.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (價格日內乖離): 價格相對於 VWAP 的純百分比乖離
    # 語意補值: 0.0 (貼合 VWAP)
    # ---------------------------------------------------------
    bias = (close / (vwap + epsilon)) - 1.0

    feat_vwap_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.05, 0.05).cast(pl.Float64)
    )
    feat_vwap_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (穿透成本加速度): 日內 Z-Score (Position) 的變化速度
    # 語意補值: 0.0 (與 VWAP 的距離維持穩定)
    # ---------------------------------------------------------
    ema_z_vwap = z_vwap.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (z_vwap - ema_z_vwap) / (ema_z_vwap.abs() + epsilon)

    feat_vwap_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    feat_vwap_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (日內波動頻寬): VWAP 日內標準差佔據 VWAP 本身的百分比
    # 語意補值: 0.0 (開盤初期或極度收斂)
    # ---------------------------------------------------------
    volatility = std_dev / (vwap + epsilon)
    log_volatility = volatility.log1p()

    feat_vwap_volatility_stable = (
        log_volatility.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.05).cast(pl.Float64)
    )
    feat_vwap_volatility_sensitive = (
        log_volatility.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.2).cast(pl.Float64)
    )

    return {
        "feat_vwap_position_stable": feat_vwap_position_stable,
        "feat_vwap_position_sensitive": feat_vwap_position_sensitive,
        "feat_vwap_bias_stable": feat_vwap_bias_stable,
        "feat_vwap_bias_sensitive": feat_vwap_bias_sensitive,
        "feat_vwap_momentum_stable": feat_vwap_momentum_stable,
        "feat_vwap_momentum_sensitive": feat_vwap_momentum_sensitive,
        "feat_vwap_volatility_stable": feat_vwap_volatility_stable,
        "feat_vwap_volatility_sensitive": feat_vwap_volatility_sensitive,
    }
