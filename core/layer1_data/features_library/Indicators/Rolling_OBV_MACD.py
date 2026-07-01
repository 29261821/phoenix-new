# ==============================================================================
# § 指標 | 滾動能量潮 MACD (Rolling OBV MACD)
# 核心職責: 將 OBV 的滾動變化率代入 MACD 模型，捕捉資金流向的動能翻轉。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口極致抗尺度特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| obv_period    | H & G | int  | 10 ~ 50  | 無 (必填)       | 獲取 OBV 滾動變化率的視窗週期 |
| fast_period   | H & G | int  | 8 ~ 21   | 無 (必填)       | MACD 快線週期 |
| slow_period   | H & G | int  | 21 ~ 55  | 無 (必填)       | MACD 慢線週期 |
| signal_period | H & G | int  | 5 ~ 13   | 無 (必填)       | MACD 訊號線週期 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | slow_period 參數| 用於 Position (資金動能水位) 的滾動 Z-Score 週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | signal_period 參數| 用於 Momentum (翻轉加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | slow_period 參數| 用於 Volatility (動能混沌度) 的滾動標準差週期 |

【特徵工程說明】
- OBV MACD 的原始輸出帶有「絕對成交量」的 Scale 污染。
- G 接口透過 `adapt_macro_p` 計算 MACD 核心線的 Z-Score 進行無量綱化，並以此標準化柱狀圖。
- Bias 衡量柱狀圖的相對乖離，Momentum 捕捉柱狀圖極速反轉(主力瞬間變臉)的動能。
"""
from typing import Dict

import numpy as np
import polars as pl

from src.features.functions.ema import calculate as ema
from src.features.functions.shift import calculate as prev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對成交量尺度的 OBV MACD 指標值。
    本指標屬於「 eager a.k.a. non-expression-based 」類型。
    [契約修復]: 將 Eager 產出的 Series 透過 pl.lit() 封裝為 pl.Expr，完美對接惰性計算圖。

    契約：
    - df 必須包含 'close', 'volume' 欄位。
    - params 必須包含 'obv_period', 'fast_period', 'slow_period', 'signal_period' 鍵。
    - [健壯性] 內化 OBV 狀態機的迭代計算。
    """
    obv_period, fast_p, slow_p, signal_p = (
        params["obv_period"],
        params["fast_period"],
        params["slow_period"],
        params["signal_period"],
    )

    close_np = df["close"].to_numpy()
    volume_np = df["volume"].to_numpy()
    n = len(df)
    obv_np = np.zeros(n, dtype=np.float64)

    prev_obv = 0.0
    for i in range(1, n):
        if close_np[i] > close_np[i - 1]:
            obv_np[i] = prev_obv + volume_np[i]
        elif close_np[i] < close_np[i - 1]:
            obv_np[i] = prev_obv - volume_np[i]
        else:
            obv_np[i] = prev_obv
        prev_obv = obv_np[i]

    obv_series = pl.Series("obv", obv_np)

    rolling_change = obv_series - prev(series=obv_series, period=obv_period)
    fast_ma = ema(series=rolling_change, length=fast_p)
    slow_ma = ema(series=rolling_change, length=slow_p)
    macd_line = fast_ma - slow_ma
    signal_line = ema(series=macd_line, length=signal_p)
    hist_val = macd_line - signal_line

    return {
        "type": "vector",
        "values": {
            "Line": pl.lit(macd_line),
            "Signal": pl.lit(signal_line),
            "Hist": pl.lit(hist_val),
        },
    }


def adapt_Rolling_OBV_MACD(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將嚴重受成交量尺度污染的 OBV MACD 轉換為無量綱的 DL/ML 特徵。
    正交分解為：動能歷史水位 (Position)、無尺度柱狀圖乖離 (Bias)、柱狀圖加速度 (Momentum)、動能混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    macd_line = h_output["values"]["Line"]
    macd_hist = h_output["values"]["Hist"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["slow_period"])
    adapt_micro_p = params.get("adapt_micro_p", params["signal_period"])
    adapt_vol_p = params.get("adapt_vol_p", params["slow_period"])

    epsilon = 1e-6

    # 【核心工程：動態 Z-Score 尺度消除】
    # 使用宏觀週期計算 MACD Line 的均值與標準差，將其無量綱化
    line_mean = macd_line.rolling_mean(window_size=adapt_macro_p)
    line_std = macd_line.rolling_std(window_size=adapt_macro_p)

    z_line = (macd_line - line_mean) / (line_std + epsilon)
    # 柱狀圖也必須除以相同的 standard deviation 以維持與 Line 相同的相對尺度
    z_hist = macd_hist / (line_std + epsilon)

    # ---------------------------------------------------------
    # (A) Position (資金動能歷史水位): 消除尺度後的 Z-MACD Line
    # 語意補值: 0.0 (長短資金動能線重合，處於歷史均值)
    # ---------------------------------------------------------
    # Stable 版：約束於 [-3.0, 3.0]
    feat_obv_macd_position_stable = (
        z_line.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-5.0, 5.0]，捕捉巨鯨史詩級單邊吸籌或派發的極值
    feat_obv_macd_position_sensitive = (
        z_line.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (無尺度柱狀圖乖離): 正規化後的 Histogram
    # 語意補值: 0.0 (MACD線與信號線完美貼合)
    # ---------------------------------------------------------
    # Stable 版：約束於 [-1.0, 1.0]
    feat_obv_macd_bias_stable = (
        z_hist.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-3.0, 3.0]，捕捉暴拉/暴跌產生的極端籌碼背離空間
    feat_obv_macd_bias_sensitive = (
        z_hist.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (資金動能翻轉加速度): 柱狀圖 (Z-Hist) 的加速度
    # 語意補值: 0.0 (籌碼動能維持等速發展)
    # 降共線性處理: 減去自身的 EMA 並自適應標準化，極度敏銳捕捉主力變臉瞬間
    # ---------------------------------------------------------
    ema_z_hist = z_hist.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (z_hist - ema_z_hist) / (ema_z_hist.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_obv_macd_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]
    feat_obv_macd_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (資金動能混沌度): Z-Line 的歷史變異數
    # 語意補值: 0.0 (籌碼動能死水一灘，或維持極度平穩的單邊推進)
    # 防禦處理: 強制套用 log1p 平滑
    # ---------------------------------------------------------
    obv_macd_volatility = z_line.rolling_std(window_size=adapt_vol_p)
    log_obv_macd_vol = obv_macd_volatility.log1p()

    # Stable 版：約束於 [0.0, 1.0]
    feat_obv_macd_volatility_stable = (
        log_obv_macd_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 2.0]
    feat_obv_macd_volatility_sensitive = (
        log_obv_macd_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 2.0).cast(pl.Float64)
    )

    return {
        "feat_obv_macd_position_stable": feat_obv_macd_position_stable,
        "feat_obv_macd_position_sensitive": feat_obv_macd_position_sensitive,
        "feat_obv_macd_bias_stable": feat_obv_macd_bias_stable,
        "feat_obv_macd_bias_sensitive": feat_obv_macd_bias_sensitive,
        "feat_obv_macd_momentum_stable": feat_obv_macd_momentum_stable,
        "feat_obv_macd_momentum_sensitive": feat_obv_macd_momentum_sensitive,
        "feat_obv_macd_volatility_stable": feat_obv_macd_volatility_stable,
        "feat_obv_macd_volatility_sensitive": feat_obv_macd_volatility_sensitive,
    }
