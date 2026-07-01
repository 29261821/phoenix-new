# ==============================================================================
# § 指標 | 成交量分佈幾何引擎 (Volume Profile Geometry Engine) v3.0
# 核心職責: 根據【第二邊：集體心理】作戰計畫，動態滾動計算並識別成交量分佈的幾何形態。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口連續化時空特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| window_size   | H & G | int  | 20 ~ 100 | 無 (必填)       | 滾動視窗大小，決定 Profile 涵蓋的 K 棒數 |
| bins          | H     | int  | 10 ~ 50  | 無 (必填)       | 價格分箱的數量 |
| va_pct        | H     | float| 0.6~0.8  | 無 (必填)       | 價值區域的成交量百分比 |
| adapt_macro_p | G 專用| int  | 21 ~ 89  | 55              | 用於 Position (幾何政權) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 5 ~ 21   | 13              | 用於 Momentum (型態切換動能) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | window_size 參數| 用於 Volatility (幾何混沌度) 的滾動標準差週期 |

【特徵工程說明】
- VP Geometry 是離散的型態特徵 (1: P-型, -1: b-型, 0: D-型)。
- G 接口將其轉換為連續政權，透過 adapt_macro_p 觀察近期市場是由多頭還是空頭的籌碼形狀主導。
- 透過 adapt_vol_p 衡量籌碼形狀重塑的頻率，識別單邊行情或上下洗盤。
"""
from typing import Dict

import numpy as np
import polars as pl


def _calculate_profile_for_window(
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    bins: int,
    va_pct: float,
) -> float:
    """為單個數據窗口計算成交量分佈的幾何形態偏移率。"""
    min_price, max_price = np.min(lows), np.max(highs)
    price_range = max_price - min_price

    if price_range < 1e-9:
        return 0.0

    bin_size = price_range / bins
    profile = np.zeros(bins)

    for i in range(len(highs)):
        h, l, v = highs[i], lows[i], volumes[i]
        if v == 0: continue
        start_bin = max(0, int((l - min_price) / bin_size))
        end_bin = min(bins - 1, int((h - min_price) / bin_size))
        num_bins_spanned = (end_bin - start_bin) + 1
        vol_per_bin = v / num_bins_spanned
        for j in range(start_bin, end_bin + 1):
            profile[j] += vol_per_bin

    total_volume = np.sum(profile)
    if total_volume < 1e-9: return 0.0

    # 尋找 POC (Point of Control)
    poc_index = np.argmax(profile)

    # 尋找價值區域 (Value Area) 中點
    target_va_volume = total_volume * va_pct
    current_volume = profile[poc_index]
    val_index, vah_index = poc_index, poc_index
    while current_volume < target_va_volume:
        vah_next, val_next = vah_index + 1, val_index - 1
        vol_vah = profile[vah_next] if vah_next < bins else -1
        vol_val = profile[val_next] if val_next >= 0 else -1
        if vol_vah == -1 and vol_val == -1: break
        if vol_vah > vol_val:
            current_volume += vol_vah
            vah_index = vah_next
        else:
            current_volume += vol_val
            val_index = val_next

    value_area_mid_point = (vah_index + val_index) / 2
    
    # 【核心優化點】：輸出 POC 相對於價值區域中點的「連續偏移率」
    # 數值範圍在 [-1, 1] 之間，0 代表平衡 (D-型)，正數為 P-型傾向，負數為 b-型傾向。
    # 這消除了離散狀態碼導致的零方差問題。
    offset_ratio = (poc_index - value_area_mid_point) / (bins / 2 + 1e-9)
    
    return float(offset_ratio)


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    對價格序列應用滾動窗口，計算成交量分佈的幾何形態。
    本指標屬於「 eager a.k.a. non-expression-based 」類型。
    [契約修復]: 將 Eager 產出的 Series 透過 pl.lit() 封裝為 pl.Expr，完美對接惰性計算圖。

    契約：
    - df 必須包含 'high', 'low', 'volume' 欄位。
    - params 必須包含 'window_size', 'bins', 'va_pct' 鍵。
    """
    window_size, bins, va_pct = params["window_size"], params["bins"], params["va_pct"]

    high_np = df.get_column("high").to_numpy()
    low_np = df.get_column("low").to_numpy()
    volume_np = df.get_column("volume").to_numpy()

    n = len(df)
    shape_values = np.zeros(n)

    for i in range(window_size, n):
        w_h, w_l, w_v = high_np[i-window_size:i], low_np[i-window_size:i], volume_np[i-window_size:i]
        if np.isnan(w_h).any(): continue
        shape_values[i] = _calculate_profile_for_window(w_h, w_l, w_v, bins, va_pct)

    result_series = pl.Series("ProfileShape", shape_values, dtype=pl.Float64)
    return {"type": "event", "values": {"ProfileShape": pl.lit(result_series)}}

def adapt_Volume_Profile_Geometry(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將離散的幾何形態狀態 (-1, 0, 1) 轉換為連續的 DL/ML 寬表特徵。
    正交分解為：幾何政權中樞 (Position)、型態微觀乖離 (Bias)、型態切換脈衝 (Momentum)、幾何重塑混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    if "ProfileShape" not in h_output.get("values", {}): return {}
    shape = h_output["values"]["ProfileShape"].cast(pl.Float64)

    adapt_macro_p = params.get("adapt_macro_p", 55)
    adapt_micro_p = params.get("adapt_micro_p", 13)
    adapt_vol_p = params.get("adapt_vol_p", params["window_size"])

    epsilon = 1e-6

    # 由於輸入已經是連續浮點數，下游的 Position, Bias, Momentum 會天然具備方差。
    regime = shape.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    feat_vp_geo_position_stable = regime.fill_nan(0.0).fill_null(0.0).clip(-0.8, 0.8).cast(pl.Float64)
    feat_vp_geo_position_sensitive = regime.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)

    bias = shape - regime
    feat_vp_geo_bias_stable = bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    feat_vp_geo_bias_sensitive = bias.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)

    shape_ema_micro = shape.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (shape - shape_ema_micro) / (shape_ema_micro.abs() + epsilon)
    feat_vp_geo_momentum_stable = momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    feat_vp_geo_momentum_sensitive = momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)

    shape_volatility = shape.rolling_std(window_size=adapt_vol_p)
    log_shape_vol = shape_volatility.log1p()
    feat_vp_geo_volatility_stable = log_shape_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    feat_vp_geo_volatility_sensitive = log_shape_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)

    return {
        "feat_vp_geo_position_stable": feat_vp_geo_position_stable,
        "feat_vp_geo_position_sensitive": feat_vp_geo_position_sensitive,
        "feat_vp_geo_bias_stable": feat_vp_geo_bias_stable,
        "feat_vp_geo_bias_sensitive": feat_vp_geo_bias_sensitive,
        "feat_vp_geo_momentum_stable": feat_vp_geo_momentum_stable,
        "feat_vp_geo_momentum_sensitive": feat_vp_geo_momentum_sensitive,
        "feat_vp_geo_volatility_stable": feat_vp_geo_volatility_stable,
        "feat_vp_geo_volatility_sensitive": feat_vp_geo_volatility_sensitive,
    }