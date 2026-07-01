# ==============================================================================
# § 指標 | 成交量分佈 (Volume Profile) v3.4 (頂規合約版)
# 核心職責: 計算每日 POC (控制點) 與價值區域 (Value Area, VAH/VAL)。
# v3.4 更新: [架構升級] 導入 H 接口合約標準與 G 接口無尺度連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| adapt_macro_p | G 專用| int  | 21 ~ 55  | 21              | 用於 Bias (POC 乖離) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (價值區域逼近加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 10 ~ 34  | 21              | (未直接使用，保留做系統對齊) |

【特徵工程說明】
- Volume Profile 的 POC, VAH, VAL 是帶有絕對價格尺度的支撐壓力線。
- G 接口將其轉換為無量綱的相對位置 (%VA)、頻寬 (Value Area Width) 與乖離。
- 透過 adapt_micro_p 決定模型對「價格撞擊價值區域邊界加速度」的敏感度。
"""
from typing import Dict

import numpy as np
import polars as pl


def _calculate_daily_profile_for_group(df_group: pl.DataFrame) -> pl.DataFrame:
    """
    為單個交易日的 DataFrame 組計算成交量分佈 (POC, VAH, VAL)。
    此函數設計為在 Polars 的 `group_by().map_groups()` 中使用。
    """
    if df_group.is_empty():
        return df_group.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("poc"),
            pl.lit(None, dtype=pl.Float64).alias("vah"),
            pl.lit(None, dtype=pl.Float64).alias("val"),
        )

    # --- 1. 從 DataFrame 中解構數據 ---
    highs = df_group["high"].to_numpy()
    lows = df_group["low"].to_numpy()
    volumes = df_group["volume"].to_numpy()

    # 參數設定 (未來可配置)
    bins = 24
    va_pct = 0.70

    # --- 2. 建立價格分箱 (Binning) ---
    min_price, max_price = np.min(lows), np.max(highs)
    price_range = max_price - min_price

    if price_range < 1e-9:  # 如果價格無波動
        poc_price = (min_price + max_price) / 2
        return df_group.with_columns(
            pl.lit(poc_price).alias("poc"),
            pl.lit(max_price).alias("vah"),
            pl.lit(min_price).alias("val"),
        )

    bin_size = price_range / bins
    profile = np.zeros(bins)
    bin_prices = min_price + np.arange(bins) * bin_size + bin_size / 2

    # --- 3. 將成交量分配到各個 bin ---
    for i in range(len(highs)):
        h, l, v = highs[i], lows[i], volumes[i]
        if v is None or v == 0:
            continue

        start_bin = max(0, int((l - min_price) / bin_size)) if l > min_price else 0
        end_bin = min(bins - 1, int((h - min_price) / bin_size)) if h > min_price else 0

        num_bins_spanned = (end_bin - start_bin) + 1

        # [v3.3] 防禦性除法
        if num_bins_spanned > 0:
            vol_per_bin = v / num_bins_spanned
            for j in range(start_bin, end_bin + 1):
                profile[j] += vol_per_bin

    total_volume = np.sum(profile)
    if total_volume < 1e-9:  # 如果無有效成交量
        poc_price = (min_price + max_price) / 2
        return df_group.with_columns(
            pl.lit(poc_price).alias("poc"),
            pl.lit(max_price).alias("vah"),
            pl.lit(min_price).alias("val"),
        )

    # --- 4. 計算 POC 和價值區域 (Value Area) ---
    poc_index = np.argmax(profile)
    poc_price = bin_prices[poc_index]

    target_va_volume = total_volume * va_pct
    current_volume = profile[poc_index]

    val_idx, vah_idx = poc_index, poc_index
    while current_volume < target_va_volume and (val_idx > 0 or vah_idx < bins - 1):
        vah_next = vah_idx + 1
        val_next = val_idx - 1

        vol_vah = profile[vah_next] if vah_next < bins else -1
        vol_val = profile[val_next] if val_next >= 0 else -1

        if vol_vah == -1 and vol_val == -1:
            break

        if vol_vah > vol_val:
            current_volume += vol_vah
            vah_idx = vah_next
        else:
            current_volume += vol_val
            val_idx = val_next

    val_price = bin_prices[max(0, val_idx)]
    vah_price = bin_prices[min(bins - 1, vah_idx)]

    # --- 5. 為該組的所有 K 棒添加相同的結果 ---
    return df_group.with_columns(
        pl.lit(poc_price).alias("poc"),
        pl.lit(vah_price).alias("vah"),
        pl.lit(val_price).alias("val"),
    )


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的價格位準 (POC, VAH, VAL)。
    本指標屬於「 eager a.k.a. non-expression-based 」類型。
    [契約修復]: 將 Eager 產出的 Series 透過 pl.lit() 封裝為 pl.Expr，完美對接惰性計算圖。

    契約：
    - df 必須包含 'timestamp', 'high', 'low', 'volume' 欄位。
    - params: 此指標目前無 H 接口強制參數。
    """

    # 步驟 1: 創建 session_id 用於分組
    df_with_session = df.with_columns(session_id=pl.col("timestamp").dt.date())

    # 步驟 2: 使用 group_by().map_groups() 模式對 DataFrame 進行分組計算
    processed_df = df_with_session.group_by(
        "session_id", maintain_order=True
    ).map_groups(_calculate_daily_profile_for_group)

    # 步驟 3: 從計算完成的 DataFrame 中提取結果 Series
    poc_series = processed_df["poc"]
    vah_series = processed_df["vah"]
    val_series = processed_df["val"]

    return {
        "type": "level",
        "values": {
            "POC": pl.lit(poc_series),
            "VAH": pl.lit(vah_series),
            "VAL": pl.lit(val_series),
        },
    }


def adapt_Volume_Profile(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將絕對價格轉換為供 DL/ML 使用的無量綱穩定特徵。
    正交分解為：相對價值位置 (%VA, Position)、價值區頻寬 (Volatility)、POC 乖離 (Bias)、穿透動能 (Momentum)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端行情) 雙版本。
    """
    poc = h_output["values"]["POC"]
    vah = h_output["values"]["VAH"]
    val = h_output["values"]["VAL"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_micro_p = params.get("adapt_micro_p", 5)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (相對價值位置): 衡量價格在 VAH 與 VAL 間的相對座標 (%VA)
    # 語意補值: 0.5 (代表處於價值區域正中央，籌碼高度共識區)
    # ---------------------------------------------------------
    pct_va = (close - val) / (vah - val + epsilon)

    # Stable 版：約束於 [0.0, 1.0] 內，防止模型 Activation 偏移
    feat_vp_position_stable = (
        pct_va.fill_nan(0.5).fill_null(0.5).clip(0.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-1.0, 2.0]，保留價格遠離價值區域的極端突破資訊
    feat_vp_position_sensitive = (
        pct_va.fill_nan(0.5).fill_null(0.5).clip(-1.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Volatility (價值區頻寬): 衡量高交易量區域的寬度 (佔 POC 的百分比)
    # 語意補值: 0.0 (極度收斂、籌碼單一價位堆積)
    # ---------------------------------------------------------
    bandwidth = (vah - val) / (poc + epsilon)
    log_bandwidth = bandwidth.log1p()

    # Stable 版：約束於 [0.0, 0.1] (容許最多 10% 的價值區域寬度)
    feat_vp_volatility_stable = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.3] (允許捕捉高波資產的價值區擴張)
    feat_vp_volatility_sensitive = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.3).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Bias (POC 乖離): 價格相對於當日控制點 (POC) 的乖離率
    # 語意補值: 0.0 (完美貼合最大成交量價位)
    # ---------------------------------------------------------
    bias = (close / (poc + epsilon)) - 1.0

    # Stable 版：約束於 [-0.05, 0.05]，代表最多 5% 的偏離
    feat_vp_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.05, 0.05).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.15, 0.15]，保留暴拉暴跌時的超限偏離
    feat_vp_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.15, 0.15).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Momentum (穿透動能): 相對價值位置 (%VA) 的加速度
    # 語意補值: 0.0 (無動能方向，在價值區內遊走)
    # 降共線性處理: 減去自身的 EMA 並標準化
    # ---------------------------------------------------------
    ema_pct_va = pct_va.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    pct_va_osc = (pct_va - ema_pct_va) / (ema_pct_va.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_vp_momentum_stable = (
        pct_va_osc.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束，捕捉極強的瞬間突破價值區動能
    feat_vp_momentum_sensitive = (
        pct_va_osc.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    return {
        "feat_vp_position_stable": feat_vp_position_stable,
        "feat_vp_position_sensitive": feat_vp_position_sensitive,
        "feat_vp_volatility_stable": feat_vp_volatility_stable,
        "feat_vp_volatility_sensitive": feat_vp_volatility_sensitive,
        "feat_vp_bias_stable": feat_vp_bias_stable,
        "feat_vp_bias_sensitive": feat_vp_bias_sensitive,
        "feat_vp_momentum_stable": feat_vp_momentum_stable,
        "feat_vp_momentum_sensitive": feat_vp_momentum_sensitive,
    }
