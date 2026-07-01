# ==============================================================================
# § 指標 | 結構傳播 (Structure Propagation)
# 核心職責: 計算自上一次突破事件 (BOS/CHOCH) 以來所經歷的時間跨度，衡量結構老化程度。
# v2.1 更新: [架構升級] 導入 H 接口合約標準與 G 接口結構老化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| left_bars     | H & G | int  | 3 ~ 10   | 無 (必填)       | 擺動點 (Pivot) 左側所需的 K 棒數 |
| right_bars    | H & G | int  | 3 ~ 10   | 無 (必填)       | 擺動點 (Pivot) 右側所需的 K 棒數 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | 55              | 用於 Position (結構老化度) 的 Z-Score 觀察週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (老化加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | 34              | 用於 Volatility (重置混沌度) 的滾動標準差週期 |

【特徵工程說明】
- 原始輸出為「距離上次事件經過的 K 棒數」，這是一個不斷遞增且無上界的數值。
- G 接口將其轉換為無量綱的「結構老化度」，使其具備神經網路友好的常態分佈特性。
- 透過 adapt_vol_p 衡量結構「重置 (Reset)」的頻率混亂度。
"""
from typing import Dict

import numpy as np
import polars as pl

from src.features.functions.pivots import calculate as pivots
from src.features.functions.shift import calculate as prev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的 K 棒計數，供量化腳本判斷「突破信號是否過期」。

    契約：
    - df 必須包含 'high', 'low' 欄位。
    - params 必須包含 'left_bars', 'right_bars' 鍵。
    - [健壯性] 採用迭代計算，完美復刻 DSL 的 prev(state, 1).fill_null(0) 契約。
    """
    left_bars = params["left_bars"]
    right_bars = params["right_bars"]
    h, l = pl.col("high"), pl.col("low")

    pivots_source = pivots(h, left_bars, right_bars)
    is_ph = pivots_source == 1
    is_pl = pivots_source == -1
    last_ph = pl.when(is_ph).then(h).otherwise(None).forward_fill()
    last_pl = pl.when(is_pl).then(l).otherwise(None).forward_fill()

    is_uptrend = (
        (last_ph > prev(last_ph, 1)) & (last_pl > prev(last_pl, 1))
    ).fill_null(False)
    is_downtrend = (
        (last_ph < prev(last_ph, 1)) & (last_pl < prev(last_pl, 1))
    ).fill_null(False)

    break_high = (h > prev(last_ph, 1)).fill_null(False)
    break_low = (l < prev(last_pl, 1)).fill_null(False)

    bos_bull = is_uptrend & break_high
    choch_bear = is_uptrend & break_low
    bos_bear = is_downtrend & break_low
    choch_bull = is_downtrend & break_high

    is_event_expr = bos_bull | choch_bear | bos_bear | choch_bull

    is_event_np = df.select(is_event_expr.alias("is_event"))["is_event"].to_numpy()

    n = len(df)
    bars_since_event_np = np.zeros(n, dtype=np.int32)

    prev_bars = 0

    for i in range(n):
        if is_event_np[i]:
            bars_since_event_np[i] = 0
        else:
            bars_since_event_np[i] = prev_bars + 1
        prev_bars = bars_since_event_np[i]

    return {
        "type": "scalar",
        "values": {
            "BarsSinceEvent": pl.lit(pl.Series("BarsSinceEvent", bars_since_event_np))
        },
    }


def adapt_Propagation(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將無限遞增的事件計數 (Time-since-event) 轉化為無量綱的時空特徵。
    正交分解為：結構老化度 (Position)、老化動量 (Momentum)、重置混沌度 (Volatility)。
    (由於是單純的時間累積，Bias 在此語意較弱，我們用動態 Z-Score 取代)
    """
    bars_since = h_output["values"]["BarsSinceEvent"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", 55)
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (結構老化度): BarsSinceEvent 的動態 Z-Score
    # 語意補值: 0.0 (代表當前的事件間隔符合歷史平均，無過度老化或頻繁發生)
    # 數值越高，代表目前的趨勢架構已經維持了超出常理的時間，隨時可能面臨重置。
    # ---------------------------------------------------------
    bars_mean = bars_since.rolling_mean(window_size=adapt_macro_p)
    bars_std = bars_since.rolling_std(window_size=adapt_macro_p)
    z_age = (bars_since - bars_mean) / (bars_std + epsilon)

    # Stable 版：約束於 [-3.0, 3.0]
    feat_prop_position_stable = (
        z_age.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-5.0, 5.0]
    feat_prop_position_sensitive = (
        z_age.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Momentum (結構老化/重置加速度): Z-Score 的變化速度
    # 語意補值: 0.0 (穩定老化中)
    # 降共線性處理: 減去短線 EMA，極致捕捉「瞬間重置為0」的強大負向脈衝
    # ---------------------------------------------------------
    ema_z_age = z_age.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (z_age - ema_z_age) / (ema_z_age.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0]
    feat_prop_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯變盤重置瞬間的強大負向加速度
    feat_prop_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Volatility (重置混沌度): 結構老化的歷史變異數
    # 語意補值: 0.0 (事件發生頻率極其穩定，或長久未發生)
    # 若數值飆高，代表市場頻繁發生假突破導致事件不斷重置，處於結構混亂期
    # ---------------------------------------------------------
    prop_volatility = z_age.rolling_std(window_size=adapt_vol_p)
    log_prop_vol = prop_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_prop_volatility_stable = (
        log_prop_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]
    feat_prop_volatility_sensitive = (
        log_prop_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_prop_position_stable": feat_prop_position_stable,
        "feat_prop_position_sensitive": feat_prop_position_sensitive,
        "feat_prop_momentum_stable": feat_prop_momentum_stable,
        "feat_prop_momentum_sensitive": feat_prop_momentum_sensitive,
        "feat_prop_volatility_stable": feat_prop_volatility_stable,
        "feat_prop_volatility_sensitive": feat_prop_volatility_sensitive,
    }
