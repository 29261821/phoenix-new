# ==============================================================================
# § 指標 | 突破確認 (Breakout Confirmation) - [v3.0 終極版]
# 核心職責: 整合市場結構、ATR 動能與成交量參與度，確認有效突破事件。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱             | 層級  | 類型  | 建議範圍 | 預設值/Fallback | 說明 |
|--------------------|-------|-------|----------|-----------------|------|
| pivots_left        | H & G | int   | 3 ~ 10   | 無 (必填)       | 擺動點 (Pivot) 左側所需的 K 棒數 |
| pivots_right       | H & G | int   | 3 ~ 10   | 無 (必填)       | 擺動點 (Pivot) 右側所需的 K 棒數 |
| atr_len            | H     | int   | 10 ~ 21  | 無 (必填)       | 動能濾網的 ATR 計算週期 |
| atr_mult           | H     | float | 0.5 ~ 3.0| 無 (必填)       | 突破所需的 ATR 乘數 |
| volume_window      | H     | int   | 10 ~ 50  | 無 (必填)       | 參與度濾網的均量週期 |
| volume_multiplier  | H     | float | 1.0 ~ 3.0| 無 (必填)       | 突破所需的均量乘數 |
| adapt_macro_p      | G 專用| int   | 13 ~ 55  | 21              | 用於突破政權記憶 (Position) 的 EMA 衰減週期 |
| adapt_vol_p        | G 專用| int   | 21 ~ 55  | 34              | 用於突破混沌度 (Volatility) 的滾動觀察週期 |

【特徵工程說明】
- 突破事件高度稀疏，G 接口將其轉換為連續的「突破後延伸率 (Post-Breakout Extension)」。
- 透過 adapt_macro_p 形成宏觀政權，衡量多空突破的歷史中樞方向。
- 透過 adapt_vol_p 計算突破信號的群集程度，識別單邊順暢趨勢或高頻雙巴洗盤。
"""
from typing import Dict

import polars as pl

from src.features.functions.atr import calculate as atr
from src.features.functions.pivots import calculate as pivots
from src.features.functions.shift import calculate as prev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    確認一個價格突破是否有效。此 v3.0 終極版整合了市場結構 (Pivots)、
    價格動能 (ATR) 與市場參與度 (Volume) 進行三維確認。
    保留絕對的離散事件碼 (1: Bullish, -1: Bearish, 0: None)，
    供傳統量化策略腳本作為高勝率進場濾網使用。

    契約：
    - df 必須包含 'high', 'low', 'close', 'volume' 欄位。
    - params 必須包含上述 6 個基礎參數鍵。
    """
    # 參數解構
    p_left = params["pivots_left"]
    p_right = params["pivots_right"]
    atr_len = params["atr_len"]
    atr_mult = params["atr_mult"]
    vol_window = params["volume_window"]
    vol_mult = params["volume_multiplier"]

    # --- 1. 市場結構確認 (Pivots) ---
    pivots_source = pivots(series=pl.col("high"), left=p_left, right=p_right)
    is_ph = pivots_source == 1
    is_pl = pivots_source == -1

    ph = pl.when(is_ph).then(pl.col("high")).otherwise(None).forward_fill()
    pl_val = pl.when(is_pl).then(pl.col("low")).otherwise(None).forward_fill()

    prev_ph = prev(series=ph, period=1)
    prev_pl = prev(series=pl_val, period=1)

    is_break_high = pl.col("high") > prev_ph
    is_break_low = pl.col("low") < prev_pl

    # --- 2. 價格動能確認 (ATR) ---
    atr_val = atr(df=df, period=atr_len)
    is_momentum_bull = (pl.col("close") - prev_ph) > (atr_val * atr_mult)
    is_momentum_bear = (prev_pl - pl.col("close")) > (atr_val * atr_mult)

    # --- 3. 市場參與度確認 (Volume) ---
    volume_threshold = (
        pl.col("volume").rolling_mean(window_size=vol_window).shift(1) * vol_mult
    )
    is_volume_confirmed = pl.col("volume") > volume_threshold

    # --- 4. 組合所有條件 ---
    is_confirmed_bull = is_break_high & is_momentum_bull & is_volume_confirmed
    is_confirmed_bear = is_break_low & is_momentum_bear & is_volume_confirmed

    event = (
        pl.when(is_confirmed_bull)
        .then(pl.lit(1, dtype=pl.Int8))
        .when(is_confirmed_bear)
        .then(pl.lit(-1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
    )

    return {"type": "event", "values": {"Event": event}}


def adapt_Breakout_Confirmation(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將極度稀疏的突破事件轉換為連續的時空特徵空間。
    正交分解為：瞬時脈衝 (Momentum)、突破政權 (Position)、突破後延伸率 (Bias)、突破混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端延伸) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，政權衰減與混沌觀察週期全面可由 YAML 配置。
    """
    event = h_output["values"]["Event"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    # 由於基礎參數與特徵的時空衰減無直接關聯，故提供預設常數 21 與 34
    adapt_macro_p = params.get("adapt_macro_p", 21)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    # 防禦性常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Momentum (瞬時確認脈衝): 發生有效突破的當下狀態
    # 語意補值: 0.0 (無突破)
    # ---------------------------------------------------------
    # Stable 版：約束於 [-1.0, 1.0]
    feat_breakout_momentum_stable = (
        event.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：同為事件本身，邊界無異，但保留雙版本一致性
    feat_breakout_momentum_sensitive = (
        event.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Position (突破政權記憶): 宏觀突破方向的衰減中樞
    # 語意補值: 0.0 (近期無突破或多空抵銷)
    # ---------------------------------------------------------
    regime = event.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：EMA 壓縮了數值，約束於 [-0.5, 0.5]，穩定 Transformer 權重
    feat_breakout_position_stable = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]，保留極端單邊連續突破的宏觀政權
    feat_breakout_position_sensitive = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Bias (突破後延伸率 / Post-Breakout Extension):
    # 提取最後一次有效突破的價格，計算當前收盤價對其的乖離。
    # 語意補值: 0.0 (未發生過突破，或剛好拉回突破點)
    # ---------------------------------------------------------
    # 取得最後一次發生突破(1 或 -1)的收盤價
    last_breakout_close = pl.when(event != 0).then(close).otherwise(None).forward_fill()

    # 若尚無突破歷史，用當前 close 代替避免 Null 擴散
    safe_last_close = (
        pl.when(last_breakout_close.is_not_null())
        .then(last_breakout_close)
        .otherwise(close)
    )

    # 為了統一方向語意：
    # 如果是向上的 Bullish 突破，目前 close 越高代表延伸率越好 (正值)
    # 如果是向下的 Bearish 突破，目前 close 越低代表延伸率越好 (正值)
    last_event_direction = (
        pl.when(event != 0).then(event).otherwise(None).forward_fill().fill_null(1.0)
    )
    raw_extension = (close / (safe_last_close + epsilon)) - 1.0
    bias_extension = raw_extension * last_event_direction

    # Stable 版：嚴格約束 [-0.1, 0.1]，專注突破後最初 10% 的健康度發展
    feat_breakout_bias_stable = (
        bias_extension.fill_nan(0.0).fill_null(0.0).clip(-0.1, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-0.3, 0.3]，讓樹模型識別狂暴主升段的 30% 超額延伸
    feat_breakout_bias_sensitive = (
        bias_extension.fill_nan(0.0).fill_null(0.0).clip(-0.3, 0.3).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (突破混沌度): 事件的頻率標準差，識別洗盤還是單邊
    # 語意補值: 0.0 (極度順暢或死水)
    # 防禦處理: 強制套用 log1p 平滑
    # ---------------------------------------------------------
    breakout_vol = event.rolling_std(window_size=adapt_vol_p)
    log_breakout_vol = breakout_vol.log1p()

    # Stable 版：約束於 [0.0, 0.3]
    feat_breakout_volatility_stable = (
        log_breakout_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.3).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.8]，保留多空反覆雙巴 (Whipsaw) 的極端混沌狀態
    feat_breakout_volatility_sensitive = (
        log_breakout_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.8).cast(pl.Float64)
    )

    return {
        "feat_breakout_momentum_stable": feat_breakout_momentum_stable,
        "feat_breakout_momentum_sensitive": feat_breakout_momentum_sensitive,
        "feat_breakout_position_stable": feat_breakout_position_stable,
        "feat_breakout_position_sensitive": feat_breakout_position_sensitive,
        "feat_breakout_bias_stable": feat_breakout_bias_stable,
        "feat_breakout_bias_sensitive": feat_breakout_bias_sensitive,
        "feat_breakout_volatility_stable": feat_breakout_volatility_stable,
        "feat_breakout_volatility_sensitive": feat_breakout_volatility_sensitive,
    }
