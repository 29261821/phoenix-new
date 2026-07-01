# ==============================================================================
# § 指標 | 市場結構 v2.0 (狀態機修正版)
# 核心職責: 識別市場高低點，捕捉順勢突破 (BOS) 與逆勢反轉 (CHOCH) 事件。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| left_bars     | H & G | int  | 3 ~ 10   | 無 (必填)       | 擺動點 (Pivot) 左側所需的 K 棒數 |
| right_bars    | H & G | int  | 3 ~ 10   | 無 (必填)       | 擺動點 (Pivot) 右側所需的 K 棒數 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | 55              | 用於長線結構中樞 (Position) 的 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 5 ~ 21   | 13              | 用於短線結構記憶 (Bias) 的 EMA 衰減週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | 34              | 用於結構混沌度 (Volatility) 的滾動觀察週期 |

【特徵工程說明】
- 原始事件為極度稀疏的離散脈衝 (-2, -1, 0, 1, 2)，神經網路難以直接學習。
- 透過 adapt_micro_p 創造「事件衰減餘波 (Event Decay)」，將瞬間脈衝擴展為時間面。
- 透過 adapt_macro_p 形成長期的宏觀政權 (Market Regime)，衡量長線結構的多空失衡。
"""
from typing import Dict

import polars as pl

from src.features.functions.pivots import calculate as pivots
from src.features.functions.shift import calculate as prev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留離散的事件代碼 (1: Bull BOS, -1: Bear BOS, 2: Bull CHOCH, -2: Bear CHOCH, 0: None)。
    確保依賴精準 K 棒突破點的傳統 SMC 策略可無縫觸發進場。

    契約：
    - df 必須包含 'high', 'low' 欄位。
    - params 必須包含 'left_bars', 'right_bars' 鍵。
    """
    left_bars = params["left_bars"]
    right_bars = params["right_bars"]

    # --- 1. 識別擺動點 (Pivots) ---
    pivots_high = pivots(series=pl.col("high"), left=left_bars, right=right_bars)
    pivots_low = pivots(
        series=pl.col("low"), left=left_bars, right=right_bars
    )  # 獨立計算低點 pivots
    is_ph = pivots_high == 1
    is_pl = pivots_low == -1

    last_ph_val = pl.when(is_ph).then(pl.col("high")).otherwise(None).forward_fill()
    last_pl_val = pl.when(is_pl).then(pl.col("low")).otherwise(None).forward_fill()

    prev_ph_val = prev(series=last_ph_val, period=1)
    prev_pl_val = prev(series=last_pl_val, period=1)

    # --- 2. 建立趨勢狀態機 ---
    trend_state = (
        pl.when(
            is_ph
            & (pl.col("high") > prev_ph_val)
            & is_pl
            & (pl.col("low") > prev_pl_val)
        )
        .then(pl.lit(1, dtype=pl.Int8))  # Confirmed Uptrend
        .when(
            is_pl
            & (pl.col("low") < prev_pl_val)
            & is_ph
            & (pl.col("high") < prev_ph_val)
        )
        .then(pl.lit(-1, dtype=pl.Int8))  # Confirmed Downtrend
        .otherwise(None)
        .forward_fill()
        .fill_null(0)  # 初始狀態為 0 (盤整)
    )

    is_uptrend_state = trend_state == 1
    is_downtrend_state = trend_state == -1

    # --- 3. 在趨勢狀態下，識別突破事件 ---
    break_high = pl.col("high") > prev_ph_val
    break_low = pl.col("low") < prev_pl_val

    bos_bull = is_uptrend_state & break_high
    choch_bear = is_uptrend_state & break_low
    bos_bear = is_downtrend_state & break_low
    choch_bull = is_downtrend_state & break_high

    # --- 4. 組合事件碼 ---
    event = (
        pl.when(bos_bull)
        .then(pl.lit(1, dtype=pl.Int8))
        .when(bos_bear)
        .then(pl.lit(-1, dtype=pl.Int8))
        .when(choch_bull)
        .then(pl.lit(2, dtype=pl.Int8))
        .when(choch_bear)
        .then(pl.lit(-2, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
    )

    return {"type": "event", "values": {"Event": event}}


def adapt_BOS_CHOCH(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將極度稀疏的離散事件 (-2, -1, 0, 1, 2) 轉換為連續的狀態特徵空間。
    正交分解為：瞬時脈衝 (Momentum)、短線結構記憶 (Bias)、長線結構中樞 (Position)、結構不穩定性 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端洗盤) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，所有事件衰減週期全面可由 YAML 配置。
    """
    event = h_output["values"]["Event"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    # 由於基礎參數是極短週期的 pivots，特徵工程需要較長週期來做事件衰減，故提供預設常數
    adapt_macro_p = params.get("adapt_macro_p", 55)
    adapt_micro_p = params.get("adapt_micro_p", 13)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Momentum (瞬時結構脈衝): 當下 K 棒發生的絕對結構事件
    # 語意補值: 0.0 (無事件發生)
    # 將 -2 到 2 的事件碼除以 2.0，標準化至 [-1.0, 1.0] 空間
    # ---------------------------------------------------------
    impulse = event / 2.0

    # Stable 版：嚴格約束於 [-1.0, 1.0]
    feat_bos_choch_momentum_stable = (
        impulse.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束，允許樹模型識別未預期的突波
    feat_bos_choch_momentum_sensitive = (
        impulse.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (短線結構記憶): 事件的短期衰減 (Event Decay)
    # 語意補值: 0.0 (近期無任何結構破壞，趨勢平緩)
    # ---------------------------------------------------------
    short_memory = event.ewm_mean(span=adapt_micro_p, ignore_nulls=True)

    # Stable 版：EMA 會讓數值縮小，約束於 [-0.5, 0.5]，穩定 Transformer 權重
    feat_bos_choch_bias_stable = (
        short_memory.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]，保留連續多次同向事件的累積峰值
    feat_bos_choch_bias_sensitive = (
        short_memory.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Position (長線結構中樞): 衡量宏觀的 Market Regime
    # 語意補值: 0.0 (多空長期勢均力敵)
    # ---------------------------------------------------------
    long_memory = event.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：長期 EMA 數值更小，約束於 [-0.2, 0.2]
    feat_bos_choch_position_stable = (
        long_memory.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.5, 0.5]，捕捉宏觀級別的結構失衡
    feat_bos_choch_position_sensitive = (
        long_memory.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (結構不穩定性): 市場結構的混亂程度 (Choppiness)
    # 語意補值: 0.0 (市場單邊順暢，無反向洗盤)
    # 防禦處理: 強制套用 log1p 平滑滾動變異數
    # ---------------------------------------------------------
    # 透過滾動標準差，如果頻繁出現 BOS 與 CHOCH 交替，數值會飆高
    structural_vol = event.rolling_std(window_size=adapt_vol_p)
    log_structural_vol = structural_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_bos_choch_volatility_stable = (
        log_structural_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留極端震盪洗盤時的群集特徵 (Volatility Clustering)
    feat_bos_choch_volatility_sensitive = (
        log_structural_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_bos_choch_momentum_stable": feat_bos_choch_momentum_stable,
        "feat_bos_choch_momentum_sensitive": feat_bos_choch_momentum_sensitive,
        "feat_bos_choch_bias_stable": feat_bos_choch_bias_stable,
        "feat_bos_choch_bias_sensitive": feat_bos_choch_bias_sensitive,
        "feat_bos_choch_position_stable": feat_bos_choch_position_stable,
        "feat_bos_choch_position_sensitive": feat_bos_choch_position_sensitive,
        "feat_bos_choch_volatility_stable": feat_bos_choch_volatility_stable,
        "feat_bos_choch_volatility_sensitive": feat_bos_choch_volatility_sensitive,
    }
