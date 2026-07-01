# ==============================================================================
# § 指標 | 陷阱引擎 (Trap Engine)
# 核心職責: 識別牛市陷阱 (Bull Trap) 和熊市陷阱 (Bear Trap)，捕捉流動性獵殺。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口連續化時空特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| pivots_left   | H & G | int  | 3 ~ 10   | 無 (必填)       | 擺動點 (Pivot) 左側所需的 K 棒數 |
| pivots_right  | H & G | int  | 3 ~ 10   | 無 (必填)       | 擺動點 (Pivot) 右側所需的 K 棒數 |
| confirm_window| H & G | int  | 2 ~ 5    | 無 (必填)       | 確認假突破的回顧/反轉時間窗口 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | 55              | 用於 Position (陷阱政權水位) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 5 ~ 21   | confirm_window  | 用於 Bias (陷阱短線餘波) 的 EMA 衰減週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | 34              | 用於 Volatility (陷阱群集混沌度) 的滾動標準差週期 |

【特徵工程說明】
- Trap 事件為極度稀疏的離散脈衝 (-2, -1, 0, 1, 2)。
- 透過 adapt_micro_p 創造「事件衰減餘波 (Event Decay)」，將瞬間獵殺脈衝擴展為時間面。
- 透過 adapt_vol_p 衡量市場發生假突破的頻率，極致敏銳地捕捉「絞肉機」行情。
"""
from typing import Dict

import polars as pl

from src.features.functions.pivots import calculate as pivots
from src.features.functions.shift import calculate as prev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的離散事件代碼 (2: 牛市陷阱, -2: 熊市陷阱, 1: 向上突破, -1: 向下突破, 0: 無)。
    確保依賴精準流動性獵殺(Stop Hunt)的傳統量化策略可無縫進場。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含 'pivots_left', 'pivots_right', 'confirm_window' 鍵。
    """
    p_left, p_right, confirm_window = (
        params["pivots_left"],
        params["pivots_right"],
        params["confirm_window"],
    )
    h, l, c = pl.col("high"), pl.col("low"), pl.col("close")

    pivots_source = pivots(series=h, left=p_left, right=p_right)
    is_ph = pivots_source == 1
    is_pl = pivots_source == -1

    swing_high = pl.when(is_ph).then(h).otherwise(None).forward_fill()
    swing_low = pl.when(is_pl).then(l).otherwise(None).forward_fill()

    is_bull_break = h > prev(series=swing_high, period=1)
    is_bear_break = l < prev(series=swing_low, period=1)

    breakout_event = (
        pl.when(is_bull_break).then(1).when(is_bear_break).then(-1).otherwise(0)
    )

    is_bull_trap = prev(series=is_bull_break, period=confirm_window) & (
        c < prev(series=swing_high, period=confirm_window)
    )
    is_bear_trap = prev(series=is_bear_break, period=confirm_window) & (
        c > prev(series=swing_low, period=confirm_window)
    )

    event = (
        pl.when(is_bull_trap)
        .then(pl.lit(2, dtype=pl.Int8))
        .when(is_bear_trap)
        .then(pl.lit(-2, dtype=pl.Int8))
        .otherwise(breakout_event)
    )

    return {"type": "event", "values": {"Event": event}}


def adapt_Trap(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將極度稀疏的離散事件 (-2, -1, 0, 1, 2) 轉換為連續的狀態特徵空間。
    正交分解為：瞬時陷阱脈衝 (Momentum)、陷阱短線餘波 (Bias)、長線陷阱政權 (Position)、陷阱群集度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端獵殺) 雙版本。
    """
    event = h_output["values"]["Event"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", 55)
    adapt_micro_p = params.get("adapt_micro_p", params.get("confirm_window", 13))
    adapt_vol_p = params.get("adapt_vol_p", 34)

    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Momentum (瞬時陷阱脈衝): 當下 K 棒發生的絕對結構事件
    # 語意補值: 0.0 (無事件發生)
    # 將 -2 到 2 的事件碼除以 2.0，標準化至 [-1.0, 1.0] 空間
    # ---------------------------------------------------------
    impulse = event / 2.0

    # Stable 版：嚴格約束於 [-1.0, 1.0]
    feat_trap_momentum_stable = (
        impulse.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：同為事件脈衝，但保留版本一致性
    feat_trap_momentum_sensitive = (
        impulse.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (陷阱短線餘波): 假突破事件的短期衰減 (Event Decay)
    # 語意補值: 0.0 (近期無任何突破或陷阱，趨勢平緩)
    # ---------------------------------------------------------
    short_memory = impulse.ewm_mean(span=adapt_micro_p, ignore_nulls=True)

    # Stable 版：EMA 會讓數值縮小，約束於 [-0.5, 0.5]，穩定 Transformer 權重
    feat_trap_bias_stable = (
        short_memory.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]，保留連續多次流動性獵殺的累積峰值
    feat_trap_bias_sensitive = (
        short_memory.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Position (長線陷阱政權): 衡量宏觀的流動性方向 (Market Regime)
    # 語意補值: 0.0 (長期來看多空的突破與假突破勢均力敵)
    # ---------------------------------------------------------
    long_memory = impulse.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：長期 EMA 數值更小，約束於 [-0.2, 0.2]
    feat_trap_position_stable = (
        long_memory.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.5, 0.5]，捕捉宏觀級別的假突破方向失衡
    feat_trap_position_sensitive = (
        long_memory.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (陷阱群集混沌度): 事件的群集程度 (Choppiness)
    # 語意補值: 0.0 (市場單邊順暢，無任何突破或洗盤)
    # 防禦處理: 強制套用 log1p 平滑滾動變異數
    # ---------------------------------------------------------
    # 如果市場進入絞肉機模式，不斷向上假突破後又向下假突破，數值會極度飆高
    trap_volatility = impulse.rolling_std(window_size=adapt_vol_p)
    log_trap_vol = trap_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_trap_volatility_stable = (
        log_trap_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留極端震盪洗盤時的群集特徵 (Event Clustering)
    feat_trap_volatility_sensitive = (
        log_trap_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_trap_momentum_stable": feat_trap_momentum_stable,
        "feat_trap_momentum_sensitive": feat_trap_momentum_sensitive,
        "feat_trap_bias_stable": feat_trap_bias_stable,
        "feat_trap_bias_sensitive": feat_trap_bias_sensitive,
        "feat_trap_position_stable": feat_trap_position_stable,
        "feat_trap_position_sensitive": feat_trap_position_sensitive,
        "feat_trap_volatility_stable": feat_trap_volatility_stable,
        "feat_trap_volatility_sensitive": feat_trap_volatility_sensitive,
    }
