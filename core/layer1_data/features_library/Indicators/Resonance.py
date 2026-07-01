# ==============================================================================
# § 指標 | 信號共振引擎 (Signal Resonance Engine)
# 核心職責: 組合多個指標條件以識別高概率的反轉或順勢突破事件。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| div_rsi_p     | H & G | int  | 7 ~ 21   | 無 (必填)       | 內化背離引擎的 RSI 週期 |
| div_p_left    | H & G | int  | 3 ~ 10   | 無 (必填)       | 內化背離引擎的左側 Pivot 週期 |
| div_p_right   | H & G | int  | 3 ~ 10   | 無 (必填)       | 內化背離引擎的右側 Pivot 週期 |
| ms_di_len     | H     | int  | 10 ~ 21  | 無 (必填)       | 內化市場狀態引擎的 DMI 週期 |
| ms_adx_len    | H & G | int  | 10 ~ 21  | 無 (必填)       | 內化市場狀態引擎的 ADX 週期 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | ms_adx_len 參數 | 用於 Position (共振政權) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 5 ~ 21   | div_rsi_p 參數  | 用於 Bias (短線共振記憶) 的 EMA 衰減週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | ms_adx_len 參數 | 用於 Volatility (共振群集度) 的滾動標準差週期 |

【特徵工程說明】
- Resonance 是由多重條件交集而成的極度稀疏事件脈衝 (0/1)。
- G 接口將其擴展為連續的時間面特徵，使得 DL/ML 模型能理解「高概率事件的餘波效應」。
- 透過 adapt_vol_p 衡量共振信號的群集程度，識別市場結構極度不穩定的混沌期。
"""
from typing import Dict

import polars as pl

# [邏輯自治] 遵循 DSL v5.0 (邏輯自治終極版) 的設計思想，此指標為「自產自銷」。
from src.features.functions.pivots import calculate as pivots
from src.features.functions.shift import calculate as prev
from src.features.functions.tr import calculate as tr
from src.features.functions.wma import calculate as wma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算多條件共振信號 (此處示範為：發生底背離 且 市場處於盤整狀態)。
    保留精確的布林觸發信號。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含所有內化引擎所需的基礎參數鍵。
    """
    div_rsi_p, div_p_left, div_p_right = (
        params["div_rsi_p"],
        params["div_p_left"],
        params["div_p_right"],
    )
    ms_di_len, ms_adx_len = params["ms_di_len"], params["ms_adx_len"]
    epsilon = 1e-9
    h, l, c = pl.col("high"), pl.col("low"), pl.col("close")

    # --- 內化的 Divergence 邏輯 ---
    delta = c - prev(series=c, period=1)
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)
    avg_gain = wma(series=gain, length=div_rsi_p)
    avg_loss = wma(series=loss, length=div_rsi_p)
    rs = avg_gain / (avg_loss + epsilon)
    indicator = 100 - (100 / (1 + rs))

    price_pivots = pivots(series=h, left=div_p_left, right=div_p_right)
    ind_pivots = pivots(series=indicator, left=div_p_left, right=div_p_right)

    price_l = pl.when(price_pivots == -1).then(l).otherwise(None)
    ind_l = pl.when(ind_pivots == -1).then(indicator).otherwise(None)

    # 100% 復刻 .forward_fill() 行為
    prev_price_l_ff = prev(series=price_l.forward_fill(), period=1)
    prev_ind_l_ff = prev(series=ind_l.forward_fill(), period=1)

    div_event = ((price_l < prev_price_l_ff) & (ind_l > prev_ind_l_ff)).fill_null(False)

    # --- 內化的 Market_State 邏輯 ---
    up_move = h - prev(series=h, period=1)
    down_move = prev(series=l, period=1) - l
    plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0)
    minus_dm = (
        pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0)
    )
    s_plus_dm = wma(series=plus_dm, length=ms_di_len)
    s_minus_dm = wma(series=minus_dm, length=ms_di_len)
    s_tr = wma(series=tr(df=df), length=ms_di_len)
    plus_di = 100 * s_plus_dm / (s_tr + epsilon)
    minus_di = 100 * s_minus_dm / (s_tr + epsilon)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + epsilon)
    adx = wma(series=dx, length=ms_adx_len)
    is_trending = adx > 25
    market_state_val = pl.when(~is_trending).then(3).otherwise(0)

    # --- 核心共振邏輯 ---
    is_resonance = (div_event == 1) & (market_state_val == 3)

    return {"type": "event", "values": {"Event": is_resonance}}


def adapt_Resonance(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將極度稀疏的 0/1 共振事件，轉換為 DL/ML 可學習的連續時空特徵。
    正交分解為：共振政權水位 (Position)、短線共振記憶 (Bias)、瞬時脈衝 (Momentum)、事件群集度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    event = h_output["values"]["Event"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["ms_adx_len"])
    adapt_micro_p = params.get("adapt_micro_p", params["div_rsi_p"])
    adapt_vol_p = params.get("adapt_vol_p", params["ms_adx_len"])

    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Momentum (瞬時脈衝): 當下 K 棒是否觸發共振事件
    # 語意補值: 0.0 (無事件)
    # ---------------------------------------------------------
    # Stable 版與 Sensitive 版先天已約束於 [0.0, 1.0]
    feat_resonance_momentum_stable = event
    feat_resonance_momentum_sensitive = event

    # ---------------------------------------------------------
    # (B) Bias (短線記憶 / Event Decay): 事件的短期衰減餘波
    # 語意補值: 0.0 (近期無任何共振信號)
    # ---------------------------------------------------------
    short_memory = event.ewm_mean(span=adapt_micro_p, ignore_nulls=True)

    # Stable 版：EMA 壓縮數值，約束於 [0.0, 0.5]
    feat_resonance_bias_stable = (
        short_memory.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]
    feat_resonance_bias_sensitive = (
        short_memory.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Position (共振政權水位): 長期高勝率事件發生的頻率中樞
    # 語意補值: 0.0 (長期來看無明顯的共振結構發生)
    # ---------------------------------------------------------
    long_memory = event.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：約束於 [0.0, 0.2]
    feat_resonance_position_stable = (
        long_memory.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.5]
    feat_resonance_position_sensitive = (
        long_memory.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (事件群集度): 共振事件的歷史變異數
    # 語意補值: 0.0 (市場順暢，無反覆洗盤觸發的密集信號)
    # ---------------------------------------------------------
    resonance_vol = event.rolling_std(window_size=adapt_vol_p)
    log_resonance_vol = resonance_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_resonance_volatility_stable = (
        log_resonance_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]
    feat_resonance_volatility_sensitive = (
        log_resonance_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_resonance_momentum_stable": feat_resonance_momentum_stable,
        "feat_resonance_momentum_sensitive": feat_resonance_momentum_sensitive,
        "feat_resonance_bias_stable": feat_resonance_bias_stable,
        "feat_resonance_bias_sensitive": feat_resonance_bias_sensitive,
        "feat_resonance_position_stable": feat_resonance_position_stable,
        "feat_resonance_position_sensitive": feat_resonance_position_sensitive,
        "feat_resonance_volatility_stable": feat_resonance_volatility_stable,
        "feat_resonance_volatility_sensitive": feat_resonance_volatility_sensitive,
    }
