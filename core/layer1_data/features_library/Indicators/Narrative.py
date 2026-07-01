# ==============================================================================
# § 指標 | 宏觀敘事上下文 (Narrative Context)
# 核心職責: 將多個高階上下文引擎 (Trend Health, Market State, Vol Geo) 打包成結構。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維融合特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| th_fast_p     | H & G | int  | 8 ~ 21   | 無 (必填)       | 趨勢健康度 (Trend Health) 快線週期 |
| th_mid_p      | H     | int  | 21 ~ 55  | 無 (必填)       | 趨勢健康度 中線週期 |
| th_slow_p     | H & G | int  | 55 ~ 200 | 無 (必填)       | 趨勢健康度 慢線週期 |
| ms_di_len     | H     | int  | 10 ~ 21  | 無 (必填)       | 市場狀態 (Market State) 內化 DMI 週期 |
| ms_adx_len    | H & G | int  | 10 ~ 21  | 無 (必填)       | 市場狀態 內化 ADX 週期 |
| vg_atr_p      | H     | int  | 10 ~ 21  | 無 (必填)       | 波動幾何 (Vol Geo) ATR 週期 |
| vg_diff_p     | H     | int  | 3 ~ 10   | 無 (必填)       | 波動幾何 的差分回顧週期 |
| adapt_macro_p | G 專用| int  | 55 ~ 200 | th_slow_p 參數  | 用於 Bias (宏觀政權) 的長線衰減週期 |
| adapt_micro_p | G 專用| int  | 5 ~ 21   | th_fast_p 參數  | 用於 Momentum (衝動加速度) 的短線平滑週期 |
| adapt_vol_p   | G 專用| int  | 10 ~ 34  | ms_adx_len 參數 | 用於 Volatility (政權切換混沌度) 的標準差週期 |

【特徵工程說明】
- Narrative 綜合了三大底層維度。G 接口將這三大維度分別正交映射。
- Position (趨勢順暢度): 提取 Trend Health 快慢線相對距離。
- Bias (宏觀政權機率): Market State (0或1) 的長期歷史機率衰減。
- Momentum (波動擴張衝動): Vol Geo (ATR差分) 佔股價的瞬間爆發力。
"""
from typing import Dict

import polars as pl

from src.features.functions.atr import calculate as atr
from src.features.functions.ema import calculate as ema
from src.features.functions.shift import calculate as prev
from src.features.functions.wma import calculate as wma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留原有的 Struct 封裝，同時解構出平坦化 (Flatten) 的維度特徵，
    以兼顧舊有腳本調用與 G 接口的無縫特徵萃取。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含所有基礎計算鍵。
    """
    th_fast_p, th_mid_p, th_slow_p = (
        params["th_fast_p"],
        params["th_mid_p"],
        params["th_slow_p"],
    )
    ms_di_len, ms_adx_len = params["ms_di_len"], params["ms_adx_len"]
    vg_atr_p, vg_diff_p = params["vg_atr_p"], params["vg_diff_p"]
    epsilon = 1e-9

    h, l, c = pl.col("high"), pl.col("low"), pl.col("close")

    # --- 內化的 Trend_Health 邏輯 ---
    fast_ma = ema(c, th_fast_p)
    mid_ma = ema(c, th_mid_p)
    slow_ma = ema(c, th_slow_p)
    trend_health = pl.struct(
        [fast_ma.alias("fast"), mid_ma.alias("mid"), slow_ma.alias("slow")]
    )

    # --- 內化的 Market_State 邏輯 ---
    up_move = h - prev(h, 1)
    down_move = prev(l, 1) - l
    plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0)
    minus_dm = (
        pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0)
    )
    s_plus_dm = wma(plus_dm, ms_di_len)
    s_minus_dm = wma(minus_dm, ms_di_len)
    s_tr = atr(df, ms_di_len)  # 使用標準 ATR
    plus_di = 100 * s_plus_dm / (s_tr + epsilon)
    minus_di = 100 * s_minus_dm / (s_tr + epsilon)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + epsilon)
    adx = wma(dx, ms_adx_len)
    market_state = pl.when(adx > 25).then(1).otherwise(0)

    # --- 內化的 Volatility_Geometry 邏輯 ---
    atr_val = atr(df, vg_atr_p)
    vol_geo = atr_val - prev(atr_val, vg_diff_p)

    # --- 核心敘事構建 ---
    narrative = pl.struct(
        [
            trend_health.alias("trend"),
            market_state.alias("market"),
            vol_geo.alias("vol_geo"),
        ]
    )

    return {
        "type": "vector",
        "values": {
            "Context": narrative,
            "TrendFast": fast_ma,
            "TrendSlow": slow_ma,
            "MarketState": market_state,
            "VolGeo": vol_geo,
        },
    }


def adapt_Narrative(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將宏觀敘事包涵的三大維度 (Trend Health, Market State, Volatility Geometry) 分別正交降維。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，各維度平滑與衰減全面可由 YAML 配置。
    """
    trend_fast = h_output["values"]["TrendFast"]
    trend_slow = h_output["values"]["TrendSlow"]
    market_state = h_output["values"]["MarketState"].cast(pl.Float64)  # 0 or 1
    vol_geo = h_output["values"]["VolGeo"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["th_slow_p"])
    adapt_micro_p = params.get("adapt_micro_p", params["th_fast_p"])
    adapt_vol_p = params.get("adapt_vol_p", params["ms_adx_len"])

    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (趨勢順暢度): 提取自 Trend Health (快慢線相對距離)
    # 語意補值: 0.0 (長短期均線重合，方向不明)
    # ---------------------------------------------------------
    trend_alignment = (trend_fast - trend_slow) / (trend_slow + epsilon)

    # Stable 版：約束於 [-0.2, 0.2]
    feat_narrative_position_stable = (
        trend_alignment.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.5, 0.5]
    feat_narrative_position_sensitive = (
        trend_alignment.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (宏觀政權機率): 提取自 Market State (單邊趨勢機率)
    # 語意補值: 0.0 (長期處於盤整死水)
    # ---------------------------------------------------------
    # 對 0/1 的二元狀態取長期 EMA，等同於「市場處於趨勢政權的歷史機率」
    regime_prob = market_state.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable & Sensitive 版：先天約束於 [0.0, 1.0]
    feat_narrative_bias_stable = (
        regime_prob.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )
    feat_narrative_bias_sensitive = (
        regime_prob.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (波動擴張衝動): 提取自 Volatility Geometry
    # 語意補值: 0.0 (近期波動率無擴張或收斂)
    # 將 ATR 差分除以股價進行無量綱化
    # ---------------------------------------------------------
    vol_surge = vol_geo / (close + epsilon)

    # Stable 版：約束於 [-0.05, 0.05]
    feat_narrative_momentum_stable = (
        vol_surge.fill_nan(0.0).fill_null(0.0).clip(-0.05, 0.05).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-0.15, 0.15]，捕捉暴漲暴跌時的極端波動率引爆
    feat_narrative_momentum_sensitive = (
        vol_surge.fill_nan(0.0).fill_null(0.0).clip(-0.15, 0.15).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (政權切換混沌度): Market State 的歷史變異數
    # 語意補值: 0.0 (市場死心塌地維持趨勢或盤整，無切換)
    # 若數值飆高，代表市場在「趨勢啟動」與「假突破」間頻繁雙巴洗盤
    # ---------------------------------------------------------
    state_volatility = market_state.rolling_std(window_size=adapt_vol_p)

    # Stable & Sensitive 版：0/1 序列的標準差理論極限為 0.5
    feat_narrative_volatility_stable = (
        state_volatility.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    feat_narrative_volatility_sensitive = (
        state_volatility.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )

    return {
        "feat_narrative_position_stable": feat_narrative_position_stable,
        "feat_narrative_position_sensitive": feat_narrative_position_sensitive,
        "feat_narrative_bias_stable": feat_narrative_bias_stable,
        "feat_narrative_bias_sensitive": feat_narrative_bias_sensitive,
        "feat_narrative_momentum_stable": feat_narrative_momentum_stable,
        "feat_narrative_momentum_sensitive": feat_narrative_momentum_sensitive,
        "feat_narrative_volatility_stable": feat_narrative_volatility_stable,
        "feat_narrative_volatility_sensitive": feat_narrative_volatility_sensitive,
    }
