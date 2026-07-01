# ==============================================================================
# § 指標 | 市場狀態引擎 (Market State Engine)
# 核心職責: 結合趨勢和波動率指標，將市場劃分為四種不同環境政權。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維狀態映射特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型  | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|-------|----------|-----------------|------|
| di_len        | H & G | int   | 10 ~ 21  | 無 (必填)       | 內化 DMI 的長度 |
| adx_len       | H & G | int   | 10 ~ 21  | 無 (必填)       | 內化 ADX 的長度 |
| bb_period     | H & G | int   | 10 ~ 50  | 無 (必填)       | 內化布林通道的長度 |
| bb_std        | H & G | float | 1.5 ~ 3.0| 無 (必填)       | 內化布林通道的標準差倍數 |
| trend_thresh  | H & G | float | 20 ~ 30  | 無 (必填)       | 定義趨勢啟動的 ADX 門檻 |
| vol_thresh_pct| H & G | float | 0.05~0.2 | 無 (必填)       | 定義波動率擴張的 BBW 門檻 |
| adapt_macro_p | G 專用| int   | 21 ~ 55  | adx_len 參數值  | 用於 Position (狀態政權) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int   | 3 ~ 10   | 5               | 用於 Momentum (狀態切換動能) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int   | 21 ~ 55  | bb_period 參數值| 用於 Volatility (狀態混沌度) 的滾動標準差週期 |

【特徵工程說明】
- 原始狀態為 1, 2, 3, 4 (名目變數)，直接輸入神經網路會產生誤導。
- G 接口將其映射為連續的「狀態評分」: 1.0 (單邊趨勢), 0.5 (趨勢收斂), -0.5 (盤整), -1.0 (極端洗盤)。
- 基於狀態評分正交分解為：政權水位、政權乖離、切換動能與混沌度。
"""
from typing import Dict

import polars as pl

from src.features.functions.shift import calculate as prev
from src.features.functions.sma import calculate as sma
from src.features.functions.stddev import calculate as stddev
from src.features.functions.tr import calculate as tr
from src.features.functions.wma import calculate as wma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    將市場劃分為 4 種離散狀態：
    1: Trending & Expanding (單邊爆發)
    2: Trending & Contracting (趨勢衰竭)
    3: Ranging & Contracting (盤整收斂)
    4: Ranging & Expanding (劇烈洗盤)
    保留絕對的狀態碼，供傳統量化腳本作為大盤環境濾網調用。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含所有基礎計算鍵。
    """
    di_len, adx_len = params["di_len"], params["adx_len"]
    bb_period, bb_std = params["bb_period"], params["bb_std"]
    trend_thresh, vol_thresh_pct = params["trend_thresh"], params["vol_thresh_pct"]
    epsilon = 1e-9

    h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
    prev_h, prev_l = prev(h, 1), prev(l, 1)

    # --- 內化的 DMI/ADX 計算邏輯 ---
    up_move = h - prev_h
    down_move = prev_l - l
    plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0)
    minus_dm = (
        pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0)
    )
    tr_val = tr(df)
    s_plus_dm = wma(plus_dm, di_len)
    s_minus_dm = wma(minus_dm, di_len)
    s_tr = wma(tr_val, di_len)
    plus_di = 100 * s_plus_dm / (s_tr + epsilon)
    minus_di = 100 * s_minus_dm / (s_tr + epsilon)
    dx_val = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + epsilon)
    adx = wma(dx_val, adx_len)

    # --- 內化的 Bollinger Bands 計算邏輯 ---
    bb_middle = sma(c, bb_period)
    bb_stdev = stddev(c, bb_period)
    bb_upper = bb_middle + bb_stdev * bb_std
    bb_lower = bb_middle - bb_stdev * bb_std

    # --- 核心狀態判斷邏輯 ---
    is_trending = adx > trend_thresh
    bbw_val = (bb_upper - bb_lower) / (bb_middle + epsilon)
    is_expanding = bbw_val > vol_thresh_pct

    state = (
        pl.when(is_trending & is_expanding)
        .then(pl.lit(1, dtype=pl.Int8))
        .when(is_trending & ~is_expanding)
        .then(pl.lit(2, dtype=pl.Int8))
        .when(~is_trending & ~is_expanding)
        .then(pl.lit(3, dtype=pl.Int8))
        .otherwise(pl.lit(4, dtype=pl.Int8))
    )

    return {"type": "event", "values": {"State": state}}


def adapt_Market_State(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將無序的名目狀態 (1, 2, 3, 4) 映射為具有方向性與強度的連續特徵。
    正交分解為：狀態政權水位 (Position)、政權乖離 (Bias)、切換加速度 (Momentum)、狀態混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端切換) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，衰減週期全面可由 YAML 配置。
    """
    state = h_output["values"]["State"]

    # 1. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["adx_len"])
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", params["bb_period"])

    epsilon = 1e-6

    # 【核心工程：狀態降維與連續化映射】
    # 將離散狀態轉化為 "Directionality Score" (方向性評分)
    # 1.0 = 強單邊趨勢, 0.5 = 趨勢收斂中, -0.5 = 穩定盤整, -1.0 = 劇烈洗盤 (無方向高波動)
    state_score = (
        pl.when(state == 1)
        .then(1.0)
        .when(state == 2)
        .then(0.5)
        .when(state == 3)
        .then(-0.5)
        .when(state == 4)
        .then(-1.0)
        .otherwise(0.0)
    ).cast(pl.Float64)

    # ---------------------------------------------------------
    # (A) Position (狀態政權水位): 市場狀態的宏觀衰減中樞
    # 語意補值: 0.0 (市場處於長期的過渡期，無明顯的極端屬性)
    # ---------------------------------------------------------
    regime = state_score.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：約束於 [-0.8, 0.8]，過濾絕對單邊造成的極端固化
    feat_ms_position_stable = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-0.8, 0.8).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0] 的完整理論空間
    feat_ms_position_sensitive = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (狀態政權乖離): 當前微觀狀態相對於長線政權的偏離
    # 語意補值: 0.0 (當前狀態完美符合近期宏觀環境)
    # ---------------------------------------------------------
    bias = state_score - regime

    # Stable 版：約束於 [-1.0, 1.0]
    feat_ms_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-2.0, 2.0]，捕捉從死水盤整 (-1.0) 瞬間切換至主升段 (1.0) 的極端斷層
    feat_ms_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (狀態切換加速度): 狀態評分的變化速度 (一階導數正規化)
    # 語意補值: 0.0 (市場狀態維持不變)
    # 降共線性處理: 減去短線 EMA 並自適應標準化，凸顯變盤瞬間的爆發力
    # ---------------------------------------------------------
    score_ema_micro = state_score.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (state_score - score_ema_micro) / (score_ema_micro.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_ms_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極致捕捉市場環境巨變的動能
    feat_ms_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (狀態混沌度): 市場狀態切換的歷史變異數
    # 語意補值: 0.0 (市場死心塌地維持單一環境)
    # 防禦處理: 強制套用 log1p 平滑
    # 數值飆高代表市場在「趨勢」與「洗盤」之間瘋狂反覆切換
    # ---------------------------------------------------------
    score_volatility = state_score.rolling_std(window_size=adapt_vol_p)
    log_score_vol = score_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_ms_volatility_stable = (
        log_score_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留極端高頻切換的混沌特徵
    feat_ms_volatility_sensitive = (
        log_score_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_ms_position_stable": feat_ms_position_stable,
        "feat_ms_position_sensitive": feat_ms_position_sensitive,
        "feat_ms_bias_stable": feat_ms_bias_stable,
        "feat_ms_bias_sensitive": feat_ms_bias_sensitive,
        "feat_ms_momentum_stable": feat_ms_momentum_stable,
        "feat_ms_momentum_sensitive": feat_ms_momentum_sensitive,
        "feat_ms_volatility_stable": feat_ms_volatility_stable,
        "feat_ms_volatility_sensitive": feat_ms_volatility_sensitive,
    }
