# ==============================================================================
# § 指標 | 沙夫趨勢週期 (Schaff Trend Cycle, STC)
# 核心職責: 結合 MACD 的趨勢性與 Stochastic 的靈敏度，捕捉市場的週期性循環。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| source        | H & G | str  | -        | 無 (必填)       | 價格來源 (如 'close') |
| fast          | H & G | int  | 12 ~ 26  | 無 (必填)       | MACD 快線週期 |
| slow          | H & G | int  | 26 ~ 50  | 無 (必填)       | MACD 慢線週期 |
| cycle         | H & G | int  | 10 ~ 20  | 無 (必填)       | 隨機指標 (%K) 的回顧週期 |
| d1            | H & G | int  | 3 ~ 10   | 無 (必填)       | 隨機指標 (%D) 的平滑週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | cycle 參數值    | 用於 Bias (週期循環乖離) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | d1 參數值       | 用於 Momentum (脫離鈍化加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 13 ~ 34  | cycle 參數值    | 用於 Volatility (循環混沌度) 的滾動標準差週期 |

【特徵工程說明】
- STC 原始輸出為 0~100 的雙重平滑指標。G 接口將其中心化並縮放至 [-1.0, 1.0]。
- STC 的特性是極易在超買(>75)與超賣(<25)區鈍化成直線。
- 透過 adapt_micro_p 計算的 Momentum，能極致敏銳地捕捉 STC「剛離開平頂/平底」的瞬間爆發力。
"""
from typing import Dict

import polars as pl

from src.features.functions.ema import calculate as ema
from src.features.functions.highest import calculate as highest
from src.features.functions.lowest import calculate as lowest
from src.features.functions.shift import calculate as prev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的 STC 數值 (0~100)。
    供傳統量化腳本判斷超買超賣 (如 STC > 75 視為超買，向下彎折視為作空信號)。

    契約：
    - df 必須包含 params['source'] 指定的欄位。
    - params 必須包含 'source', 'fast', 'slow', 'cycle', 'd1' 鍵。
    """
    source_col, fast, slow, cycle, d1 = (
        params["source"],
        params["fast"],
        params["slow"],
        params["cycle"],
        params["d1"],
    )
    source = pl.col(source_col)
    epsilon = 1e-9

    # --- 1. 計算 MACD ---
    fast_ma = ema(series=source, length=fast)
    slow_ma = ema(series=source, length=slow)
    macd_val = fast_ma - slow_ma

    # --- 2. 第一次隨機指標計算 (Stochastic of MACD) ---
    lowest_macd = lowest(series=macd_val, period=cycle)
    highest_macd = highest(series=macd_val, period=cycle)

    stoch_k1_raw = pl.when(highest_macd > lowest_macd).then(
        100 * (macd_val - lowest_macd) / (highest_macd - lowest_macd + epsilon)
    )
    stoch_k1 = stoch_k1_raw.forward_fill().fill_null(50)

    stoch_d1 = ema(series=stoch_k1, length=d1)

    # --- 3. 第二次隨機指標計算 (Stochastic of %D1) ---
    lowest_d1 = lowest(series=stoch_d1, period=cycle)
    highest_d1 = highest(series=stoch_d1, period=cycle)

    stoch_k2_raw = pl.when(highest_d1 > lowest_d1).then(
        100 * (stoch_d1 - lowest_d1) / (highest_d1 - lowest_d1 + epsilon)
    )
    stoch_k2 = stoch_k2_raw.forward_fill().fill_null(50)

    # --- 4. 最終 STC 值 ---
    stc_val = ema(series=stoch_k2, length=d1)

    return {"type": "scalar", "values": {"STC": stc_val}}


def adapt_Schaff_Trend_Cycle(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將 [0, 100] 的雙重平滑循環指標轉換為 [-1, 1] 的無量綱時空特徵。
    正交分解為：週期循環水位 (Position)、循環宏觀乖離 (Bias)、脫離鈍化加速度 (Momentum)、循環混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    stc_val = h_output["values"]["STC"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["cycle"])
    adapt_micro_p = params.get("adapt_micro_p", params["d1"])
    adapt_vol_p = params.get("adapt_vol_p", params["cycle"])

    epsilon = 1e-6

    # 【核心：中心化與縮放】
    # 將 0~100 的 STC 映射為 -1.0 ~ 1.0 的對稱空間
    centered_stc = (stc_val - 50.0) / 50.0

    # ---------------------------------------------------------
    # (A) Position (週期循環水位): STC 的絕對相對位置
    # 語意補值: 0.0 (代表循環處於多空轉換的 50 中線)
    # ---------------------------------------------------------
    # Stable 版：嚴格約束於 [-1.0, 1.0]
    feat_stc_position_stable = (
        centered_stc.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：略微放寬至 [-1.2, 1.2]，包容數值微小抖動
    feat_stc_position_sensitive = (
        centered_stc.fill_nan(0.0).fill_null(0.0).clip(-1.2, 1.2).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (循環宏觀乖離): 循環信號相對於其長線政權的背離
    # 語意補值: 0.0 (當前循環週期與近期宏觀頻率一致)
    # ---------------------------------------------------------
    stc_ema_macro = centered_stc.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = centered_stc - stc_ema_macro

    # Stable 版：約束於 [-1.0, 1.0]
    feat_stc_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-2.0, 2.0]
    feat_stc_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (脫離鈍化加速度): 循環信號的變化速度 (一階導數)
    # 語意補值: 0.0 (處於平頂、平底的鈍化期，或等速移動中)
    # 降共線性處理: 減去短線 EMA 並自適應標準化，極致捕捉 STC 從鈍化區彎折的瞬間
    # ---------------------------------------------------------
    ema_centered_stc = centered_stc.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (centered_stc - ema_centered_stc) / (ema_centered_stc.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0]
    feat_stc_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯變盤瞬間的強大反轉加速度
    feat_stc_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (循環混沌度): STC 信號的歷史變異數
    # 語意補值: 0.0 (循環極度規律，或處於長期的平頂/平底鈍化)
    # ---------------------------------------------------------
    stc_vol = centered_stc.rolling_std(window_size=adapt_vol_p)
    log_stc_vol = stc_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_stc_volatility_stable = (
        log_stc_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]
    feat_stc_volatility_sensitive = (
        log_stc_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_stc_position_stable": feat_stc_position_stable,
        "feat_stc_position_sensitive": feat_stc_position_sensitive,
        "feat_stc_bias_stable": feat_stc_bias_stable,
        "feat_stc_bias_sensitive": feat_stc_bias_sensitive,
        "feat_stc_momentum_stable": feat_stc_momentum_stable,
        "feat_stc_momentum_sensitive": feat_stc_momentum_sensitive,
        "feat_stc_volatility_stable": feat_stc_volatility_stable,
        "feat_stc_volatility_sensitive": feat_stc_volatility_sensitive,
    }
