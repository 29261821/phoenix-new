# ==============================================================================
# § 指標 | 滾動協整 (Rolling Cointegration)
# 核心職責: 計算主商品與一個或多個基準資產之間的滾動協整關係 (統計套利基礎)。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口動態展開特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| series_a      | H & G | str  | -        | 無 (必填)       | 關聯資產 A (主資產) 的欄位名稱 |
| series_b      | H & G | str  | -        | 無 (可選)       | 關聯資產 B 的欄位名稱 |
| series_c      | H & G | str  | -        | 無 (可選)       | 關聯資產 C 的欄位名稱 |
| series_d      | H & G | str  | -        | 無 (可選)       | 關聯資產 D 的欄位名稱 |
| window        | H & G | int  | 20 ~ 100 | 無 (必填)       | 滾動協整關係的觀察窗口 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | window 參數值   | 用於 Bias (協整宏觀乖離) 計算的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 13   | 5               | 用於 Momentum (協整翻轉加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | window 參數值   | 用於 Volatility (協整關係混沌度) 的滾動標準差週期 |

【特徵工程說明】
- H 接口輸出的 Spread Z-Score 已經具備了良好的無尺度與均值回歸特性。
- G 接口會動態遍歷所有有效配對 (B, C, D)，並以其 Spread Z-Score 為基底，正交分解為：
  協整絕對水位 (Position)、協整宏觀乖離 (Bias)、協整翻轉加速度 (Momentum)、協整混沌度 (Volatility)。
"""
from typing import Dict

import polars as pl

from src.features.functions.sma import calculate as sma
from src.features.functions.stddev import calculate as stddev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算主商品與多個基準資產間的動態避險比例 (Hedge Ratio) 與價差 Z-Score。
    保留絕對的 Z-Score，供傳統統計套利腳本 (如 Z-Score > 2 做空價差) 無縫執行。

    契約：
    - df 必須包含 params['series_a'] 和 params 中其他 series_* 指定的欄位。
    - params 必須包含 'series_a', 'window' 鍵。
    """
    series_a_col = params["series_a"]
    window = params["window"]
    epsilon = 1e-9
    series_a = pl.col(series_a_col)

    outputs = {}

    for series_key in ["series_b", "series_c", "series_d"]:
        suffix = series_key[-1]  # 'b', 'c', or 'd'
        if series_key in params and params[series_key] is not None:
            series_other_col = params[series_key]
            series_other = pl.col(series_other_col)

            # 使用 Polars 內建函數進行高效計算
            cov_ao = pl.rolling_cov(series_a, series_other, window_size=window)
            var_other = stddev(series=series_other, period=window).pow(2)

            hedge_ratio = cov_ao / (var_other + epsilon)
            spread = series_a - (hedge_ratio * series_other)

            spread_mean = sma(series=spread, length=window)
            spread_std = stddev(series=spread, period=window)

            z_score = (spread - spread_mean) / (spread_std + epsilon)

            outputs[f"HedgeRatio_{suffix}"] = hedge_ratio
            outputs[f"Spread_ZScore_{suffix}"] = z_score
        # 【核心修復】：移除 else 區段，不產生 pl.lit(None)。
        # 交由 adapt 階段的 `if z_key not in h_output["values"]: continue` 自然略過。

    return {"type": "vector", "values": outputs}


def adapt_Rolling_Cointegration(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    動態遍歷 H 接口提取出的所有 Spread Z-Score，並將其轉換為動態時空特徵。
    正交分解為：協整絕對水位 (Position)、協整宏觀乖離 (Bias)、協整翻轉加速度 (Momentum)、協整混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統。特徵名稱會根據有效資產配對動態生成。
    """
    window = params["window"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", window)
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", window)

    epsilon = 1e-6
    adapted_features = {}

    # 動態處理所有的有效配對
    for suffix in ["b", "c", "d"]:
        z_key = f"Spread_ZScore_{suffix}"
        if z_key not in h_output["values"]:
            continue

        z_score_val = h_output["values"][z_key].cast(pl.Float64)

        # --- [核心修復點：移除 Expr 的直接 if 判斷] ---
        # 避免觸發 `the truth value of an Expr is ambiguous` 錯誤
        # 零方差與全 Null 過濾留待 FeatureExecutor 取出 DataFrame 後統一處理

        prefix = f"feat_coint_{suffix}"

        # ---------------------------------------------------------
        # (A) Position (協整絕對水位): 價差的 Z-Score 本身就是最佳的水位特徵
        # 語意補值: 0.0 (代表價差處於歷史均值，無套利空間)
        # ---------------------------------------------------------
        # Stable 版：約束於 [-3.0, 3.0]
        adapted_features[f"{prefix}_position_stable"] = (
            z_score_val.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
        )
        # Sensitive 版：放寬至 [-6.0, 6.0]，捕捉黑天鵝級別的極端脫鉤
        adapted_features[f"{prefix}_position_sensitive"] = (
            z_score_val.fill_nan(0.0).fill_null(0.0).clip(-6.0, 6.0).cast(pl.Float64)
        )

        # ---------------------------------------------------------
        # (B) Bias (協整宏觀乖離): 當前 Z-Score 相對於近期均線的偏離
        # 語意補值: 0.0 (價差回歸的軌跡符合近期宏觀慣性)
        # ---------------------------------------------------------
        z_ema_macro = z_score_val.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
        bias = z_score_val - z_ema_macro

        # Stable 版：約束於 [-1.0, 1.0]
        adapted_features[f"{prefix}_bias_stable"] = (
            bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
        )
        # Sensitive 版：約束於 [-3.0, 3.0]
        adapted_features[f"{prefix}_bias_sensitive"] = (
            bias.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
        )

        # ---------------------------------------------------------
        # (C) Momentum (協整翻轉加速度): Z-Score 的變化速度
        # 語意補值: 0.0 (價差收斂或擴張的速度維持等速)
        # ---------------------------------------------------------
        ema_z_micro = z_score_val.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
        momentum = (z_score_val - ema_z_micro) / (ema_z_micro.abs() + epsilon)

        # Stable 版：嚴格約束 [-1.0, 1.0]
        adapted_features[f"{prefix}_momentum_stable"] = (
            momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
        )
        # Sensitive 版：放寬約束至 [-5.0, 5.0]
        adapted_features[f"{prefix}_momentum_sensitive"] = (
            momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
        )

        # ---------------------------------------------------------
        # (D) Volatility (協整關係混沌度): Z-Score 的滾動變異數
        # 語意補值: 0.0 (兩者價差極度平穩，呈現完美的同調)
        # ---------------------------------------------------------
        coint_volatility = z_score_val.rolling_std(window_size=adapt_vol_p)
        log_coint_vol = coint_volatility.log1p()

        # Stable 版：約束於 [0.0, 1.0]
        adapted_features[f"{prefix}_volatility_stable"] = (
            log_coint_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
        )
        # Sensitive 版：約束於 [0.0, 2.0]
        adapted_features[f"{prefix}_volatility_sensitive"] = (
            log_coint_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 2.0).cast(pl.Float64)
        )

    # 若全無有效配對，確保回傳空字典，Executor 會安全略過
    return adapted_features
