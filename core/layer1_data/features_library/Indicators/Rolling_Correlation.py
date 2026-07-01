# ==============================================================================
# § 指標 | 滾動相關性 (Rolling Correlation)
# 核心職責: 計算多個資產間在滾動窗口內的線性相關係數，監控跨市場板塊輪動與脫鉤。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口動態展開特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| series_a      | H & G | str  | -        | 無 (必填)       | 關聯資產 A (主資產) 的欄位名稱 |
| series_b      | H & G | str  | -        | 無 (可選)       | 關聯資產 B 的欄位名稱 |
| series_c      | H & G | str  | -        | 無 (可選)       | 關聯資產 C 的欄位名稱 |
| window        | H & G | int  | 20 ~ 100 | 無 (必填)       | 滾動相關係數的觀察窗口 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | window 參數值   | 用於 Bias (關聯脫鉤乖離) 計算的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 13   | 5               | 用於 Momentum (板塊切換加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | window 參數值   | 用於 Volatility (關聯混沌度) 的滾動標準差週期 |

【特徵工程說明】
- 相關係數天生具備 [-1.0, 1.0] 的完美邊界，無需進行 Z-Score 降維。
- G 接口會動態遍歷所有有效配對 (B, C)，並正交分解為：
  相關性絕對水位 (Position)、相關性脫鉤乖離 (Bias)、板塊切換加速度 (Momentum)、關聯混沌度 (Volatility)。
"""
from typing import Dict

import polars as pl


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算主商品與多個基準資產間的皮爾森相關係數 (Pearson Correlation)。
    保留絕對的相關係數 [-1.0, 1.0]，供傳統宏觀濾網腳本無縫執行。

    契約：
    - df 必須包含 params['series_a'] 和 params 中其他 series_* 指定的欄位。
    - params 必須包含 'series_a', 'window' 鍵。
    - series_b, series_c 是可選的。
    """
    series_a_col = params["series_a"]
    window = params["window"]
    series_a = pl.col(series_a_col)

    outputs = {}

    for series_key in ["series_b", "series_c"]:
        suffix = series_key[-1]
        if series_key in params and params[series_key] is not None:
            series_other_col = params[series_key]
            series_other = pl.col(series_other_col)

            # 使用 Polars 內建的高效滾動相關係數函數
            correlation = pl.rolling_corr(series_a, series_other, window_size=window)
            outputs[f"Corr_{suffix}"] = correlation
        # 【核心修復】：移除 else 區段，不要回傳 pl.lit(None)。
        # 不存在的資產參數，就不要產生對應的 key，避免 adapt 填補後產生整列 0.0 的零方差特徵！

    return {"type": "vector", "values": outputs}


def adapt_Rolling_Correlation(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    動態遍歷 H 接口提取出的所有 Correlation，並將其轉換為動態時空特徵。
    正交分解為：絕對水位 (Position)、脫鉤乖離 (Bias)、切換加速度 (Momentum)、關聯混沌度 (Volatility)。
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
    for suffix in ["b", "c"]:
        k = f"Corr_{suffix}"
        if k not in h_output["values"]:
            continue

        corr_val = h_output["values"][k].cast(pl.Float64)

        # --- [核心修復點：移除 Expr 的直接 if 判斷] ---
        # 避免觸發 `the truth value of an Expr is ambiguous` 錯誤
        # 零方差與全 Null 過濾留待 FeatureExecutor 取出 DataFrame 後統一處理

        prefix = f"feat_rolling_corr_{suffix}"

        # ---------------------------------------------------------
        # (A) Position (相關性絕對水位): 相關係數天生的 [-1.0, 1.0]
        # 語意補值: 0.0 (代表資產間毫無線性關聯)
        # ---------------------------------------------------------
        # Stable 版與 Sensitive 版先天已約束
        adapted_features[f"{prefix}_position_stable"] = (
            corr_val.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
        )
        adapted_features[f"{prefix}_position_sensitive"] = (
            corr_val.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
        )

        # ---------------------------------------------------------
        # (B) Bias (關聯脫鉤乖離): 當前相關係數相對於近期均線的偏離
        # 語意補值: 0.0 (關聯性符合近期宏觀慣性)
        # ---------------------------------------------------------
        corr_ema_macro = corr_val.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
        bias = corr_val - corr_ema_macro

        # Stable 版：約束於 [-0.5, 0.5]
        adapted_features[f"{prefix}_bias_stable"] = (
            bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
        )
        # Sensitive 版：約束於 [-1.0, 1.0]，捕捉史詩級的獨立行情啟動 (極端脫鉤)
        adapted_features[f"{prefix}_bias_sensitive"] = (
            bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
        )

        # ---------------------------------------------------------
        # (C) Momentum (板塊切換加速度): 相關係數的變化速度
        # 語意補值: 0.0 (關聯性維持等速)
        # ---------------------------------------------------------
        ema_corr_micro = corr_val.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
        momentum = (corr_val - ema_corr_micro) / (ema_corr_micro.abs() + epsilon)

        # Stable 版：嚴格約束 [-1.0, 1.0]
        adapted_features[f"{prefix}_momentum_stable"] = (
            momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
        )
        # Sensitive 版：放寬約束至 [-5.0, 5.0]，極致捕捉跨市場資金瘋狂切換的動能
        adapted_features[f"{prefix}_momentum_sensitive"] = (
            momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
        )

        # ---------------------------------------------------------
        # (D) Volatility (關聯混沌度): 相關係數的滾動變異數
        # 語意補值: 0.0 (兩者關係極度平穩，呈現完美的同調或背離)
        # 防禦處理: 強制套用 log1p 平滑
        # ---------------------------------------------------------
        corr_volatility = corr_val.rolling_std(window_size=adapt_vol_p)
        log_corr_vol = corr_volatility.log1p()

        # Stable 版：約束於 [0.0, 0.5]
        adapted_features[f"{prefix}_volatility_stable"] = (
            log_corr_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
        )
        # Sensitive 版：約束於 [0.0, 1.0]
        adapted_features[f"{prefix}_volatility_sensitive"] = (
            log_corr_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
        )

    # 若全無有效配對，確保回傳空字典，Executor 會安全略過
    return adapted_features
