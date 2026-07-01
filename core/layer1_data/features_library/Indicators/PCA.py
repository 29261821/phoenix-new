# ==============================================================================
# § 指標 | 主成分分析引擎 v4.0 (Join對齊最終版)
# 核心職責: 根據【第三邊：統計套利】作戰計畫，實現因子分析與降維。
# v4.0 更新: [根本性重構] 徹底廢除有問題的 Series.set() 方法，採用 DataFrame.join。
# v5.0 更新: [架構升級] 導入 H 接口惰性封裝，以及 G 接口動態主成分正交特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| n_components  | H & G | int  | 1 ~ 3    | 無 (必填)       | 要提取的主成分數量 |
| ao_fast       | H     | int  | 3 ~ 10   | 5               | AO 的快線週期 |
| ao_slow       | H     | int  | 20 ~ 50  | 34              | AO 的慢線週期 |
| cmf_period    | H     | int  | 10 ~ 30  | 20              | CMF 的計算週期 |
| rsi_period    | H     | int  | 7 ~ 21   | 14              | RSI 的計算週期 |
| rsi_source    | H     | str  | -        | "close"         | RSI 的價格來源 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | 55              | 用於 Bias (因子乖離) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 13   | 5               | 用於 Momentum (因子加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | 34              | 用於 Volatility (因子混沌度) 的滾動標準差週期 |

【特徵工程說明】
- PCA 產出的主成分 (PC1, PC2...) 在全域樣本上均值為 0，變異數為 1。
- G 接口會動態遍歷所有提取出的 PC，並將其轉換為時間序列的動態正交特徵 (Position, Bias, Momentum, Volatility)。
"""
from typing import Dict

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    對內部計算的特徵集 (AO, CMF, RSI) 執行主成分分析 (PCA)。
    本指標屬於「 eager a.k.a. non-expression-based 」類型。
    [契約修復]: 將 Eager 產出的 Series 透過 pl.lit() 封裝為 pl.Expr，完美對接惰性計算圖。

    契約：
    - df: pl.DataFrame, 原始的 OHLCV 數據。
    - params: Dict, 必須包含 'n_components'。
    """
    # --- 1. 契約驗證與參數提取 ---
    n_components: int = params.get("n_components")
    if not n_components:
        raise ValueError("PCA_Engine 的參數 'n_components' 必須被提供。")

    ao_fast = params.get("ao_fast", 5)
    ao_slow = params.get("ao_slow", 34)
    cmf_period = params.get("cmf_period", 20)
    rsi_period = params.get("rsi_period", 14)
    rsi_source = params.get("rsi_source", "close")

    # --- 2. 內部計算依賴特徵 (Polars 表達式) ---
    median_price = (pl.col("high") + pl.col("low")) / 2
    ao_expr = (
        median_price.rolling_mean(window_size=ao_fast)
        - median_price.rolling_mean(window_size=ao_slow)
    ).alias("AO_internal")

    mfm_expr = (
        (pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))
    ) / (
        pl.when(pl.col("high") == pl.col("low"))
        .then(1)
        .otherwise(pl.col("high") - pl.col("low"))
    )
    mfv_expr = mfm_expr.fill_null(0) * pl.col("volume")
    sum_mfv = mfv_expr.rolling_sum(window_size=cmf_period)
    sum_vol = pl.col("volume").rolling_sum(window_size=cmf_period)
    cmf_expr = (pl.when(sum_vol == 0).then(None).otherwise(sum_mfv / sum_vol)).alias(
        "CMF_internal"
    )

    delta = pl.col(rsi_source).diff()
    gain = delta.clip(lower_bound=0).fill_null(0)
    loss = -delta.clip(upper_bound=0).fill_null(0)
    avg_gain = gain.ewm_mean(span=rsi_period, adjust=False)
    avg_loss = loss.ewm_mean(span=rsi_period, adjust=False)
    rs = pl.when(avg_loss == 0).then(None).otherwise(avg_gain / avg_loss)
    rsi_expr = (100 - (100 / (1 + rs))).fill_null(100).alias("RSI_internal")

    # --- 3. 數據準備與遮罩生成 ---
    features_df = df.select(
        ao_expr,
        cmf_expr,
        rsi_expr,
    )
    features_filled_df = features_df.with_columns(pl.all().forward_fill())
    valid_rows_mask = features_filled_df.select(
        pl.all_horizontal(pl.all().is_not_null())
    ).to_series()
    valid_feature_values = features_filled_df.filter(valid_rows_mask).to_numpy()

    output_values = {}
    if valid_feature_values.shape[0] < n_components:
        nan_series = pl.Series(values=np.nan, length=len(df), dtype=pl.Float64)
        for i in range(n_components):
            output_values[f"PC{i+1}"] = pl.lit(nan_series.alias(f"PC{i+1}"))
        return {"type": "vector", "values": output_values}

    # --- 4. 核心 PCA 計算 (Scikit-learn) ---
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(valid_feature_values)
    pca = PCA(n_components=n_components)
    principal_components_values = pca.fit_transform(scaled_values)

    # --- 5. 結果對齊 (Join-Based Alignment) ---
    original_indices = pl.Series("index", np.arange(len(df))).filter(valid_rows_mask)
    base_df = pl.DataFrame({"index": np.arange(len(df))})

    for i in range(n_components):
        pc_name = f"PC{i+1}"
        payload_df = pl.DataFrame(
            {"index": original_indices, pc_name: principal_components_values[:, i]}
        )
        final_df = base_df.join(payload_df, on="index", how="left")

        # 封裝為 pl.Expr 以滿足合約
        output_values[pc_name] = pl.lit(final_df.get_column(pc_name))

    return {"type": "vector", "values": output_values}


def adapt_PCA(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    動態遍歷 H 接口提取出的所有主成分 (PC1, PC2...)，並將其轉換為動態時空特徵。
    正交分解為：潛在因子水位 (Position)、因子乖離 (Bias)、切換加速度 (Momentum)、因子混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統。特徵名稱會根據 PC 動態生成。
    """
    n_components = params.get("n_components", 1)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", 55)
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    epsilon = 1e-6
    adapted_features = {}

    # 動態處理所有的 Principal Components
    for i in range(1, n_components + 1):
        pc_key = f"PC{i}"
        if pc_key not in h_output["values"]:
            continue

        pc_val = h_output["values"][pc_key].cast(pl.Float64)
        prefix = f"feat_pca_pc{i}"

        # ---------------------------------------------------------
        # (A) Position (潛在因子水位): PC 已經是全域標準化的值
        # 語意補值: 0.0 (代表該維度特徵處於市場共識的絕對中立區)
        # ---------------------------------------------------------
        # Stable 版：約束於 [-3.0, 3.0]
        adapted_features[f"{prefix}_position_stable"] = (
            pc_val.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
        )
        # Sensitive 版：放寬至 [-5.0, 5.0]，捕捉極端的黑天鵝離群狀態
        adapted_features[f"{prefix}_position_sensitive"] = (
            pc_val.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
        )

        # ---------------------------------------------------------
        # (B) Bias (因子宏觀乖離): 當前 PC 相對於近期均線的偏離
        # 語意補值: 0.0 (潛在因子特性與近期宏觀一致)
        # ---------------------------------------------------------
        pc_ema_macro = pc_val.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
        bias = pc_val - pc_ema_macro

        # Stable 版：約束於 [-1.0, 1.0]
        adapted_features[f"{prefix}_bias_stable"] = (
            bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
        )
        # Sensitive 版：約束於 [-3.0, 3.0]
        adapted_features[f"{prefix}_bias_sensitive"] = (
            bias.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
        )

        # ---------------------------------------------------------
        # (C) Momentum (因子切換加速度): PC 的變化速度
        # 語意補值: 0.0 (該維度因子維持等速發展)
        # ---------------------------------------------------------
        ema_pc_micro = pc_val.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
        momentum = (pc_val - ema_pc_micro) / (ema_pc_micro.abs() + epsilon)

        # Stable 版：嚴格約束 [-1.0, 1.0]
        adapted_features[f"{prefix}_momentum_stable"] = (
            momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
        )
        # Sensitive 版：放寬約束至 [-5.0, 5.0]
        adapted_features[f"{prefix}_momentum_sensitive"] = (
            momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
        )

        # ---------------------------------------------------------
        # (D) Volatility (因子混沌度): PC 的滾動變異數
        # 語意補值: 0.0 (潛在因子的狀態死心塌地維持不變)
        # ---------------------------------------------------------
        pc_volatility = pc_val.rolling_std(window_size=adapt_vol_p)
        log_pc_vol = pc_volatility.log1p()

        # Stable 版：約束於 [0.0, 1.0]
        adapted_features[f"{prefix}_volatility_stable"] = (
            log_pc_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
        )
        # Sensitive 版：約束於 [0.0, 2.0]
        adapted_features[f"{prefix}_volatility_sensitive"] = (
            log_pc_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 2.0).cast(pl.Float64)
        )

    return adapted_features
