# ==============================================================================
# § 指標 | 信號精煉引擎 (Signal Refinement Engine)
# 核心職責: 對動能信號(如RSI)進行平滑降噪與精煉，消除虛假突破。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口連續化降維特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| rsi_source    | H & G | str  | -        | 無 (必填)       | 價格來源 (如 'close') |
| rsi_period    | H & G | int  | 7 ~ 21   | 無 (必填)       | 底層 RSI 的計算週期 |
| refine_period | H & G | int  | 3 ~ 10   | 無 (必填)       | 信號精煉 (EMA) 的平滑週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | rsi_period 參數 | 用於 Bias (信號宏觀乖離) 的長線衰減週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (信號加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 13 ~ 34  | refine_period 參數| 用於 Volatility (信號混沌度) 的滾動標準差週期 |

【特徵工程說明】
- Refinement 輸出的本質是 [0, 100] 的平滑 RSI。G 接口將其中心化並縮放至 [-1.0, 1.0]。
- 透過 adapt_micro_p 計算精煉信號的二階導數 (加速度)，能夠在趨勢反轉初期極早給出預警。
"""
from typing import Dict

import polars as pl

# [邏輯自治] 遵循 DSL v5.0 (邏輯自治版) 的設計思想，此指標為「自產自銷」。
from src.features.functions.ema import calculate as ema
from src.features.functions.shift import calculate as prev
from src.features.functions.wma import calculate as wma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的精煉數值 (0~100)。
    供傳統量化腳本作為過濾器 (如 Refined > 70 視為確認超買) 調用。

    契約：
    - df 必須包含 params['rsi_source'] 指定的欄位。
    - params 必須包含 'rsi_source', 'rsi_period', 'refine_period' 鍵。
    """
    rsi_source_col, rsi_period, refine_period = (
        params["rsi_source"],
        params["rsi_period"],
        params["refine_period"],
    )
    rsi_source = pl.col(rsi_source_col)
    epsilon = 1e-9

    # --- 內化的 RSI 計算邏輯 ---
    delta = rsi_source - prev(series=rsi_source, period=1)
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)
    avg_gain = wma(series=gain, length=rsi_period)
    avg_loss = wma(series=loss, length=rsi_period)
    rs = avg_gain / (avg_loss + epsilon)
    rsi_val = 100 - (100 / (1 + rs))

    # --- 核心精煉邏輯 (簡化濾波器) ---
    refined_signal = ema(series=rsi_val, length=refine_period)

    return {"type": "scalar", "values": {"Refined": refined_signal}}


def adapt_Refinement(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將 [0, 100] 的平滑動能指標轉換為 [-1, 1] 的無量綱時空特徵。
    正交分解為：精煉動能水位 (Position)、精煉信號乖離 (Bias)、信號加速度 (Momentum)、信號混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    refined = h_output["values"]["Refined"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["rsi_period"])
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", params["refine_period"])

    epsilon = 1e-6

    # 【核心：中心化與縮放】
    # 將 0~100 映射為 -1.0 ~ 1.0 的對稱空間，0 代表中立
    centered_ref = (refined - 50.0) / 50.0

    # ---------------------------------------------------------
    # (A) Position (精煉動能水位): 信號的絕對相對位置
    # 語意補值: 0.0 (代表動能處於多空平衡的 50 中樞)
    # ---------------------------------------------------------
    # Stable 版：嚴格約束於 [-1.0, 1.0]
    feat_refinement_position_stable = (
        centered_ref.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：略微放寬至 [-1.2, 1.2]，包容數值微小抖動
    feat_refinement_position_sensitive = (
        centered_ref.fill_nan(0.0).fill_null(0.0).clip(-1.2, 1.2).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (精煉信號乖離): 信號相對於其長線政權的背離
    # 語意補值: 0.0 (當前動能與近期宏觀動能一致)
    # ---------------------------------------------------------
    ref_ema_macro = centered_ref.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = centered_ref - ref_ema_macro

    # Stable 版：約束於 [-0.5, 0.5]
    feat_refinement_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]
    feat_refinement_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (信號加速度): 精煉信號的變化速度 (一階導數)
    # 語意補值: 0.0 (動能發展維持等速)
    # ---------------------------------------------------------
    ema_centered_ref = centered_ref.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (centered_ref - ema_centered_ref) / (ema_centered_ref.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0]
    feat_refinement_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯變盤瞬間強大的反轉加速度
    feat_refinement_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (信號混沌度): 精煉信號的歷史變異數
    # 語意補值: 0.0 (動能維持單向且平穩的推進)
    # ---------------------------------------------------------
    ref_vol = centered_ref.rolling_std(window_size=adapt_vol_p)
    log_ref_vol = ref_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_refinement_volatility_stable = (
        log_ref_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]
    feat_refinement_volatility_sensitive = (
        log_ref_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_refinement_position_stable": feat_refinement_position_stable,
        "feat_refinement_position_sensitive": feat_refinement_position_sensitive,
        "feat_refinement_bias_stable": feat_refinement_bias_stable,
        "feat_refinement_bias_sensitive": feat_refinement_bias_sensitive,
        "feat_refinement_momentum_stable": feat_refinement_momentum_stable,
        "feat_refinement_momentum_sensitive": feat_refinement_momentum_sensitive,
        "feat_refinement_volatility_stable": feat_refinement_volatility_stable,
        "feat_refinement_volatility_sensitive": feat_refinement_volatility_sensitive,
    }
