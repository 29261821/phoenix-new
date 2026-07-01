# ==============================================================================
# § 指標 | 全局上下文 v3.0 (150分典範版)
# 核心職責: 將多個跨市場分析指標(如滾動相關性)打包成結構化的跨市場地理學特徵。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| series_a      | H & G | str  | -        | 無 (必填)       | 關聯資產 A 的欄位名稱 |
| series_b      | H & G | str  | -        | 無 (必填)       | 關聯資產 B 的欄位名稱 |
| corr_window   | H & G | int  | 20 ~ 100 | 無 (必填)       | 滾動相關係數的觀察窗口 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | 21              | 用於 Bias (聯動宏觀乖離) 計算的長線 EMA 週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (聯動切換動能) 計算的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | 34              | 用於 Volatility (聯動混沌度) 的滾動標準差週期 |

【特徵工程說明】
- Global Context 將跨市場資產的絕對皮爾森相關係數 ([-1, 1]) 轉化為深度學習特徵。
- 透過 adapt_macro_p 觀察資產聯動性相對於歷史均值的乖離，捕捉資金板塊的脫鉤預警。
- 透過 adapt_vol_p 衡量兩者關係的穩定度，識別適合配對交易的穩定期與極端混沌期。
"""
from typing import Dict

import polars as pl


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留兩個資產間絕對的滾動皮爾森相關係數 (Correlation) 理論值 [-1.0, 1.0]。
    確保依賴跨市場共振 (如 Correlation > 0.8) 的配對交易或宏觀濾網策略能無縫執行。

    契約:
    - df: pl.DataFrame, 必須包含 params['series_a'] 和 params['series_b'] 指定的欄位。
    - params: Dict, 必須包含 'series_a', 'series_b', 'corr_window' 鍵。
    """
    # --- 1. 契約驗證與參數提取 ---
    series_a_col: str = params.get("series_a")
    series_b_col: str = params.get("series_b")
    corr_window: int = params.get("corr_window")

    if not all([series_a_col, series_b_col, corr_window]):
        raise ValueError(
            "Global_Context 的參數 'series_a', 'series_b', 'corr_window' 必須被提供。"
        )

    # 在計算前，驗證所有必需的欄位都存在於 DataFrame 中
    for col in [series_a_col, series_b_col]:
        if col not in df.columns:
            raise ValueError(f"輸入 DataFrame 缺少 Global_Context 所需的欄位: {col}")

    series_a = pl.col(series_a_col)
    series_b = pl.col(series_b_col)

    # --- 2. 核心相關性計算 ---
    # 使用 Polars 內建的高效滾動相關係數函數 `pl.rolling_corr`
    correlation = pl.rolling_corr(series_a, series_b, window_size=corr_window)

    return {"type": "scalar", "values": {"Correlation": correlation}}


def adapt_Global_Context(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將跨市場的相關性轉換為 DL/ML 寬表特徵，挖掘資產聯動的深層微觀動態。
    正交分解為：聯動絕對水位 (Position)、聯動宏觀乖離 (Bias)、聯動切換動能 (Momentum)、聯動混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端脫鉤) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，所有動能、乖離與變異數週期全面可由 YAML 配置。
    """
    corr = h_output["values"]["Correlation"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    # 保留原 ML 專家調校的預設值，同時開放 YAML 覆寫
    adapt_macro_p = params.get("adapt_macro_p", 21)
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (聯動絕對水位): 當前跨市場資產的同步程度
    # 語意補值: 0.0 (代表資產完全脫鉤，走勢無任何線性關聯)
    # 理論上相關係數必落於 [-1.0, 1.0] 之間
    # ---------------------------------------------------------
    # Stable 版 & Sensitive 版：由於先天已約束，兩者均在 [-1.0, 1.0] 內
    feat_global_context_position_stable = (
        corr.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    feat_global_context_position_sensitive = (
        corr.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (聯動宏觀乖離 / 脫鉤預警): 相關性相對於長線均線的偏離
    # 語意補值: 0.0 (代表當前的聯動關係與歷史長線共識完美一致)
    # ---------------------------------------------------------
    corr_ema_macro = corr.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = corr - corr_ema_macro

    # Stable 版：約束於 [-0.5, 0.5]，過濾掉過度極端的關聯偏移，專注於穩定的資金輪動
    feat_global_context_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]，捕捉史詩級的獨立行情啟動 (極端脫鉤)
    feat_global_context_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (聯動切換動能): 相關性的變化速度 (一階導數正規化)
    # 語意補值: 0.0 (兩個資產的關係維持等速發展，無突然的共振或斷裂)
    # 降共線性處理: 減去短線 EMA 並自適應標準化，凸顯突發宏觀事件導致的瞬間板塊碰撞
    # ---------------------------------------------------------
    corr_ema_micro = corr.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (corr - corr_ema_micro) / (corr_ema_micro.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_global_context_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極致捕捉跨市場資金瘋狂切換的動能峰值
    feat_global_context_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (聯動混沌度 / 關係穩定性): 相關係數的歷史變異數
    # 語意補值: 0.0 (關係極度穩定，適合做傳統的配對交易)
    # 防禦處理: 強制套用 log1p 平滑極端的變異數爆炸
    # ---------------------------------------------------------
    corr_volatility = corr.rolling_std(window_size=adapt_vol_p)
    log_corr_vol = corr_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_global_context_volatility_stable = (
        log_corr_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留跨市場資金博弈時造成的極端混沌關係特徵
    feat_global_context_volatility_sensitive = (
        log_corr_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_global_context_position_stable": feat_global_context_position_stable,
        "feat_global_context_position_sensitive": feat_global_context_position_sensitive,
        "feat_global_context_bias_stable": feat_global_context_bias_stable,
        "feat_global_context_bias_sensitive": feat_global_context_bias_sensitive,
        "feat_global_context_momentum_stable": feat_global_context_momentum_stable,
        "feat_global_context_momentum_sensitive": feat_global_context_momentum_sensitive,
        "feat_global_context_volatility_stable": feat_global_context_volatility_stable,
        "feat_global_context_volatility_sensitive": feat_global_context_volatility_sensitive,
    }
