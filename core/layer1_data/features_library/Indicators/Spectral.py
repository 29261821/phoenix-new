# ==============================================================================
# § 指標 | 頻譜分析引擎 (Spectral Engine) v5.0
# 核心職責: 利用快速傅立葉轉換或類似頻域演算法，提取市場當前的主導週期。
# v5.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口頻域無尺度化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| input_feature | H & G | str  | -        | 無 (必填)       | 頻譜分析的目標欄位 (如 'close') |
| window_size   | H & G | int  | 60 ~ 200 | 無 (必填)       | 滾動頻譜分析的總觀察視窗大小 |
| min_period    | H & G | int  | 10 ~ 20  | 無 (必填)       | 搜索的最小週期限制 |
| max_period    | H & G | int  | 40 ~ 100 | 無 (必填)       | 搜索的最大週期限制 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | window_size 參數| 用於 Bias (週期循環乖離) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (頻率切換加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | 34              | 用於 Volatility (頻譜混沌度) 的滾動標準差週期 |

【特徵工程說明】
- 主導週期帶有「時間絕對尺度」(如 30 天)。G 接口利用 min_period 與 max_period
  將其標準化為「相對頻率水位」[-1.0, 1.0]。
- -1.0 代表市場陷入極短線高頻雜訊；1.0 代表市場處於長週期趨勢中。
"""
from typing import Any, Dict

import polars as pl

# --- [v5.0 核心升級] ---
# 導入新一代的、基於表達式的原生核心接口
from src.features.rust_functions import rolling_spectral_expr


def calculate(df: pl.DataFrame, params: Dict[str, Any], **kwargs) -> Dict[str, pl.Expr]:
    """
    v5.0 (原生表達式版):
    - [根本性架構回歸] 返回類型從 `pl.Series` 回歸到 `pl.Expr`。
    - [性能與契約統一] 將核心計算任務，委託給 `rolling_spectral_expr`，
      返回一個完整的計算圖 (`pl.Expr`)。

    契約：
    - 返回標準的 H 接口字典 (含 'type' 與 'values')，交由 executor 的標準惰性模式處理。
    """
    # --- 1. 契約驗證與參數提取 ---
    feature_col: str = params.get("input_feature")
    window_size: int = params.get("window_size")
    min_period: int = params.get("min_period")
    max_period: int = params.get("max_period")

    if not all([feature_col, window_size, min_period, max_period]):
        raise ValueError(
            "Spectral_Engine 的參數 'input_feature', 'window_size', 'min_period', 'max_period' 必須被提供。"
        )
    if feature_col not in df.columns:
        raise ValueError(f"DataFrame 中缺少頻譜分析所需的欄位: {feature_col}")

    # --- 2. 數據準備 (表達式) ---
    input_expr = pl.col(feature_col).forward_fill()

    # --- 3. 授權原生核心構建計算圖 ---
    spectral_expr = rolling_spectral_expr(
        expr=input_expr,
        window_size=window_size,
        min_period=min_period,
        max_period=max_period,
    )

    # --- 4. 格式化輸出 ---
    return {"type": "scalar", "values": {"DominantPeriod": spectral_expr}}


def adapt_Spectral(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將具備絕對時間尺度的「主導週期」轉換為 [-1, 1] 的無量綱頻譜特徵。
    正交分解為：相對頻率水位 (Position)、週期循環乖離 (Bias)、頻率切換加速度 (Momentum)、頻譜混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    dom_period = h_output["values"]["DominantPeriod"].cast(pl.Float64)

    # 1. 提取基礎參數
    min_period = params["min_period"]
    max_period = params["max_period"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["window_size"])
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    epsilon = 1e-6

    # 【核心：頻域標準化】
    # 將界於 [min_period, max_period] 的週期轉換為 [-1.0, 1.0] 的相對頻率
    # -1 代表最短週期 (高頻雜訊/震盪)；1 代表最長週期 (低頻信號/長趨勢)
    norm_period = (
        (dom_period - min_period) / (max_period - min_period + epsilon)
    ) * 2.0 - 1.0

    # ---------------------------------------------------------
    # (A) Position (相對頻率水位): 當前市場被哪種頻率主導
    # 語意補值: 0.0 (頻率處於搜索區間正中央)
    # ---------------------------------------------------------
    # Stable 版與 Sensitive 版先天已理論約束於 [-1.0, 1.0]
    feat_spectral_position_stable = (
        norm_period.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    feat_spectral_position_sensitive = (
        norm_period.fill_nan(0.0).fill_null(0.0).clip(-1.2, 1.2).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (週期循環乖離): 當前主導週期相對於歷史均線的偏離
    # 語意補值: 0.0 (市場週期頻率維持慣性不變)
    # ---------------------------------------------------------
    period_ema_macro = norm_period.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = norm_period - period_ema_macro

    # Stable 版：約束於 [-1.0, 1.0]
    feat_spectral_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-2.0, 2.0]，捕捉市場瞬間從高頻雜訊切換到長趨勢的大幅跨越
    feat_spectral_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (頻率切換加速度): 頻域變化的速度 (一階導數)
    # 語意補值: 0.0 (頻率轉換維持等速或定頻)
    # 降共線性處理: 減去短線 EMA 並自適應標準化
    # ---------------------------------------------------------
    ema_norm_period = norm_period.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (norm_period - ema_norm_period) / (ema_norm_period.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0]
    feat_spectral_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯變盤瞬間頻率結構的猛烈翻轉
    feat_spectral_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (頻譜混沌度): 頻率波動的歷史變異數
    # 語意補值: 0.0 (市場死心塌地維持單一週期，極度純粹)
    # ---------------------------------------------------------
    spectral_vol = norm_period.rolling_std(window_size=adapt_vol_p)
    log_spectral_vol = spectral_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_spectral_volatility_stable = (
        log_spectral_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留多種頻率打架產生的頻域極度不穩定狀態
    feat_spectral_volatility_sensitive = (
        log_spectral_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_spectral_position_stable": feat_spectral_position_stable,
        "feat_spectral_position_sensitive": feat_spectral_position_sensitive,
        "feat_spectral_bias_stable": feat_spectral_bias_stable,
        "feat_spectral_bias_sensitive": feat_spectral_bias_sensitive,
        "feat_spectral_momentum_stable": feat_spectral_momentum_stable,
        "feat_spectral_momentum_sensitive": feat_spectral_momentum_sensitive,
        "feat_spectral_volatility_stable": feat_spectral_volatility_stable,
        "feat_spectral_volatility_sensitive": feat_spectral_volatility_sensitive,
    }
