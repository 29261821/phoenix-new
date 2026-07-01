# 檔案: src/features/indicators/Hurst.py
# 版本: v4.0 (原生表達式版 + 頂規特徵工程升級)
# ==============================================================================
# § 指標 | 赫斯特指數引擎 (Hurst Engine)
# 核心職責: 計算時間序列的長期記憶性，判斷市場處於趨勢延續、隨機漫步或均值回歸政權。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| input_feature | H & G | str  | -        | 無 (必填)       | 赫斯特指數分析的目標欄位 (如 'close') |
| window_size   | H & G | int  | 100 ~ 500| 無 (必填)       | 滾動赫斯特指數的總觀察視窗大小 |
| min_sub_period| H & G | int  | 10 ~ 30  | 無 (必填)       | 最小的子區間長度 (用於 R/S 分析) |
| max_sub_period| H & G | int  | 50 ~ 200 | 無 (必填)       | 最大的子區間長度 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | window_size 參數| 用於 Bias (政權乖離) 計算的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 5 ~ 14   | 5               | 用於 Momentum (切換加速度) 計算的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | 34              | 用於 Volatility (政權混沌度) 的滾動標準差週期 |

【特徵工程說明】
- 原始 Hurst 指數範圍為 [0, 1]，0.5 為隨機漫步。
- G 接口將其中心化並縮放至 [-1, 1]，0.0 代表隨機漫步，正值代表趨勢，負值代表均值回歸。
- 透過 adapt_macro_p 觀察市場「趨勢性」相對於長期歷史的乖離狀態。
- 透過 adapt_vol_p 衡量政權是否在趨勢與震盪間頻繁且混亂地切換。
"""
from typing import Any, Dict

import polars as pl

# --- [v4.0 核心升級] ---
# 導入新一代的、基於表達式的原生核心接口
from src.features.rust_functions import rolling_hurst_expr


def calculate(df: pl.DataFrame, params: Dict[str, Any], **kwargs) -> Dict[str, pl.Expr]:
    """
    v4.0 (原生表達式版):
    - [根本性架構回歸] 為遵從 Polars 契約，本指標的返回類型從 `pl.Series`
      回歸到 `pl.Expr`。
    - [性能與契約統一] 將核心計算任務，委託給 `rolling_hurst_expr`。
      此函數返回一個完整的、可被 Polars 優化的計算圖 (`pl.Expr`)，
      從而與【惰性閃擊軍團】的所有成員在架構上達成完全統一。

    契約：
    - 返回標準的 H 接口字典 (含 'type' 與 'values')，交由 executor 的標準惰性模式處理。
    """
    # --- 1. 契約驗證與參數提取 ---
    feature_col: str = params.get("input_feature")
    window_size: int = params.get("window_size")
    min_sub_period: int = params.get("min_sub_period")
    max_sub_period: int = params.get("max_sub_period")

    if not all([feature_col, window_size, min_sub_period, max_sub_period]):
        raise ValueError(
            "Hurst_Engine 的參數 'input_feature', 'window_size', 'min_sub_period', 'max_sub_period' 必須被提供。"
        )
    if feature_col not in df.columns:
        raise ValueError(f"DataFrame 中缺少赫斯特指數分析所需的欄位: {feature_col}")

    # --- 2. 數據準備 (表達式) ---
    # 創建一個代表輸入數據列的表達式
    input_expr = pl.col(feature_col).forward_fill()

    # --- 3. 授權原生核心構建計算圖 ---
    hurst_expr = rolling_hurst_expr(
        expr=input_expr,
        window_size=window_size,
        min_sub_period=min_sub_period,
        max_sub_period=max_sub_period,
    )

    # --- 4. 格式化輸出 ---
    return {"type": "scalar", "values": {"Hurst": hurst_expr}}


def adapt_Hurst(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將絕對數值 [0, 1] 的赫斯特指數轉換為以 0.0 (隨機漫步) 為中心的對稱連續特徵。
    正交分解為：市場政權水位 (Position)、政權宏觀乖離 (Bias)、切換加速度 (Momentum)、政權混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，所有動能與變異數週期全面可由 YAML 配置。
    """
    hurst = h_output["values"]["Hurst"].cast(pl.Float64)

    # 1. 提取基礎參數
    window_size = params["window_size"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", window_size)
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (市場政權水位): 將 Hurst 理論值 [0, 1] 轉換為 [-1, 1] 的對稱空間
    # 語意補值: 0.0 (代表 Hurst = 0.5，處於純粹的隨機漫步)
    # > 0 代表趨勢延續性強，< 0 代表均值回歸性強
    # ---------------------------------------------------------
    centered_hurst = (hurst - 0.5) * 2.0

    # Stable & Sensitive 版：先天理論邊界即為 [-1.0, 1.0]，無需過度放寬
    feat_hurst_position_stable = (
        centered_hurst.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    feat_hurst_position_sensitive = (
        centered_hurst.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (政權宏觀乖離): 市場政權相對於其長線均線的偏離
    # 語意補值: 0.0 (當前市場特性與長期歷史屬性一致)
    # ---------------------------------------------------------
    hurst_ema_macro = centered_hurst.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = centered_hurst - hurst_ema_macro

    # Stable 版：約束於 [-0.5, 0.5]，過濾微小波動
    feat_hurst_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]，捕捉政權發生歷史級別反轉的乖離
    feat_hurst_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (切換加速度): 政權水位 (Position) 的變化速度
    # 語意補值: 0.0 (政權特性維持等速發展)
    # 降共線性處理: 減去自身的短線 EMA 並自適應標準化，凸顯瞬間從震盪轉向趨勢的爆發力
    # ---------------------------------------------------------
    ema_centered_hurst = centered_hurst.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (centered_hurst - ema_centered_hurst) / (
        ema_centered_hurst.abs() + epsilon
    )

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_hurst_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極致捕捉市場結構瞬間巨變的動能峰值
    feat_hurst_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (政權混沌度): Hurst 指數的歷史變異數
    # 語意補值: 0.0 (市場政權極度穩定，死心塌地的維持趨勢或震盪)
    # 防禦處理: 強制套用 log1p 平滑
    # 若數值飆高，代表市場在「趨勢」與「均值回歸」之間頻繁切換，處於極難交易的混沌期
    # ---------------------------------------------------------
    hurst_volatility = centered_hurst.rolling_std(window_size=adapt_vol_p)
    log_hurst_vol = hurst_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_hurst_volatility_stable = (
        log_hurst_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留極端混沌政權特徵
    feat_hurst_volatility_sensitive = (
        log_hurst_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_hurst_position_stable": feat_hurst_position_stable,
        "feat_hurst_position_sensitive": feat_hurst_position_sensitive,
        "feat_hurst_bias_stable": feat_hurst_bias_stable,
        "feat_hurst_bias_sensitive": feat_hurst_bias_sensitive,
        "feat_hurst_momentum_stable": feat_hurst_momentum_stable,
        "feat_hurst_momentum_sensitive": feat_hurst_momentum_sensitive,
        "feat_hurst_volatility_stable": feat_hurst_volatility_stable,
        "feat_hurst_volatility_sensitive": feat_hurst_volatility_sensitive,
    }
