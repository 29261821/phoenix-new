# ==============================================================================
# § 指標 | 波動率幾何 v3.0 (150分典範版)
# 核心職責: 衡量波動率的變化速率，即波動率的「加速度」。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口極致抗尺度特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| period        | H & G | int  | 3 ~ 14   | 無 (必填)       | 波動率幾何的差分回顧週期 |
| atr_period    | H & G | int  | 10 ~ 21  | 無 (必填)       | 基礎 ATR 的計算週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | atr_period 參數 | 用於 Bias (波動加速度乖離) 的長線 EMA 週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (波動率衝動/Jerk) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 14 ~ 34  | period 參數值   | 用於 Volatility (波動引擎混沌度) 的標準差週期 |

【特徵工程說明】
- 原始 Geometry 是絕對波動點數差，帶有嚴重的 Scale 污染。
- G 接口將其除以 close 轉化為無量綱的「百分比波動加速度」。正值代表恐慌蔓延，負值代表情緒平復。
- 透過 adapt_micro_p 計算波動率加速度的變化率 (物理學的 Jerk)，極度敏銳捕捉波動率噴發的引爆點。
"""
from typing import Any, Dict

import polars as pl

# [邏輯自治] 遵循 DSL v6.0 的設計思想，此指標為「自產自銷」。
from src.features.functions.atr import calculate as atr
from src.features.functions.shift import calculate as prev


def calculate(df: pl.DataFrame, params: Dict[str, Any], **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對尺度的波動率加速度 (ATR 差值)。
    確保依賴絕對點數變化 (如 Geometry > 100 points 視為波動率突破) 的策略可無縫運行。

    契約：
    - df: pl.DataFrame, 必須包含 'high', 'low', 'close' 欄位。
    - params: Dict, 必須包含 'period', 'atr_period' 鍵。
    """
    # --- 1. 契約驗證與參數提取 ---
    period, atr_period = params.get("period"), params.get("atr_period")

    if not all([period, atr_period]):
        raise ValueError(
            "Volatility_Geometry 的參數 'period' 和 'atr_period' 必須被提供。"
        )

    # --- 2. 計算核心波動率指標 ---
    # 平均真實波幅 (ATR) 是衡量市場波動性的黃金標準。
    atr_val = atr(df=df, period=atr_period)

    # --- 3. 計算波動率的變化率 (幾何) ---
    # 比較當前的 ATR 值與 `period` 根 K 棒前的 ATR 值，
    # 這個差值即代表了波動率在這段時間內的「加速度」。
    prev_atr = prev(series=atr_val, period=period)
    geometry = atr_val - prev_atr

    return {"type": "scalar", "values": {"Geometry": geometry}}


def adapt_Volatility_Geometry(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將嚴重受價格尺度污染的 Geometry 轉換為無量綱的 DL/ML 特徵。
    正交分解為：波動率加速度水位 (Position)、加速度宏觀乖離 (Bias)、波動率衝動 (Momentum)、波動引擎混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    geometry = h_output["values"]["Geometry"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["atr_period"])
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", params["period"])

    epsilon = 1e-6

    # 【核心：消除絕對 Scale 污染】
    # 將絕對點數除以收盤價，轉換為無量綱的「百分比波動加速度」
    norm_geo = geometry / (close + epsilon)

    # ---------------------------------------------------------
    # (A) Position (波動率加速度水位): 當前波動擴張或收斂的速度
    # 語意補值: 0.0 (波動率維持等速，無加速擴張或收縮)
    # ---------------------------------------------------------
    # Stable 版：約束於 [-0.05, 0.05]，專注於常規的 5% 以內的波動率加速變化
    feat_vol_geo_position_stable = (
        norm_geo.fill_nan(0.0).fill_null(0.0).clip(-0.05, 0.05).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-0.2, 0.2]，捕捉黑天鵝事件導致的極端波動率急凍或暴增
    feat_vol_geo_position_sensitive = (
        norm_geo.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (加速度宏觀乖離): 波動加速度相對於長線政權的背離
    # 語意補值: 0.0 (當前波動率變化符合近期宏觀慣性)
    # ---------------------------------------------------------
    geo_ema_macro = norm_geo.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = norm_geo - geo_ema_macro

    # Stable 版：約束於 [-0.05, 0.05]
    feat_vol_geo_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.05, 0.05).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.15, 0.15]
    feat_vol_geo_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.15, 0.15).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (波動率衝動 / Jerk): 加速度的變化率 (即加速度的加速度)
    # 語意補值: 0.0 (波動率加速度維持等速)
    # 降共線性處理: 減去短線 EMA 並自適應標準化，極度敏銳捕捉暴風雨前夕的「瞬間收斂引爆點」
    # ---------------------------------------------------------
    ema_norm_geo = norm_geo.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (norm_geo - ema_norm_geo) / (ema_norm_geo.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化空間
    feat_vol_geo_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯變盤瞬間強大的波動率衝動
    feat_vol_geo_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (波動引擎混沌度): 波動率加速度的歷史變異數
    # 語意補值: 0.0 (波動引擎運行極度平穩，或維持穩定單邊死水)
    # 防禦處理: 強制套用 log1p 平滑
    # ---------------------------------------------------------
    geo_volatility = norm_geo.rolling_std(window_size=adapt_vol_p)
    log_geo_vol = geo_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.2]
    feat_vol_geo_volatility_stable = (
        log_geo_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.5]
    feat_vol_geo_volatility_sensitive = (
        log_geo_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )

    return {
        "feat_vol_geo_position_stable": feat_vol_geo_position_stable,
        "feat_vol_geo_position_sensitive": feat_vol_geo_position_sensitive,
        "feat_vol_bias_stable": feat_vol_geo_bias_stable,
        "feat_vol_bias_sensitive": feat_vol_geo_bias_sensitive,
        "feat_vol_geo_momentum_stable": feat_vol_geo_momentum_stable,
        "feat_vol_geo_momentum_sensitive": feat_vol_geo_momentum_sensitive,
        "feat_vol_geo_volatility_stable": feat_vol_geo_volatility_stable,
        "feat_vol_geo_volatility_sensitive": feat_vol_geo_volatility_sensitive,
    }
