# ==============================================================================
# § 指標 | 公允價值缺口 (Fair Value Gap)
# 核心職責: 識別由於價格快速移動而導致的市場失衡與流動性真空區域 (FVG Zone)。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| adapt_macro_p | G 專用| int  | 13 ~ 55  | 21              | 用於 Position (缺口政權) 的 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 13   | 5               | 用於 Momentum (逃逸動能) 計算時的 EMA 平滑週期 |

【特徵工程說明】
- FVG 本身無基礎參數，其觸發完全由連續 3 根 K 棒的微觀結構決定。
- 原始 FVG 是極度稀疏的缺口價格區間，G 接口透過 Forward Fill 將其轉為連續空間特徵。
- 透過 adapt_macro_p 決定模型觀察多空失衡政權 (Market Regime) 的歷史記憶長度。
- 透過 adapt_micro_p 決定模型對「觸碰缺口後瞬間拒絕 (Rejection)」動能的敏感度。
"""
from typing import Dict

import polars as pl

from src.features.functions.shift import calculate as prev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留精確的布林觸發信號與絕對價格的缺口區間 (Top, Bottom)。
    確保依賴 SMC (聰明錢概念) 理論的傳統量化策略，能準確將 FVG 區間
    作為掛單的支撐/壓力區域使用。

    契約：
    - df 必須包含 'high', 'low' 欄位。
    - params: 此指標無輸入參數 (特徵工程參數由 adapt 層接收)。
    """
    h, l = pl.col("high"), pl.col("low")
    prev2_h = prev(series=h, period=2)
    prev2_l = prev(series=l, period=2)

    is_bull = l > prev2_h
    is_bear = h < prev2_l

    top = pl.when(is_bull).then(l).when(is_bear).then(prev2_l).otherwise(None)
    bottom = pl.when(is_bull).then(prev2_h).when(is_bear).then(h).otherwise(None)

    return {
        "type": "zone",
        "values": {
            "isBullish": is_bull,
            "isBearish": is_bear,
            "Top": top,
            "Bottom": bottom,
        },
    }


def adapt_FVG(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將極度稀疏且局部的缺口區域，轉換為 DL/ML 可學習的連續時空特徵。
    正交分解為：缺口政權狀態 (Position)、缺口失衡度 (Volatility)、回補距離 (Bias)、逃逸動能 (Momentum)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉逃逸缺口) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，政權衰減與動能平滑週期全面可由 YAML 配置。
    """
    is_bull = h_output["values"]["isBullish"]
    is_bear = h_output["values"]["isBearish"]
    top_raw = h_output["values"]["Top"]
    bottom_raw = h_output["values"]["Bottom"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", 21)
    adapt_micro_p = params.get("adapt_micro_p", 5)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # 將離散信號轉為脈衝 (-1.0, 0.0, 1.0)
    impulse = pl.when(is_bull).then(1.0).when(is_bear).then(-1.0).otherwise(0.0)

    # 空間延續處理：記憶最後一次出現的缺口區間
    # 初始無缺口時使用當前 close 填補，避免 Null 污染整個資料列
    top = top_raw.forward_fill().fill_null(close)
    bottom = bottom_raw.forward_fill().fill_null(close)
    midpoint = (top + bottom) / 2.0

    # ---------------------------------------------------------
    # (A) Position (缺口政權狀態): 宏觀的市場失衡方向與頻率
    # 語意補值: 0.0 (近期多空失衡抵銷，或無缺口)
    # ---------------------------------------------------------
    regime = impulse.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：約束於 [-0.5, 0.5]，過濾極端連續跳空，穩定 DL 權重
    feat_fvg_position_stable = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]，捕捉史詩級的單邊失衡連續跳空
    feat_fvg_position_sensitive = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Volatility (缺口失衡度 / 頻寬): 最後一次缺口的寬度佔價格的百分比
    # 語意補值: 0.0 (無缺口)
    # 防禦處理: 強制套用 log1p 平滑極端跳空時的變異數爆炸
    # ---------------------------------------------------------
    width = (top - bottom) / (midpoint + epsilon)
    log_width = width.log1p()

    # Stable 版：約束於 [0.0, 0.05] (最多關注 5% 寬度的常規失衡區間)
    feat_fvg_volatility_stable = (
        log_width.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.05).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.2] (捕捉高波動資產高達 20% 的流動性黑洞)
    feat_fvg_volatility_sensitive = (
        log_width.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.2).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Bias (缺口中樞乖離 / 回補距離): 當前價格距離最後一次缺口中樞的偏離
    # 語意補值: 0.0 (價格完美回補並停留在缺口中樞，或無缺口)
    # ---------------------------------------------------------
    bias = (close / (midpoint + epsilon)) - 1.0

    # Stable 版：約束於 [-0.1, 0.1]，專注於價格在缺口附近 10% 的微觀互動與回補測試
    feat_fvg_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.1, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.5, 0.5]，允許樹模型捕捉發生「突破性逃逸缺口」後的不回頭行情
    feat_fvg_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Momentum (缺口逃逸/回歸動能): 回補距離 (Bias) 的變化速度
    # 語意補值: 0.0 (價格在缺口內或外平穩遊走，無加速遠離或回歸)
    # 降共線性處理: 減去自身的 EMA 並標準化，凸顯價格「觸碰缺口後瞬間被拒絕」的彈射動能
    # ---------------------------------------------------------
    ema_bias = bias.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (bias - ema_bias) / (ema_bias.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_fvg_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，捕捉碰觸 FVG 後劇烈反抽的極端加速度
    feat_fvg_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    return {
        "feat_fvg_position_stable": feat_fvg_position_stable,
        "feat_fvg_position_sensitive": feat_fvg_position_sensitive,
        "feat_fvg_volatility_stable": feat_fvg_volatility_stable,
        "feat_fvg_volatility_sensitive": feat_fvg_volatility_sensitive,
        "feat_fvg_bias_stable": feat_fvg_bias_stable,
        "feat_fvg_bias_sensitive": feat_fvg_bias_sensitive,
        "feat_fvg_momentum_stable": feat_fvg_momentum_stable,
        "feat_fvg_momentum_sensitive": feat_fvg_momentum_sensitive,
    }
