# ==============================================================================
# § 指標 | 赫爾移動平均線 (Hull Moving Average)
# 核心職責: 透過加權移動平均的差值計算，提供一種極其平滑且反應靈敏的移動平均線。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| source        | H & G | str  | -        | 無 (必填)       | 價格來源 (如 'close') |
| length        | H & G | int  | 10 ~ 100 | 無 (必填)       | 赫爾均線的基礎計算週期 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | 55              | 用於 Position (均線歷史水位) 的 Z-Score 觀察週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (乖離加速度) 的 EMA 平滑週期 |
| adapt_vol_p   | G 專用| int  | 13 ~ 55  | length 參數的值 | 用於 Volatility (均線纏繞混沌度) 的滾動標準差週期 |

【特徵工程說明】
- 原始 HMA 是帶有絕對價格尺度的趨勢線，G 接口將其轉換為無量綱連續特徵。
- 透過 adapt_macro_p 計算 HMA 的滾動 Z-Score，衡量趨勢的絕對高低水位。
- 透過 adapt_vol_p 衡量價格圍繞均線波動的劇烈程度，識別頻繁穿越的洗盤或單邊行情。
"""
from typing import Dict

import polars as pl

from src.features.functions.floor import calculate as floor
from src.features.functions.sqrt import calculate as sqrt
from src.features.functions.wma import calculate as wma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算赫爾移動平均線 (HMA)。
    保留絕對的價格位準，供傳統量化策略 (如價格穿越 HMA 視為趨勢反轉) 直接調用。

    契約：
    - df 必須包含 params['source'] 指定的欄位。
    - params 必須包含 'source', 'length' 鍵。
    """
    source_col = params["source"]
    length = params["length"]
    source = pl.col(source_col)

    # 核心邏輯需要純量計算，Polars 表達式無法直接處理，故在 Python 層完成
    half_len = int(length / 2)
    sqrt_len = int(length**0.5)

    wma_h = wma(series=source, length=half_len)
    wma_f = wma(series=source, length=length)
    hma_raw = 2 * wma_h - wma_f
    hma_val = wma(series=hma_raw, length=sqrt_len)

    return {"type": "vector", "values": {"HMA": hma_val}}


def adapt_Hull_MA(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將絕對價格尺度的 HMA 轉換為 DL/ML 可學習的無量綱時空特徵。
    正交分解為：均線歷史水位 (Position)、價格乖離 (Bias)、乖離加速度 (Momentum)、纏繞混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端趨勢) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，動能與變異數週期全面可由 YAML 配置。
    """
    hma = h_output["values"]["HMA"]

    # 1. 提取基礎參數
    length = params["length"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", 55)
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", length)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (均線歷史水位): HMA 的滾動 Z-Score
    # 語意補值: 0.0 (HMA 處於歷史常態均值)
    # 衡量當前的平滑趨勢線在宏觀歷史中的相對高低位置
    # ---------------------------------------------------------
    hma_mean = hma.rolling_mean(window_size=adapt_macro_p)
    hma_std = hma.rolling_std(window_size=adapt_macro_p)
    z_score = (hma - hma_mean) / (hma_std + epsilon)

    # Stable 版：約束於 [-3.0, 3.0]
    feat_hma_position_stable = (
        z_score.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-5.0, 5.0]，捕捉史詩級單邊趨勢的極端水位
    feat_hma_position_sensitive = (
        z_score.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (價格乖離): 實際收盤價相對於 HMA 的偏離程度
    # 語意補值: 0.0 (價格完美貼合 HMA 趨勢線)
    # ---------------------------------------------------------
    bias = (close / (hma + epsilon)) - 1.0

    # Stable 版：約束於 [-0.1, 0.1]，專注於常規的均值回歸與微小突破
    feat_hma_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.1, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.3, 0.3]，保留暴跌暴漲時價格甩開均線的極端乖離
    feat_hma_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.3, 0.3).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (乖離加速度): 價格乖離 (Bias) 的變化速度
    # 語意補值: 0.0 (價格與 HMA 的距離維持等速發展)
    # 降共線性處理: 減去自身的短線 EMA 並自適應標準化，凸顯瞬間突破或猛烈均值回歸的動能
    # ---------------------------------------------------------
    ema_bias = bias.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (bias - ema_bias) / (ema_bias.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_hma_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極致捕捉變盤瞬間暴力刺穿 HMA 的動能峰值
    feat_hma_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (均線纏繞混沌度): 價格乖離的歷史變異數
    # 語意補值: 0.0 (價格平穩地在均線一側發展，無劇烈震盪)
    # 防禦處理: 強制套用 log1p 平滑
    # 衡量價格是在穩定的單邊趨勢中，還是處於頻繁上穿下穿均線的「絞肉機」洗盤狀態
    # ---------------------------------------------------------
    bias_volatility = bias.rolling_std(window_size=adapt_vol_p)
    log_volatility = bias_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.1]
    feat_hma_volatility_stable = (
        log_volatility.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.3]，保留極端雙巴洗盤時的混沌特徵
    feat_hma_volatility_sensitive = (
        log_volatility.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.3).cast(pl.Float64)
    )

    return {
        "feat_hma_position_stable": feat_hma_position_stable,
        "feat_hma_position_sensitive": feat_hma_position_sensitive,
        "feat_hma_bias_stable": feat_hma_bias_stable,
        "feat_hma_bias_sensitive": feat_hma_bias_sensitive,
        "feat_hma_momentum_stable": feat_hma_momentum_stable,
        "feat_hma_momentum_sensitive": feat_hma_momentum_sensitive,
        "feat_hma_volatility_stable": feat_hma_volatility_stable,
        "feat_hma_volatility_sensitive": feat_hma_volatility_sensitive,
    }
