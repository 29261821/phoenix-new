# ==============================================================================
# § 指標 | 波動率偏斜 (Volatility Skew) v3.0
# 核心職責: 測量價格相對於其基於 ATR 的極值波動帶的位置，辨識極度超買賣。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| price         | H & G | str  | -        | 無 (必填)       | 價格來源 (如 'close') |
| vol_period    | H & G | int  | 10 ~ 21  | 無 (必填)       | 波動帶寬度 (ATR) 的計算週期 |
| window        | H & G | int  | 10 ~ 30  | 無 (必填)       | 尋找極值 (Highest/Lowest) 的窗口 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | window 參數值   | 用於 Bias (偏斜宏觀乖離) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (偏斜翻轉加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 14 ~ 34  | vol_period 參數 | 用於 Volatility (偏斜狀態混沌度) 的滾動標準差週期 |

【特徵工程說明】
- Skew 的原始輸出通常介於 0~1 之間。G 接口將其中心化並映射至 [-1.0, 1.0] 的對稱神經網路空間。
- 透過 adapt_micro_p 捕捉偏斜指標逼近 1.0 或 0.0 後的瞬間彎折，極速識別超買/超賣區的反轉。
"""
from typing import Dict

import polars as pl

from src.features.functions.atr import calculate as atr
from src.features.functions.highest import calculate as highest
from src.features.functions.lowest import calculate as lowest


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的 Skew 數值 (通常為 0 ~ 1.0 之間)。
    供傳統量化腳本作為過濾器 (如 Skew > 1 視為極端超買，價格刺穿波動帶) 調用。

    契約：
    - df 必須包含 params['price'], 'high', 'low', 'close' 指定的欄位。
    - params 必須包含 'price', 'vol_period', 'window' 鍵。
    """
    price_col, vol_period, window = (
        params["price"],
        params["vol_period"],
        params["window"],
    )
    price = pl.col(price_col)
    epsilon = 1e-9

    vol_series = atr(df=df, period=vol_period)

    highest_price = highest(series=price, period=window)
    lowest_price = lowest(series=price, period=window)

    upper_band = highest_price - vol_series
    lower_band = lowest_price + vol_series

    skew = (price - lower_band) / (upper_band - lower_band + epsilon)

    return {"type": "scalar", "values": {"Skew": skew}}


def adapt_Volatility_Skew(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將 [0, 1] 的偏斜指標轉換為 [-1, 1] 的無量綱時空特徵。
    正交分解為：偏斜絕對水位 (Position)、偏斜宏觀乖離 (Bias)、偏斜翻轉加速度 (Momentum)、狀態混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    skew = h_output["values"]["Skew"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["window"])
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", params["vol_period"])

    epsilon = 1e-6

    # 【核心：中心化與縮放】
    # 將 0~1.0 映射為 -1.0 ~ 1.0 的對稱空間，0 代表多空平衡
    centered_skew = (skew - 0.5) * 2.0

    # ---------------------------------------------------------
    # (A) Position (偏斜絕對水位): 價格在極值波動帶的相對位置
    # 語意補值: 0.0 (代表位於波動帶正中央)
    # ---------------------------------------------------------
    # Stable 版：嚴格約束於 [-1.0, 1.0]
    feat_skew_position_stable = (
        centered_skew.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：略微放寬至 [-1.5, 1.5]，包容價格嚴重刺穿波動極限帶的黑天鵝行情
    feat_skew_position_sensitive = (
        centered_skew.fill_nan(0.0).fill_null(0.0).clip(-1.5, 1.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (偏斜宏觀乖離): 偏斜狀態相對於其長線政權的背離
    # 語意補值: 0.0 (當前偏斜符合近期宏觀慣性)
    # ---------------------------------------------------------
    skew_ema_macro = centered_skew.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = centered_skew - skew_ema_macro

    # Stable 版：約束於 [-1.0, 1.0]
    feat_skew_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-2.0, 2.0]
    feat_skew_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (偏斜翻轉加速度): 核心 Skew 的變化速度 (一階導數)
    # 語意補值: 0.0 (偏斜變化維持等速)
    # 降共線性處理: 減去短線 EMA 並自適應標準化，提早捕捉價格碰觸極值波動帶後的反向彎折
    # ---------------------------------------------------------
    ema_centered_skew = centered_skew.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (centered_skew - ema_centered_skew) / (ema_centered_skew.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0]
    feat_skew_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極度凸顯 V 轉或 A 轉瞬間強大的反轉加速度
    feat_skew_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (偏斜狀態混沌度): Skew 信號的歷史變異數
    # 語意補值: 0.0 (偏斜位置維持死水，或保持極其穩定的推進)
    # 防禦處理: 強制套用 log1p 平滑
    # ---------------------------------------------------------
    skew_vol = centered_skew.rolling_std(window_size=adapt_vol_p)
    log_skew_vol = skew_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_skew_volatility_stable = (
        log_skew_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留價格在波動帶上下兩端頻繁亂竄的高熵狀態
    feat_skew_volatility_sensitive = (
        log_skew_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_skew_position_stable": feat_skew_position_stable,
        "feat_skew_position_sensitive": feat_skew_position_sensitive,
        "feat_skew_bias_stable": feat_skew_bias_stable,
        "feat_skew_bias_sensitive": feat_skew_bias_sensitive,
        "feat_skew_momentum_stable": feat_skew_momentum_stable,
        "feat_skew_momentum_sensitive": feat_skew_momentum_sensitive,
        "feat_skew_volatility_stable": feat_skew_volatility_stable,
        "feat_skew_volatility_sensitive": feat_skew_volatility_sensitive,
    }
