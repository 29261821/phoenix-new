# ==============================================================================
# § 指標 | 交易者動態指數 (Traders Dynamic Function Index, TDFI)
# 核心職責: 一種經過雙重平滑和歸一化的高階動能指標，衡量價格趨勢方向與強度。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型  | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|-------|----------|-----------------|------|
| price_src     | H & G | str   | -        | 無 (必填)       | 價格來源 (如 'close') |
| mma_length    | H & G | int   | 3 ~ 10   | 無 (必填)       | MMA (單次平滑均線) 的週期 |
| smma_length   | H & G | int   | 10 ~ 30  | 無 (必填)       | SMMA (雙重平滑均線) 的週期 |
| std_length    | H & G | int   | 10 ~ 30  | 無 (必填)       | 計算標準差通道的週期 |
| multiplier    | H     | float | 1.0 ~ 3.0| 無 (必填)       | 決定波動帶寬度的標準差乘數 |
| adapt_macro_p | G 專用| int   | 21 ~ 55  | smma_length 參數| 用於 Bias (動能宏觀乖離) 計算的長線 EMA 週期 |
| adapt_micro_p | G 專用| int   | 3 ~ 10   | 5               | 用於 Momentum (翻轉加速度) 計算的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int   | 13 ~ 34  | std_length 參數 | 用於 Volatility (動能混沌度) 的滾動標準差週期 |

【特徵工程說明】
- TDFI 原始輸出通常介於 -50 到 50 之間。G 接口將其除以 50，完美縮放至 [-1.0, 1.0] 的對稱空間。
- 透過 adapt_macro_p 觀察 TDFI 偏離其近期歷史中樞的程度，捕捉動能的頂底背離。
- 透過 adapt_micro_p 的加速度特徵，提早識別 TDFI 撞擊上下限時的彎折與衰竭。
"""
from typing import Dict

import polars as pl

from src.features.functions.ema import calculate as ema
from src.features.functions.stddev import calculate as stddev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的 TDFI 數值 (通常為 -50 ~ 50 之間，但極端時可能溢出)。
    供傳統量化腳本作為過濾器 (如 TDFI > 0 視為作多信號) 調用。

    契約：
    - df 必須包含 params['price_src'] 指定的欄位。
    - params 必須包含 'price_src', 'mma_length', 'smma_length',
      'std_length', 'multiplier' 鍵。
    """
    price_src_col, mma_len, smma_len, std_len, mult = (
        params["price_src"],
        params["mma_length"],
        params["smma_length"],
        params["std_length"],
        params["multiplier"],
    )
    price_src = pl.col(price_src_col)
    epsilon = 1e-9

    # --- 1. 計算均線 ---
    # mma 是指標的核心，smma 僅用於計算標準差，定義波動帶
    mma = ema(series=price_src, length=mma_len)
    smma = ema(series=mma, length=smma_len)  # 雙重平滑均線

    # --- 2. 計算波動帶 ---
    # 標準差和波動帶應圍繞 `smma` 計算
    std_dev = stddev(series=smma, period=std_len)
    upper_band = smma + mult * std_dev
    lower_band = smma - mult * std_dev

    # --- 3. 計算 TDFI 值 ---
    # TDFI 的值是 `mma` 在 `upper_band` 和 `lower_band` 之間的位置
    tdfi_val = (
        pl.when(upper_band == lower_band)
        .then(0.5)  # 如果通道寬度為0，返回中間值 0.5 (對應 -0.5*100 -> 0)
        .otherwise((mma - lower_band) / (upper_band - lower_band + epsilon))
    )

    # 將結果從 0-1 區間轉換到 -50 到 50 區間
    final_tdfi = (tdfi_val - 0.5) * 100

    return {"type": "scalar", "values": {"TDFI": final_tdfi}}


def adapt_TDFI(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將絕對動能 TDFI 轉換為 [-1, 1] 的無量綱時空特徵。
    正交分解為：動能絕對水位 (Position)、動能宏觀乖離 (Bias)、翻轉加速度 (Momentum)、動能混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    tdfi_val = h_output["values"]["TDFI"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["smma_length"])
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", params["std_length"])

    epsilon = 1e-6

    # 【核心：中心化與縮放】
    # 將 -50~50 映射為 -1.0 ~ 1.0 的對稱空間，0 代表多空平衡
    norm_tdfi = tdfi_val / 50.0

    # ---------------------------------------------------------
    # (A) Position (動能絕對水位): 核心 TDFI 線的相對位置
    # 語意補值: 0.0 (代表動能處於多空平衡的中立區)
    # ---------------------------------------------------------
    # Stable 版：嚴格約束於 [-1.0, 1.0]
    feat_tdfi_position_stable = (
        norm_tdfi.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：略微放寬至 [-2.0, 2.0]，包容極端暴拉/暴跌時穿透標準差通道的溢出
    feat_tdfi_position_sensitive = (
        norm_tdfi.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (動能宏觀乖離): 動能相對於其長線政權的背離
    # 語意補值: 0.0 (當前動能與近期宏觀動能一致)
    # ---------------------------------------------------------
    tdfi_ema_macro = norm_tdfi.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = norm_tdfi - tdfi_ema_macro

    # Stable 版：約束於 [-1.0, 1.0]
    feat_tdfi_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-2.0, 2.0]
    feat_tdfi_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (動能翻轉加速度): 核心 TDFI 的變化速度 (一階導數)
    # 語意補值: 0.0 (動能推進維持等速)
    # 降共線性處理: 減去短線 EMA 並自適應標準化，提早捕捉 TDFI 的彎折
    # ---------------------------------------------------------
    ema_norm_tdfi = norm_tdfi.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (norm_tdfi - ema_norm_tdfi) / (ema_norm_tdfi.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0]
    feat_tdfi_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極度凸顯 V 轉或 A 轉瞬間強大的反轉加速度
    feat_tdfi_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (動能混沌度): TDFI 信號的歷史變異數
    # 語意補值: 0.0 (動能維持單向推進，極度順暢)
    # 防禦處理: 強制套用 log1p 平滑
    # ---------------------------------------------------------
    tdfi_vol = norm_tdfi.rolling_std(window_size=adapt_vol_p)
    log_tdfi_vol = tdfi_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_tdfi_volatility_stable = (
        log_tdfi_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留多空拉鋸導致動能反覆震盪的混沌狀態
    feat_tdfi_volatility_sensitive = (
        log_tdfi_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_tdfi_position_stable": feat_tdfi_position_stable,
        "feat_tdfi_position_sensitive": feat_tdfi_position_sensitive,
        "feat_tdfi_bias_stable": feat_tdfi_bias_stable,
        "feat_tdfi_bias_sensitive": feat_tdfi_bias_sensitive,
        "feat_tdfi_momentum_stable": feat_tdfi_momentum_stable,
        "feat_tdfi_momentum_sensitive": feat_tdfi_momentum_sensitive,
        "feat_tdfi_volatility_stable": feat_tdfi_volatility_stable,
        "feat_tdfi_volatility_sensitive": feat_tdfi_volatility_sensitive,
    }
