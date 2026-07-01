# ==============================================================================
# § 指標 | 超級趨勢 (SuperTrend)
# 核心職責: 結合 ATR 與中樞價格的經典趨勢追蹤指標，識別市場政權與動態支撐壓力。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型  | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|-------|----------|-----------------|------|
| atr_len       | H & G | int   | 10 ~ 21  | 無 (必填)       | ATR 波幅的計算週期 |
| multiplier    | H     | float | 1.5 ~ 3.0| 無 (必填)       | 決定距離均線寬度的 ATR 乘數 |
| adapt_macro_p | G 專用| int   | 34 ~ 89  | 55              | 用於 Position (政權水位) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int   | 5 ~ 14   | atr_len 參數值  | 用於 Momentum (逼近/遠離加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int   | 21 ~ 55  | 34              | 用於 Volatility (政權混沌度) 的滾動標準差週期 |

【特徵工程說明】
- 原始 SuperTrend 軌道帶有絕對價格尺度，且方向訊號為離散的 +1/-1。
- G 接口將絕對價格轉化為無量綱的 `Bias (軌道乖離)`，衡量價格偏離支撐壓力的空間。
- 透過 `adapt_vol_p` 對 +1/-1 的切換狀態計算變異數，極端敏銳地識別「假突破雙巴洗盤」的絞肉機行情。
"""
from typing import Dict

import numpy as np
import polars as pl

from src.features.functions.atr import calculate as atr


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算超級趨勢軌道與當前多空方向。
    本指標屬於「 eager a.k.a. non-expression-based 」類型，因其複雜的遞迴狀態。
    [契約修復]: 將 Eager 產出的 Series 透過 pl.lit() 封裝為 pl.Expr，完美對接惰性計算圖。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含 'atr_len', 'multiplier' 鍵。
    - [健壯性] 採用迭代計算，完美復刻 DSL 的 prev(state, 1).fill_null() 契約。
    """
    atr_len = params["atr_len"]
    multiplier = params["multiplier"]

    atr_val_expr = atr(df=df, period=atr_len)
    hl2 = (pl.col("high") + pl.col("low")) / 2
    up_band_expr = hl2 - (multiplier * atr_val_expr)
    dn_band_expr = hl2 + (multiplier * atr_val_expr)

    df_with_bands = df.with_columns(
        up_band=up_band_expr,
        dn_band=dn_band_expr,
        close=pl.col("close"),
    )

    close_np = df_with_bands["close"].to_numpy()
    up_band_np = df_with_bands["up_band"].to_numpy()
    dn_band_np = df_with_bands["dn_band"].to_numpy()

    n = len(df)
    direction_np = np.ones(n, dtype=np.int8)
    trend_np = np.zeros(n, dtype=np.float64)

    prev_direction = 1
    prev_trend = 0.0

    for i in range(1, n):
        if close_np[i] > prev_trend:
            direction_np[i] = 1
        else:
            direction_np[i] = -1

        if direction_np[i] == 1 and prev_direction == -1:
            trend_np[i] = up_band_np[i]
        elif direction_np[i] == -1 and prev_direction == 1:
            trend_np[i] = dn_band_np[i]
        elif direction_np[i] == 1:
            trend_np[i] = max(up_band_np[i], prev_trend)
        else:
            trend_np[i] = min(dn_band_np[i], prev_trend)

        prev_direction = direction_np[i]
        prev_trend = trend_np[i]

    return {
        "type": "vector",
        "values": {
            "Trend": pl.lit(pl.Series("Trend", trend_np)),
            "Direction": pl.lit(pl.Series("Direction", direction_np)),
        },
    }


def adapt_SuperTrend(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將包含絕對價格與二元狀態的 SuperTrend，轉換為無量綱的 DL/ML 特徵。
    正交分解為：多空政權水位 (Position)、趨勢軌道乖離 (Bias)、軌道逼近加速度 (Momentum)、政權混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    trend = h_output["values"]["Trend"]
    direction = h_output["values"]["Direction"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", 55)
    adapt_micro_p = params.get("adapt_micro_p", params["atr_len"])
    adapt_vol_p = params.get("adapt_vol_p", 34)

    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (多空政權水位): 宏觀趨勢方向的長線記憶
    # 語意補值: 0.0 (長線多空勢均力敵，呈現盤整政權)
    # ---------------------------------------------------------
    regime = direction.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：約束於 [-0.8, 0.8]，過濾極端固化
    feat_supertrend_position_stable = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-0.8, 0.8).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0] 的完整理論空間
    feat_supertrend_position_sensitive = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (趨勢軌道乖離): 當前價格相對於 SuperTrend 軌道的偏離百分比
    # 語意補值: 0.0 (價格完美貼合軌道死線，即將引發突破切換)
    # 統一語意: 方向與 Trend 一致，正值代表向上脫離軌道，負值代表向下脫離
    # ---------------------------------------------------------
    bias = (close - trend) / (trend + epsilon)

    # Stable 版：約束於 [-0.1, 0.1]，專注於價格與軌道 10% 內的微觀博弈
    feat_supertrend_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.1, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.3, 0.3]，捕捉暴拉/暴跌時遠遠甩開防守線的極端延伸
    feat_supertrend_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.3, 0.3).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (軌道逼近加速度): Bias 的變化速度 (一階導數正規化)
    # 語意補值: 0.0 (價格距離軌道維持等速發展)
    # 降共線性處理: 減去自身的 EMA 並自適應標準化，極致敏銳捕捉「價格極速撞擊死線」的動能
    # ---------------------------------------------------------
    ema_bias = bias.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (bias - ema_bias) / (ema_bias.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0]
    feat_supertrend_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]
    feat_supertrend_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (政權切換混沌度): Direction (+1/-1) 狀態的歷史變異數
    # 語意補值: 0.0 (趨勢死心塌地維持多或空，極度順暢)
    # 防禦處理: 強制套用 log1p 平滑
    # 若數值飆高，代表市場在頻繁觸發假突破、SuperTrend 不斷翻轉的「雙巴絞肉機」行情
    # ---------------------------------------------------------
    dir_volatility = direction.rolling_std(window_size=adapt_vol_p)
    log_dir_vol = dir_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_supertrend_volatility_stable = (
        log_dir_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]
    feat_supertrend_volatility_sensitive = (
        log_dir_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_supertrend_position_stable": feat_supertrend_position_stable,
        "feat_supertrend_position_sensitive": feat_supertrend_position_sensitive,
        "feat_supertrend_bias_stable": feat_supertrend_bias_stable,
        "feat_supertrend_bias_sensitive": feat_supertrend_bias_sensitive,
        "feat_supertrend_momentum_stable": feat_supertrend_momentum_stable,
        "feat_supertrend_momentum_sensitive": feat_supertrend_momentum_sensitive,
        "feat_supertrend_volatility_stable": feat_supertrend_volatility_stable,
        "feat_supertrend_volatility_sensitive": feat_supertrend_volatility_sensitive,
    }
