# ==============================================================================
# § 指標 | 唐奇安趨勢 (Donchian Trend)
# 核心職責: 根據價格是否突破唐奇安通道的歷史極值，建立具備記憶效應的趨勢狀態機。
# v2.0 更新: [健壯性修正] 移除 to_numpy 的 zero_copy_only 限制。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| period        | H & G | int  | 10 ~ 50  | 無 (必填)       | 唐奇安通道的極值觀察週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 89  | 21              | 用於 Bias (趨勢政權固化度) 的長線 EMA 衰減週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | 34              | 用於 Volatility (洗盤混沌度) 的滾動標準差週期 |

【特徵工程說明】
- 原始狀態機輸出為離散的 -1, 0, 1。G 接口將其擴展為連續的政權特徵。
- 透過 adapt_macro_p 觀察趨勢的「深入人心」程度 (長期中樞)。
- 透過 adapt_vol_p 計算狀態機切換的頻率變異數，識別單邊行情或高頻雙巴洗盤 (Whipsaw)。
"""
from typing import Dict

import numpy as np
import polars as pl

from src.features.functions.highest import calculate as highest
from src.features.functions.lowest import calculate as lowest
from src.features.functions.shift import calculate as prev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算唐奇安趨勢狀態機。
    採用迭代計算，完美復刻 DSL 的 prev(trend, 1) 狀態遞迴邏輯。
    保留絕對的狀態碼 (1: 多頭, -1: 空頭, 0: 無)，並包裝為 pl.Expr 以相容系統合約。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含 'period' 鍵。
    """
    period = params["period"]

    # 1. 預先計算所有無狀態依賴
    df_with_deps = df.with_columns(
        prev_upper=prev(series=highest(series=pl.col("high"), period=period), period=1),
        prev_lower=prev(series=lowest(series=pl.col("low"), period=period), period=1),
    ).fill_null(
        0
    )  # 填充 NaN 以便 NumPy 處理

    # 2. 轉換為 NumPy 以進行高效的迭代計算
    close_np = df_with_deps["close"].to_numpy()
    prev_upper_np = df_with_deps["prev_upper"].to_numpy()
    prev_lower_np = df_with_deps["prev_lower"].to_numpy()

    n = len(df)
    trend_np = np.zeros(n, dtype=np.int8)

    # 3. 初始化狀態 (100% 復刻 `var: trend(0);` 的 .fill_null(0) 行為)
    prev_trend = 0

    # 4. 執行迭代計算
    for i in range(n):
        if close_np[i] > prev_upper_np[i] and prev_upper_np[i] != 0:
            trend_np[i] = 1
        elif close_np[i] < prev_lower_np[i] and prev_lower_np[i] != 0:
            trend_np[i] = -1
        else:
            trend_np[i] = prev_trend

        # 更新狀態以供下一次迭代使用
        prev_trend = trend_np[i]

    return {
        "type": "event",
        "values": {
            # 將 numpy array 包裝為 pl.Expr
            "Trend": pl.lit(pl.Series("Trend", trend_np))
        },
    }


def adapt_Donchian_Trend(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將離散的唐奇安狀態機 (-1, 1) 轉換為連續的 DL/ML 寬表特徵。
    正交分解為：政權狀態 (Position)、政權固化度 (Bias)、翻轉脈衝 (Momentum)、洗盤頻率 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端洗盤) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，政權衰減與混沌觀察週期全面可由 YAML 配置。
    """
    trend = h_output["values"]["Trend"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    # 唐奇安趨勢為狀態機，衰減與變異數分析通常需要固定的宏觀視角，故提供預設常數 21 與 34
    adapt_macro_p = params.get("adapt_macro_p", 21)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    # 防禦性常數
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (唐奇安政權狀態): 原始的宏觀趨勢狀態
    # 語意補值: 0.0 (無趨勢)
    # ---------------------------------------------------------
    # Stable & Sensitive 版在原始狀態上差異不大，均約束於 [-1.0, 1.0]
    feat_donchian_trend_position_stable = (
        trend.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    feat_donchian_trend_position_sensitive = (
        trend.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (趨勢政權固化度): 趨勢狀態的長週期衰減中樞
    # 語意補值: 0.0 (多空勢均力敵或頻繁切換)
    # ---------------------------------------------------------
    # 使用 EMA 觀察趨勢的「深入人心」程度
    regime_solidity = trend.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：約束於 [-0.8, 0.8]，過濾極端絕對固化，穩定權重
    feat_donchian_trend_bias_stable = (
        regime_solidity.fill_nan(0.0).fill_null(0.0).clip(-0.8, 0.8).cast(pl.Float64)
    )
    # Sensitive 版：允許 [-1.0, 1.0]，捕捉史詩級單邊牛熊市的絕對統治
    feat_donchian_trend_bias_sensitive = (
        regime_solidity.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (趨勢翻轉脈衝): 狀態機的瞬間切換 (一階導數)
    # 語意補值: 0.0 (趨勢未發生切換)
    # 處理手法: 發生 1 與 -1 互換時，差值會是 2 或 -2，將其除以 2.0 正規化
    # ---------------------------------------------------------
    impulse = (trend - trend.shift(1)) / 2.0

    # Stable 版：嚴格約束於 [-1.0, 1.0]
    feat_donchian_trend_momentum_stable = (
        impulse.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束，容許樹模型捕捉異常狀態抖動
    feat_donchian_trend_momentum_sensitive = (
        impulse.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (趨勢洗盤頻率/混沌度): 狀態機的歷史切換頻率
    # 語意補值: 0.0 (單邊趨勢順暢，無反覆洗盤)
    # 防禦處理: 強制套用 log1p 平滑滾動變異數
    # ---------------------------------------------------------
    # 如果市場在盤整期頻繁觸發假突破，Rolling Std 會飆高，代表高混沌狀態 (Choppiness)
    trend_volatility = trend.rolling_std(window_size=adapt_vol_p)
    log_trend_vol = trend_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.4]
    feat_donchian_trend_volatility_stable = (
        log_trend_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.4).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留極端雙巴洗盤 (Whipsaw) 行情特徵
    feat_donchian_trend_volatility_sensitive = (
        log_trend_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_donchian_trend_position_stable": feat_donchian_trend_position_stable,
        "feat_donchian_trend_position_sensitive": feat_donchian_trend_position_sensitive,
        "feat_donchian_trend_bias_stable": feat_donchian_trend_bias_stable,
        "feat_donchian_trend_bias_sensitive": feat_donchian_trend_bias_sensitive,
        "feat_donchian_trend_momentum_stable": feat_donchian_trend_momentum_stable,
        "feat_donchian_trend_momentum_sensitive": feat_donchian_trend_momentum_sensitive,
        "feat_donchian_trend_volatility_stable": feat_donchian_trend_volatility_stable,
        "feat_donchian_trend_volatility_sensitive": feat_donchian_trend_volatility_sensitive,
    }
