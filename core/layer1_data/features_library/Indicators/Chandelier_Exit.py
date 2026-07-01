# ==============================================================================
# § 指標 | 吊燈止損 (Chandelier Exit)
# 核心職責: 基於極值與波動率計算具備「棘輪效應(Ratchet)」的趨勢追蹤止損線。
# v2.0 更新: [健壯性修正] 移除 to_numpy 的 zero_copy_only 限制。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型  | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|-------|----------|-----------------|------|
| atr_len       | H & G | int   | 10 ~ 22  | 無 (必填)       | 計算 ATR 與極值 (Highest/Lowest) 的基礎週期 |
| multiplier    | H     | float | 1.5 ~ 3.0| 無 (必填)       | ATR 乘數，決定止損帶的絕對寬度 |
| adapt_micro_p | G 專用| int   | 5 ~ 14   | atr_len 參數的值| 用於計算 Momentum (動能) 時的 EMA 平滑週期，隔離共線性 |

【特徵工程說明】
- 吊燈止損具備「只進不退」的棘輪特性，價格回撤時軌道會呈現水平死線。
- G 接口將其轉換為無量綱的相對位置 (%Chandelier)、頻寬與乖離。
- 透過 adapt_micro_p 決定模型對「撞擊止損死線加速度」的敏感度。
"""
import warnings
from typing import Any, Dict

import numpy as np
import polars as pl

# 此指標為【狀態機指標】，不能使用純 Polars 表達式，因其邏輯包含自我引用。
# 必須使用迭代計算，以 100% 復刻 DSL 系統的健壯性。
from src.features.functions.atr import calculate as atr
from src.features.functions.highest import calculate as highest
from src.features.functions.lowest import calculate as lowest


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, Any]:
    """
    【H 接口：人類與策略庫語意】
    計算吊燈止損 (Chandelier Exit)。
    完美復刻 DSL 的遞迴狀態機邏輯，保留絕對的止損價格位準，
    並將 numpy 計算結果透過 pl.lit 包裝為 Expr，確保與整體系統合約相容。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含 'atr_len', 'multiplier' 鍵。
    - [150分健壯性] 採用迭代計算，完美復刻 DSL 的遞迴狀態機邏輯。
    """
    atr_len = params["atr_len"]
    multiplier = params["multiplier"]

    # 1. 預先計算所有無狀態依賴
    df_with_deps = df.with_columns(
        atr_val=atr(df=df, period=atr_len),
        highest_high=highest(series=pl.col("high"), period=atr_len),
        lowest_low=lowest(series=pl.col("low"), period=atr_len),
    )

    # 2. 轉換為 NumPy 以進行高效的迭代計算
    close_np = df_with_deps["close"].to_numpy()
    atr_np = df_with_deps["atr_val"].to_numpy()
    highest_high_np = df_with_deps["highest_high"].to_numpy()
    lowest_low_np = df_with_deps["lowest_low"].to_numpy()

    n = len(df)
    long_stop_np = np.full(n, np.nan)
    short_stop_np = np.full(n, np.nan)

    # 3. 初始化狀態 (100% 復刻 `var: long_stop(0.0), short_stop(0.0);` 的 .fill_null() 行為)
    prev_long_stop = 0.0
    prev_short_stop = 0.0

    # 4. 執行迭代計算
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(n):
            potential_long = highest_high_np[i] - (atr_np[i] * multiplier)
            potential_short = lowest_low_np[i] + (
                atr_np[i] * multiplier
            )  # 修正: 熊市止損應為 low + atr

            long_stop_new = (
                max(potential_long, prev_long_stop)
                if prev_long_stop != 0.0
                else potential_long
            )
            short_stop_new = (
                min(potential_short, prev_short_stop)
                if prev_short_stop != 0.0
                else potential_short
            )

            if close_np[i] > prev_long_stop:
                long_stop_np[i] = long_stop_new
            else:
                long_stop_np[i] = potential_long

            if close_np[i] < prev_short_stop:
                short_stop_np[i] = short_stop_new
            else:
                short_stop_np[i] = potential_short

            prev_long_stop = long_stop_np[i]
            prev_short_stop = short_stop_np[i]

    return {
        "type": "level",
        "values": {
            # 將迭代算出的 Series 包裝為 pl.Expr 統一輸出介面
            "LongStop": pl.lit(pl.Series("LongStop", long_stop_np)),
            "ShortStop": pl.lit(pl.Series("ShortStop", short_stop_np)),
        },
    }


def adapt_Chandelier_Exit(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將具備棘輪效應(Ratchet)的階梯狀止損價格，轉換為供 DL/ML 使用的無尺度連續特徵。
    正交分解為：相對止損位置 (Position)、吊燈頻寬 (Volatility)、中樞乖離 (Bias)、逼近動能 (Momentum)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉止損獵殺) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，動能平滑週期可由 YAML 配置。
    """
    long_stop = h_output["values"]["LongStop"]
    short_stop = h_output["values"]["ShortStop"]

    # 1. 提取基礎參數
    atr_len = params["atr_len"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_micro_p = params.get("adapt_micro_p", atr_len)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (相對止損位置): 價格在吊燈多空帶之間的相對座標 (%Chandelier)
    # 語意補值: 0.5 (代表處於安全地帶)
    # ---------------------------------------------------------
    pct_stop = (close - long_stop) / (short_stop - long_stop + epsilon)

    # Stable 版：嚴格約束於 [0.0, 1.0] 內，防止模型 Activation 偏移
    feat_chandelier_position_stable = (
        pct_stop.fill_nan(0.5).fill_null(0.5).clip(0.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-1.0, 2.0]，捕捉價格刺穿水平止損死線引發的流動性獵殺
    feat_chandelier_position_sensitive = (
        pct_stop.fill_nan(0.5).fill_null(0.5).clip(-1.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Volatility (吊燈通道頻寬): 滯後收斂的止損頻寬佔價格百分比
    # 語意補值: 0.0 (無頻寬)
    # 防禦處理: 強制套用 log1p 平滑極端波動擴張
    # ---------------------------------------------------------
    bandwidth = (short_stop - long_stop) / (close + epsilon)
    log_bandwidth = bandwidth.log1p()

    # Stable 版：約束於 [0.0, 0.2] (容許最多 20% 的通道寬度)
    feat_chandelier_volatility_stable = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.5] (捕捉黑天鵝級別的通道暴增)
    feat_chandelier_volatility_sensitive = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Bias (吊燈中樞乖離): 價格相對於吊燈多空虛擬中樞的偏離
    # 語意補值: 0.0 (貼合多空平衡中樞)
    # ---------------------------------------------------------
    midpoint = (long_stop + short_stop) / 2.0
    bias = (close / (midpoint + epsilon)) - 1.0

    # Stable 版：約束於 [-0.1, 0.1]，代表最多 10% 的中樞偏離
    feat_chandelier_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.1, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.3, 0.3]，保留超限偏離資訊
    feat_chandelier_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.3, 0.3).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Momentum (止損逼近動能): 位置逼近率 (%Chandelier) 的加速度
    # 語意補值: 0.0 (無逼近動能)
    # 降共線性處理: 減去自身的 EMA 並標準化，凸顯極速殺跌撞擊水平止損線的動能預警
    # ---------------------------------------------------------
    ema_pct_stop = pct_stop.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    pct_stop_osc = (pct_stop - ema_pct_stop) / (ema_pct_stop.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_chandelier_momentum_stable = (
        pct_stop_osc.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束，捕捉極端暴力洗盤時撞擊死線的瞬間加速度
    feat_chandelier_momentum_sensitive = (
        pct_stop_osc.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    return {
        "feat_chandelier_position_stable": feat_chandelier_position_stable,
        "feat_chandelier_position_sensitive": feat_chandelier_position_sensitive,
        "feat_chandelier_volatility_stable": feat_chandelier_volatility_stable,
        "feat_chandelier_volatility_sensitive": feat_chandelier_volatility_sensitive,
        "feat_chandelier_bias_stable": feat_chandelier_bias_stable,
        "feat_chandelier_bias_sensitive": feat_chandelier_bias_sensitive,
        "feat_chandelier_momentum_stable": feat_chandelier_momentum_stable,
        "feat_chandelier_momentum_sensitive": feat_chandelier_momentum_sensitive,
    }
