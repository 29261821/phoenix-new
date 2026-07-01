# ==============================================================================
# § 指標 | 相對強度 (Relative Strength)
# 核心職責: 計算一個資產相對於另一個基準資產的價格表現比率與動能。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| series_b      | H & G | str  | -        | 無 (必填)       | 作為基準比較的資產價格欄位 |
| period        | H & G | int  | 10 ~ 30  | 無 (必填)       | 相對比率的 RSI 計算週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | period 參數值   | 用於 Bias (強弱宏觀乖離) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (強弱翻轉加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 13 ~ 34  | period 參數值   | 用於 Volatility (強弱混沌度) 的滾動標準差週期 |

【特徵工程說明】
- RS 原始輸出為 0~100 的指標。G 接口將其中心化並縮放至 [-1.0, 1.0] 以符合神經網路胃口。
- 透過 adapt_macro_p 觀察資產相對強弱表現偏離其歷史均值的程度，捕捉跨市場資金板塊輪動。
"""
from typing import Dict

import polars as pl

from src.features.functions.shift import calculate as prev
from src.features.functions.wma import calculate as wma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算主資產 ('close') 相對於基準資產 ('series_b') 的 RSI 表現。
    保留原始的 0~100 數值，供傳統量化腳本作為強弱濾網 (如 RS > 50 代表強於大盤)。

    契約：
    - df 必須包含 'close' 欄位，以及 params['series_b'] 指定的基準資產欄位。
    - params 必須包含 'series_b', 'period' 鍵。
    - [健壯性] 假定主資產的價格序列為 'close'。
    """
    series_b_col = params["series_b"]
    period = params["period"]
    epsilon = 1e-9

    series_a = pl.col("close")
    series_b = pl.col(series_b_col)

    # 1. 計算價格比率
    rs_ratio = series_a / (series_b + epsilon)

    # 2. 對價格比率應用 RSI 計算邏輯
    delta = rs_ratio - prev(series=rs_ratio, period=1)
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)

    avg_gain = wma(series=gain, length=period)
    avg_loss = wma(series=loss, length=period)

    rs = avg_gain / (avg_loss + epsilon)
    rs_val = 100 - (100 / (1 + rs))

    return {"type": "scalar", "values": {"RS": rs_val}}


def adapt_Relative_Strength(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將 [0, 100] 的絕對強弱指標轉換為 [-1, 1] 的無量綱時空特徵。
    正交分解為：相對強弱水位 (Position)、強弱宏觀乖離 (Bias)、強弱翻轉加速度 (Momentum)、強弱混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    rs_val = h_output["values"]["RS"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["period"])
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", params["period"])

    epsilon = 1e-6

    # 【核心：中心化與縮放】
    # 將 0~100 的 RSI 映射為 -1.0 ~ 1.0 的對稱空間，0 代表與基準資產表現同步
    centered_rs = (rs_val - 50.0) / 50.0

    # ---------------------------------------------------------
    # (A) Position (相對強弱水位): 資產強弱表現的絕對座標
    # 語意補值: 0.0 (代表與基準大盤完全同步，無超額報酬)
    # ---------------------------------------------------------
    # Stable 版：嚴格約束於 [-1.0, 1.0]
    feat_rs_position_stable = (
        centered_rs.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：略微放寬至 [-1.2, 1.2]，包容數值抖動
    feat_rs_position_sensitive = (
        centered_rs.fill_nan(0.0).fill_null(0.0).clip(-1.2, 1.2).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (強弱宏觀乖離): 強弱表現相對於其長線政權的背離
    # 語意補值: 0.0 (當前強弱表現與近期歷史趨勢一致)
    # ---------------------------------------------------------
    rs_ema_macro = centered_rs.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = centered_rs - rs_ema_macro

    # Stable 版：約束於 [-0.5, 0.5]
    feat_rs_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]，捕捉資金板塊瞬間暴力移轉的極端斷層
    feat_rs_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (強弱翻轉加速度): 相對強弱的變化速度
    # 語意補值: 0.0 (強弱關係維持現狀)
    # ---------------------------------------------------------
    ema_centered_rs = centered_rs.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (centered_rs - ema_centered_rs) / (ema_centered_rs.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_rs_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯變盤瞬間強大的補漲或補跌動能
    feat_rs_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (強弱混沌度): 兩者相對表現的歷史變異數
    # 語意補值: 0.0 (兩者關聯極度平穩，呈現完美的死水同調或死心塌地背離)
    # ---------------------------------------------------------
    rs_vol = centered_rs.rolling_std(window_size=adapt_vol_p)
    log_rs_vol = rs_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_rs_volatility_stable = (
        log_rs_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]
    feat_rs_volatility_sensitive = (
        log_rs_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_rs_position_stable": feat_rs_position_stable,
        "feat_rs_position_sensitive": feat_rs_position_sensitive,
        "feat_rs_bias_stable": feat_rs_bias_stable,
        "feat_rs_bias_sensitive": feat_rs_bias_sensitive,
        "feat_rs_momentum_stable": feat_rs_momentum_stable,
        "feat_rs_momentum_sensitive": feat_rs_momentum_sensitive,
        "feat_rs_volatility_stable": feat_rs_volatility_stable,
        "feat_rs_volatility_sensitive": feat_rs_volatility_sensitive,
    }
