# ==============================================================================
# § 指標 | 平均 K 線 (Heikin Ashi)
# 核心職責: 過濾市場雜訊，透過價格平滑呈現更純粹的趨勢方向與強度。
# v2.0 更新: [健壯性修正] 移除 to_numpy 的 zero_copy_only 限制。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (純度加速度) 計算時的 EMA 平滑週期 |
| adapt_vol_p   | G 專用| int  | 13 ~ 34  | 21              | 用於 Volatility (實體混沌度) 的滾動標準差週期 |

【特徵工程說明】
- Heikin Ashi 本身無基礎參數，完全依賴開高低收的遞迴計算。
- 原始 HA 帶有絕對價格尺度，G 接口將其轉換為無量綱的實體純度 (Position)、實體變異 (Volatility)、真實乖離 (Bias) 與純度加速度 (Momentum)。
"""
from typing import Dict

import numpy as np
import polars as pl


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算 Heikin Ashi K 線。
    採用迭代計算，完美復刻 DSL 的遞迴狀態機契約。
    保留絕對的價格位準，將 NumPy 陣列包裝為 pl.Expr 供後續無縫調用。

    契約：
    - df 必須包含 'open', 'high', 'low', 'close' 欄位。
    - params: 此指標無輸入參數 (特徵工程參數由 adapt 層接收)。
    """
    o, h, l, c = (
        df["open"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
    )
    n = len(df)
    ha_open = np.zeros(n)
    ha_close = (o + h + l + c) / 4.0

    if n > 0:
        ha_open[0] = (o[0] + c[0]) / 2.0

        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])

    return {
        "type": "vector",
        "values": {
            "HA_Open": pl.lit(pl.Series("HA_Open", ha_open)),
            "HA_High": pl.lit(pl.Series("HA_High", ha_high)),
            "HA_Low": pl.lit(pl.Series("HA_Low", ha_low)),
            "HA_Close": pl.lit(pl.Series("HA_Close", ha_close)),
        },
    }


def adapt_Heikin_Ashi(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將帶有價格尺度的 Heikin Ashi 轉換為 DL/ML 可學習的連續時空特徵。
    正交分解為：趨勢純度 (Position)、真實價格乖離 (Bias)、純度加速度 (Momentum)、實體混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端趨勢) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，動能與變異數週期可由 YAML 配置。
    """
    ha_open = h_output["values"]["HA_Open"]
    ha_high = h_output["values"]["HA_High"]
    ha_low = h_output["values"]["HA_Low"]
    ha_close = h_output["values"]["HA_Close"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    # HA 無基礎參數，故直接賦予合理的預設常數 5 與 21
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", 21)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (趨勢純度): HA 實體長度佔全體 K 線範圍的比例
    # 語意補值: 0.0 (代表十字星，多空完全拉扯)
    # 理論上 1.0 代表完美的光頭光腳大多頭，-1.0 代表光頭光腳大空頭
    # ---------------------------------------------------------
    ha_body = ha_close - ha_open
    ha_range = ha_high - ha_low
    purity = ha_body / (ha_range + epsilon)

    # Stable & Sensitive 版：先天理論邊界即為 [-1.0, 1.0]，無極端值問題
    feat_ha_position_stable = (
        purity.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    feat_ha_position_sensitive = (
        purity.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (真實價格乖離): 真實收盤價相對於 HA 收盤價的偏離
    # 語意補值: 0.0 (真實價格與平滑趨勢完美貼合)
    # 衡量 HA 是否過度平滑導致嚴重落後真實市場 (例如突發性的 V 型反轉)
    # ---------------------------------------------------------
    bias = (close / (ha_close + epsilon)) - 1.0

    # Stable 版：約束於 [-0.05, 0.05]，過濾掉雜訊，專注於微小且常規的乖離
    feat_ha_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.05, 0.05).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.2, 0.2]，捕捉市場暴跌/暴漲時 HA K線來不及反應的極端落後
    feat_ha_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (純度加速度): 趨勢純度 (Position) 的變化速度
    # 語意補值: 0.0 (趨勢純度維持平穩，無轉強或衰退)
    # 降共線性處理: 減去自身的短線 EMA 並自適應標準化
    # ---------------------------------------------------------
    ema_purity = purity.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (purity - ema_purity) / (ema_purity.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_ha_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，捕捉多空純度瞬間翻轉的爆發動能
    feat_ha_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (實體混沌度): 趨勢純度的歷史變異數
    # 語意補值: 0.0 (市場處於極度穩定的單邊趨勢，純度恆定)
    # 防禦處理: 強制套用 log1p 平滑
    # 衡量市場是在穩定的單邊趨勢中，還是處於上下影線極長、陰陽頻繁切換的洗盤區
    # ---------------------------------------------------------
    purity_volatility = purity.rolling_std(window_size=adapt_vol_p)
    log_purity_vol = purity_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_ha_volatility_stable = (
        log_purity_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留極端雙巴洗盤時的混沌特徵
    feat_ha_volatility_sensitive = (
        log_purity_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_ha_position_stable": feat_ha_position_stable,
        "feat_ha_position_sensitive": feat_ha_position_sensitive,
        "feat_ha_bias_stable": feat_ha_bias_stable,
        "feat_ha_bias_sensitive": feat_ha_bias_sensitive,
        "feat_ha_momentum_stable": feat_ha_momentum_stable,
        "feat_ha_momentum_sensitive": feat_ha_momentum_sensitive,
        "feat_ha_volatility_stable": feat_ha_volatility_stable,
        "feat_ha_volatility_sensitive": feat_ha_volatility_sensitive,
    }
