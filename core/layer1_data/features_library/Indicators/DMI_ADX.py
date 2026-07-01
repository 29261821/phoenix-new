# ==============================================================================
# § 指標 | 動向指數 (DMI/ADX)
# 核心職責: 計算 +DI, -DI 和 ADX，衡量趨勢的方向與強度。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| di_len        | H & G | int  | 10 ~ 21  | 無 (必填)       | +DI 與 -DI 的基礎計算與平滑週期 |
| adx_len       | H & G | int  | 10 ~ 21  | 無 (必填)       | ADX (趨勢強度) 的平滑週期 |
| adapt_macro_p | G 專用| int  | 14 ~ 34  | adx_len 參數的值| 用於 Bias (趨勢強度乖離) 的長線 EMA 中樞週期 |
| adapt_micro_p | G 專用| int  | 5 ~ 14   | di_len 參數的值 | 用於 Momentum (統治力翻轉動能) 的短線 EMA 平滑週期 |

【特徵工程說明】
- 原始 DMI/ADX 為 0~100 的絕對數值，G 接口將其轉換為無量綱的正交特徵。
- 透過 adapt_macro_p 觀察 ADX (趨勢強度) 相對於歷史均值的擴張或衰竭。
- 透過 adapt_micro_p 觀察多空統治力 (+DI 與 -DI 的對抗) 的瞬間翻轉加速度。
"""
from typing import Dict

import polars as pl

from src.features.functions.atr import calculate as atr
from src.features.functions.shift import calculate as prev
from src.features.functions.wma import calculate as wma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留原始的 0~100 絕對數值 (+DI, -DI, ADX)。
    確保依賴絕對門檻 (如 ADX > 25 或 +DI > -DI) 的傳統量化策略可無縫使用。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含 'di_len', 'adx_len' 鍵。
    """
    di_len = params["di_len"]
    adx_len = params["adx_len"]
    epsilon = 1e-9

    h, l = pl.col("high"), pl.col("low")
    prev_h, prev_l = prev(series=h, period=1), prev(series=l, period=1)

    up_move = h - prev_h
    down_move = prev_l - l

    plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0)
    minus_dm = (
        pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0)
    )

    # 100% 復刻 DSL v6.0 修正案: 使用標準 ATR 函數平滑 TR
    s_tr = atr(df=df, period=di_len)
    s_plus_dm = wma(series=plus_dm, length=di_len)
    s_minus_dm = wma(series=minus_dm, length=di_len)

    plus_di = 100 * s_plus_dm / (s_tr + epsilon)
    minus_di = 100 * s_minus_dm / (s_tr + epsilon)

    dx_val = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + epsilon)
    adx_val = wma(series=dx_val, length=adx_len)

    return {
        "type": "vector",
        "values": {"PlusDI": plus_di, "MinusDI": minus_di, "ADX": adx_val},
    }


def adapt_DMI_ADX(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將 DMI/ADX 轉換為供 DL/ML 使用的無尺度、正交化穩定特徵。
    正交分解為：多空統治力 (Position)、趨勢絕對強度 (Volatility)、強度乖離 (Bias)、統治翻轉動能 (Momentum)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端行情) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，所有動能與乖離衰減週期全面可由 YAML 配置。
    """
    plus_di = h_output["values"]["PlusDI"]
    minus_di = h_output["values"]["MinusDI"]
    adx = h_output["values"]["ADX"]

    # 1. 提取基礎參數
    di_len = params["di_len"]
    adx_len = params["adx_len"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", adx_len)
    adapt_micro_p = params.get("adapt_micro_p", di_len)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (多空統治力 / 方向政權): 衡量當前由多方還是空方主導
    # 語意補值: 0.0 (多空勢均力敵)
    # 計算公式: (+DI - -DI) / (+DI + -DI)
    # ---------------------------------------------------------
    dominance = (plus_di - minus_di) / (plus_di + minus_di + epsilon)

    # Stable 版：約束於 [-0.8, 0.8]，過濾極少見的 100% 絕對統治，穩定 Transformer 注意力
    feat_dmi_position_stable = (
        dominance.fill_nan(0.0).fill_null(0.0).clip(-0.8, 0.8).cast(pl.Float64)
    )
    # Sensitive 版：保留 [-1.0, 1.0] 的完整理論極限，捕捉極端單邊行情
    feat_dmi_position_sensitive = (
        dominance.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Volatility (趨勢動能強度): ADX 的百分比標準化 (無方向性)
    # 語意補值: 0.0 (無趨勢死水)
    # ---------------------------------------------------------
    adx_norm = adx / 100.0

    # Stable 版：約束於 [0.0, 0.6] (實務上 ADX 超過 60 屬於異端，截斷以防權重偏移)
    feat_dmi_volatility_stable = (
        adx_norm.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.6).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留史詩級趨勢爆發時的極值
    feat_dmi_volatility_sensitive = (
        adx_norm.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Bias (趨勢強度乖離 / 衰竭預警): ADX 相對於其中期均線的偏離
    # 語意補值: 0.0 (趨勢強度發展平穩，無加速或減速現象)
    # ---------------------------------------------------------
    adx_ema = adx.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = (adx - adx_ema) / 100.0

    # Stable 版：約束於 [-0.2, 0.2]，關注常規的趨勢加速與減速
    feat_dmi_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.5, 0.5]，捕捉趨勢瞬間崩潰 (高檔暴跌) 的強烈信號
    feat_dmi_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Momentum (統治力翻轉速度): 多空統治力 (Position) 的加速度
    # 語意補值: 0.0 (主導權無移轉)
    # 降共線性處理: 減去自身的 EMA 並進行自適應標準化，凸顯極速搶奪主導權的行為
    # ---------------------------------------------------------
    dom_ema = dominance.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (dominance - dom_ema) / (dom_ema.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_dmi_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，捕捉多空瞬間劇烈交叉時的恐怖動能
    feat_dmi_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    return {
        "feat_dmi_position_stable": feat_dmi_position_stable,
        "feat_dmi_position_sensitive": feat_dmi_position_sensitive,
        "feat_dmi_volatility_stable": feat_dmi_volatility_stable,
        "feat_dmi_volatility_sensitive": feat_dmi_volatility_sensitive,
        "feat_dmi_bias_stable": feat_dmi_bias_stable,
        "feat_dmi_bias_sensitive": feat_dmi_bias_sensitive,
        "feat_dmi_momentum_stable": feat_dmi_momentum_stable,
        "feat_dmi_momentum_sensitive": feat_dmi_momentum_sensitive,
    }
