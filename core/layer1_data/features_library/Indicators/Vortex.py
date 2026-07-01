# ==============================================================================
# § 指標 | 渦流指標 (Vortex Indicator)
# 核心職責: 比較當前K線與前一K線的極值，識別趨勢的開始、方向與強弱。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口淨差值正交特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| period        | H & G | int  | 10 ~ 30  | 無 (必填)       | Vortex 計算的基礎週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | period 參數值   | 用於 Bias (渦流宏觀乖離) 計算的長線 EMA 週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (翻轉加速度) 計算的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 14 ~ 34  | period 參數值   | 用於 Volatility (渦流混沌度) 的滾動標準差週期 |

【特徵工程說明】
- VIP (+VI) 和 VIM (-VI) 理論上皆圍繞著 1.0 震盪。
- G 接口將其轉換為「渦流淨差值 (Net Vortex Difference = VIP - VIM)」，使其天然中心化於 0。
- 透過適配週期將淨差值正交分解為：絕對水位 (Position)、宏觀乖離 (Bias) 與 翻轉加速度 (Momentum)。
"""
from typing import Dict

import polars as pl

from src.features.functions.abs import calculate as abs_val
from src.features.functions.shift import calculate as prev
from src.features.functions.sum import calculate as rolling_sum
from src.features.functions.tr import calculate as tr


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的 VIP 與 VIM 指標值。
    供傳統量化腳本作為過濾器 (如 VIP 向上穿越 VIM 視為作多信號) 調用。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含 'period' 鍵。
    """
    period = params["period"]
    h, l = pl.col("high"), pl.col("low")
    epsilon = 1e-9

    tr_val = tr(df=df)
    vp = abs_val(series=(h - prev(series=l, period=1)))
    vm = abs_val(series=(l - prev(series=h, period=1)))

    sum_tr = rolling_sum(series=tr_val, period=period)
    sum_vp = rolling_sum(series=vp, period=period)
    sum_vm = rolling_sum(series=vm, period=period)

    vip = sum_vp / (sum_tr + epsilon)
    vim = sum_vm / (sum_tr + epsilon)

    return {"type": "vector", "values": {"VIP": vip, "VIM": vim}}


def adapt_Vortex(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將兩條絕對渦流指標轉換為 [-1, 1] 之間的無量綱單一動量特徵。
    正交分解為：渦流絕對水位 (Position)、渦流宏觀乖離 (Bias)、翻轉加速度 (Momentum)、渦流混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    vip = h_output["values"]["VIP"].cast(pl.Float64)
    vim = h_output["values"]["VIM"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["period"])
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", params["period"])

    epsilon = 1e-6

    # 【核心：中心化】
    # 計算渦流淨差值 (Net Vortex Difference)
    # 當 VIP > VIM 時，為正值(多頭)；反之為負值(空頭)。天然圍繞 0.0 震盪，通常界於 [-0.5, 0.5]
    vi_diff = vip - vim

    # ---------------------------------------------------------
    # (A) Position (渦流絕對水位): 渦流淨差值的相對位置
    # 語意補值: 0.0 (代表多空渦流力量完全抵銷)
    # ---------------------------------------------------------
    # Stable 版：嚴格約束於 [-1.0, 1.0]
    feat_vortex_position_stable = (
        vi_diff.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-1.5, 1.5]，包容極值區的異常溢出
    feat_vortex_position_sensitive = (
        vi_diff.fill_nan(0.0).fill_null(0.0).clip(-1.5, 1.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (渦流宏觀乖離): 當前淨差值相對於其長線政權的背離
    # 語意補值: 0.0 (當前動能與近期宏觀動能一致)
    # ---------------------------------------------------------
    vi_ema_macro = vi_diff.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = vi_diff - vi_ema_macro

    # Stable 版：約束於 [-0.5, 0.5]
    feat_vortex_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]，捕捉暴拉/暴跌產生的極大雙線張口
    feat_vortex_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (渦流翻轉加速度): 渦流淨差值的變化速度 (一階導數)
    # 語意補值: 0.0 (動能維持等速或在頂底陷入絕對鈍化)
    # 降共線性處理: 減去短線 EMA 並自適應標準化，提早捕捉趨勢線的彎折
    # ---------------------------------------------------------
    ema_vi_diff = vi_diff.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (vi_diff - ema_vi_diff) / (ema_vi_diff.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_vortex_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極度凸顯 V 轉或 A 轉瞬間強大的反轉加速度
    feat_vortex_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (渦流混沌度): 渦流淨差值的歷史變異數
    # 語意補值: 0.0 (動能維持單向推進，或處於絕對的平靜死水)
    # ---------------------------------------------------------
    vi_vol = vi_diff.rolling_std(window_size=adapt_vol_p)
    log_vi_vol = vi_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_vortex_volatility_stable = (
        log_vi_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留多空反覆爭奪的混沌糾纏狀態
    feat_vortex_volatility_sensitive = (
        log_vi_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_vortex_position_stable": feat_vortex_position_stable,
        "feat_vortex_position_sensitive": feat_vortex_position_sensitive,
        "feat_vortex_bias_stable": feat_vortex_bias_stable,
        "feat_vortex_bias_sensitive": feat_vortex_bias_sensitive,
        "feat_vortex_momentum_stable": feat_vortex_momentum_stable,
        "feat_vortex_momentum_sensitive": feat_vortex_momentum_sensitive,
        "feat_vortex_volatility_stable": feat_vortex_volatility_stable,
        "feat_vortex_volatility_sensitive": feat_vortex_volatility_sensitive,
    }
