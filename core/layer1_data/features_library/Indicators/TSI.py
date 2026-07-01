# ==============================================================================
# § 指標 | 真實強度指數 (True Strength Index, TSI)
# 核心職責: 一個經過雙重平滑的動能指標，捕捉趨勢的真實力量，並減少雜訊。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| source        | H & G | str  | -        | 無 (必填)       | 價格來源 (如 'close') |
| long          | H & G | int  | 20 ~ 50  | 無 (必填)       | 第一重平滑 (長週期 EMA) |
| short         | H & G | int  | 5 ~ 21   | 無 (必填)       | 第二重平滑 (短週期 EMA) |
| signal        | H & G | int  | 5 ~ 13   | 無 (必填)       | 信號線 (EMA) 週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | long 參數值     | (未直接使用，保留做系統對齊) |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | signal 參數值   | 用於 Momentum (翻轉加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 13 ~ 34  | short 參數值    | 用於 Volatility (動能混沌度) 的滾動標準差週期 |

【特徵工程說明】
- TSI 原始輸出為 -100 ~ 100。G 接口將其除以 100，完美縮放至 [-1.0, 1.0] 的神經網路對稱空間。
- 透過 TSI 與 Signal 的差值構成 Bias (信號乖離)，反映 MACD-like 的發散與收斂特徵。
- 透過 adapt_micro_p 提早捕捉 TSI 在超買超賣區的彎折與反向爆發加速度。
"""
from typing import Dict

import polars as pl

from src.features.functions.abs import calculate as abs_val
from src.features.functions.ema import calculate as ema
from src.features.functions.shift import calculate as prev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的 TSI 數值 (-100 ~ 100)。
    供傳統量化腳本作為過濾器 (如 TSI 向上穿越 Signal 線視為黃金交叉) 調用。

    契約：
    - df 必須包含 params['source'] 指定的欄位。
    - params 必須包含 'source', 'long', 'short', 'signal' 鍵。
    """
    source_col, long, short, signal = (
        params["source"],
        params["long"],
        params["short"],
        params["signal"],
    )
    source = pl.col(source_col)
    epsilon = 1e-9

    pc = source - prev(series=source, period=1)
    pc_abs = abs_val(series=pc)

    first_smooth = ema(series=pc, length=long)
    double_smooth = ema(series=first_smooth, length=short)

    first_smooth_abs = ema(series=pc_abs, length=long)
    double_smooth_abs = ema(series=first_smooth_abs, length=short)

    tsi_val = 100 * double_smooth / (double_smooth_abs + epsilon)
    signal_line = ema(series=tsi_val, length=signal)

    return {"type": "vector", "values": {"TSI": tsi_val, "Signal": signal_line}}


def adapt_TSI(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將 [-100, 100] 的雙重平滑動能指標轉換為 [-1, 1] 的無量綱時空特徵。
    正交分解為：真實動能水位 (Position)、信號線乖離 (Bias)、翻轉加速度 (Momentum)、動能混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    tsi_val = h_output["values"]["TSI"].cast(pl.Float64)
    signal_line = h_output["values"]["Signal"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_micro_p = params.get("adapt_micro_p", params["signal"])
    adapt_vol_p = params.get("adapt_vol_p", params["short"])

    epsilon = 1e-6

    # 【核心：中心化與縮放】
    # 將 -100~100 映射為 -1.0 ~ 1.0 的對稱空間
    norm_tsi = tdfi_val = tsi_val / 100.0
    norm_signal = signal_line / 100.0

    # ---------------------------------------------------------
    # (A) Position (真實動能水位): TSI 主線的相對位置
    # 語意補值: 0.0 (代表動能處於多空平衡的中立區)
    # ---------------------------------------------------------
    # Stable 版：嚴格約束於 [-1.0, 1.0]
    feat_tsi_position_stable = (
        norm_tsi.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：略微放寬至 [-1.2, 1.2]，包容數值微小抖動
    feat_tsi_position_sensitive = (
        norm_tsi.fill_nan(0.0).fill_null(0.0).clip(-1.2, 1.2).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (信號線乖離): TSI 與 Signal 線的發散程度 (類似 MACD Hist)
    # 語意補值: 0.0 (TSI 與信號線完美貼合，可能即將交叉)
    # ---------------------------------------------------------
    bias = norm_tsi - norm_signal

    # Stable 版：約束於 [-0.5, 0.5]
    feat_tsi_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]，捕捉暴拉/暴跌產生的極端背離空間
    feat_tsi_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (翻轉加速度): 核心 TSI 的變化速度 (一階導數)
    # 語意補值: 0.0 (動能維持等速或陷入鈍化)
    # 降共線性處理: 減去短線 EMA 並自適應標準化
    # ---------------------------------------------------------
    ema_norm_tsi = norm_tsi.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (norm_tsi - ema_norm_tsi) / (ema_norm_tsi.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0]
    feat_tsi_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極度凸顯 V 轉或 A 轉瞬間強大的反轉加速度
    feat_tsi_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (動能混沌度): TSI 信號的歷史變異數
    # 語意補值: 0.0 (動能維持單向推進，極度順暢)
    # 防禦處理: 強制套用 log1p 平滑
    # ---------------------------------------------------------
    tsi_vol = norm_tsi.rolling_std(window_size=adapt_vol_p)
    log_tsi_vol = tsi_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_tsi_volatility_stable = (
        log_tsi_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留多空反覆爭奪的混沌狀態
    feat_tsi_volatility_sensitive = (
        log_tsi_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_tsi_position_stable": feat_tsi_position_stable,
        "feat_tsi_position_sensitive": feat_tsi_position_sensitive,
        "feat_tsi_bias_stable": feat_tsi_bias_stable,
        "feat_tsi_bias_sensitive": feat_tsi_bias_sensitive,
        "feat_tsi_momentum_stable": feat_tsi_momentum_stable,
        "feat_tsi_momentum_sensitive": feat_tsi_momentum_sensitive,
        "feat_tsi_volatility_stable": feat_tsi_volatility_stable,
        "feat_tsi_volatility_sensitive": feat_tsi_volatility_sensitive,
    }
