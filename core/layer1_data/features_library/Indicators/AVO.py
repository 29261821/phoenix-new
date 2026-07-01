# ==============================================================================
# § 指標 | 自適應波動率振盪器 (Adaptive Volatility Oscillator)
# 核心職責: 計算 ATR 的歷史標準差 (Z-Score)，衡量波動率的相對擴張與收斂狀態。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| atr_len       | H & G | int  | 10 ~ 21  | 無 (必填)       | 基礎 ATR 的計算週期 |
| analysis_period| H & G | int  | 30 ~ 100 | 無 (必填)       | 觀察歷史波動率的長週期 (Z-Score 窗口) |
| adapt_macro_p | G 專用| int  | 9 ~ 21   | analysis_period | 用於 Bias (乖離) 計算時的基準中樞週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | atr_len 參數的值| 用於 Momentum (動量) 計算時的 EMA 平滑週期，隔離共線性 |
| adapt_vol_p   | G 專用| int  | 14 ~ 34  | analysis_period | 用於 Volatility (波動的波動) 的滾動標準差觀察週期 |

【特徵工程說明】
- AVO 本身為 Z-Score，已無尺度邊界。
- 透過 adapt_macro_p 衡量波動率偏離其「近期常態」的程度。
- 透過 adapt_vol_p 計算 Vol of Vol，評估市場波動率引擎本身的穩定性。
"""
from typing import Dict

import polars as pl

from src.features.functions.atr import calculate as atr
from src.features.functions.zscore import calculate as zscore


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算自適應波動率振盪器 (AVO)。
    保留原始的波動率 Z-Score 絕對數值，不進行任何截斷。
    確保舊有量化腳本可直接利用 Z-Score > 2 或 < -2 等絕對門檻進行波動率濾網判斷。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含 'atr_len', 'analysis_period' 鍵。
    """
    atr_len = params["atr_len"]
    analysis_period = params["analysis_period"]

    metric = atr(df=df, period=atr_len)
    zscore_val = zscore(series=metric, period=analysis_period)

    return {"type": "scalar", "values": {"ZScore": zscore_val}}


def adapt_AVO(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將單維的波動率 Z-Score 進行高階特徵萃取，轉化為 DL/ML 寬表特徵。
    正交分解為：歷史座標 (Position)、短線乖離 (Bias)、爆發動能 (Momentum) 與 波動的波動 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉黑天鵝) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，所有滾動週期全面可由 YAML 配置。
    """
    zscore_val = h_output["values"]["ZScore"]

    # 1. 提取基礎參數
    atr_len = params["atr_len"]
    analysis_period = params["analysis_period"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", analysis_period)
    adapt_micro_p = params.get("adapt_micro_p", atr_len)
    adapt_vol_p = params.get("adapt_vol_p", analysis_period)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (波動率歷史座標): 波動率的絕對 Z-Score 分位
    # 語意補值: 0.0 (代表波動率處於完美的歷史均值)
    # ---------------------------------------------------------
    # Stable 版：約束於 [-3.0, 3.0] (涵蓋 99.7% 常態分佈，穩定 Transformer 權重)
    feat_avo_position_stable = (
        zscore_val.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-6.0, 6.0] (允許捕捉如熔斷、閃崩時的極端黑天鵝波動)
    feat_avo_position_sensitive = (
        zscore_val.fill_nan(0.0).fill_null(0.0).clip(-6.0, 6.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (波動率短線乖離): AVO 相對於自身短線均線的突發偏移
    # 語意補值: 0.0 (波動率發展平穩，無突波)
    # ---------------------------------------------------------
    avo_sma = zscore_val.rolling_mean(window_size=adapt_macro_p)
    bias = zscore_val - avo_sma

    # Stable 版：約束於 [-1.0, 1.0] (關注常規的波動率微幅偏移)
    feat_avo_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-3.0, 3.0] (捕捉瞬間波動率的暴力抽升或急凍)
    feat_avo_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (波動率爆發動能): 波動率的加速度 (一階導數正規化)
    # 語意補值: 0.0 (波動率無加速擴張/收斂現象)
    # 降共線性處理: 減去自身的 EMA 並進行自適應標準化
    # ---------------------------------------------------------
    ema_avo = zscore_val.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (zscore_val - ema_avo) / (ema_avo.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_avo_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯波動率引擎的瞬間爆發力
    feat_avo_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (波動率的波動 / Vol of Vol): AVO 自身的歷史變異數
    # 語意補值: 0.0 (波動率狀態極度穩定，死水一灘)
    # 防禦處理: 強制套用 log1p 平滑極端的變異數爆炸
    # ---------------------------------------------------------
    avo_volatility = zscore_val.rolling_std(window_size=adapt_vol_p)
    log_avo_vol = avo_volatility.log1p()

    # Stable 版：約束於 [0.0, 1.0]
    feat_avo_volatility_stable = (
        log_avo_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 3.0]，保留多空激烈交戰導致的波動率狀態失控資訊
    feat_avo_volatility_sensitive = (
        log_avo_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 3.0).cast(pl.Float64)
    )

    return {
        "feat_avo_position_stable": feat_avo_position_stable,
        "feat_avo_position_sensitive": feat_avo_position_sensitive,
        "feat_avo_bias_stable": feat_avo_bias_stable,
        "feat_avo_bias_sensitive": feat_avo_bias_sensitive,
        "feat_avo_momentum_stable": feat_avo_momentum_stable,
        "feat_avo_momentum_sensitive": feat_avo_momentum_sensitive,
        "feat_avo_volatility_stable": feat_avo_volatility_stable,
        "feat_avo_volatility_sensitive": feat_avo_volatility_sensitive,
    }
