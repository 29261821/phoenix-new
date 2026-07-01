# ==============================================================================
# § 指標 | 因子融合引擎 (Factor Fusion Engine)
# 核心職責: 融合多個基礎指標 (AO, RSI, CMF)，提取標準化後的市場共識因子。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| rsi_source    | H & G | str  | -        | 無 (必填)       | RSI 的計算價格來源 (如 'close') |
| rsi_period    | H & G | int  | 7 ~ 21   | 無 (必填)       | 內化 RSI 的計算週期 |
| cmf_period    | H & G | int  | 10 ~ 40  | 無 (必填)       | 內化 CMF 的計算週期 |
| ao_fast       | H & G | int  | 3 ~ 10   | 無 (必填)       | 內化 AO 的快線週期 |
| ao_slow       | H & G | int  | 20 ~ 50  | 無 (必填)       | 內化 AO 的慢線週期 |
| z_window      | H & G | int  | 30 ~ 100 | 無 (必填)       | 將三大指標標準化 (Z-Score) 的窗口長度 |
| signal_period | H & G | int  | 5 ~ 21   | 無 (必填)       | 共識因子 (Factor1) 的信號線 (Factor2) 週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (共識加速度) 計算的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 20 ~ 55  | z_window 參數值 | 用於 Volatility (共識分歧度) 的滾動標準差週期 |

【特徵工程說明】
- Fusion 天生已是 Z-Score 融合的無尺度特徵。Bias 透過 Factor1 - Factor2 計算。
- 透過 adapt_micro_p 決定模型對「突發利好/利空導致的共識瞬間引爆」的敏感度。
- 透過 adapt_vol_p 衡量三大底層指標 (AO, RSI, CMF) 的分歧與混沌狀態。
"""
from typing import Dict

import polars as pl

# [邏輯自治] 遵循 DSL v6.0 (邏輯自治終極版) 的設計思想，此指標為「自產自銷」。
from src.features.functions.ema import calculate as ema
from src.features.functions.shift import calculate as prev
from src.features.functions.sma import calculate as sma
from src.features.functions.stddev import calculate as stddev
from src.features.functions.sum import calculate as rolling_sum
from src.features.functions.wma import calculate as wma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留標準化後的共識因子 (Factor1) 與其信號線 (Factor2)。
    確保依賴「Factor1 穿越 Factor2」作為綜合多空進場濾網的舊有量化策略能無縫執行。

    契約：
    - df 必須包含 'high', 'low', 'close', 'volume' 及 params 中指定的 source 欄位。
    - params 必須包含 'rsi_source', 'rsi_period', 'cmf_period', 'ao_fast',
      'ao_slow', 'z_window', 'signal_period' 鍵。
    """
    # 解構所有參數
    rsi_source_col, rsi_period = params["rsi_source"], params["rsi_period"]
    cmf_period = params["cmf_period"]
    ao_fast, ao_slow = params["ao_fast"], params["ao_slow"]
    z_window, signal_period = params["z_window"], params["signal_period"]
    epsilon = 1e-9

    h, l, c, v = pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")
    rsi_source = pl.col(rsi_source_col)

    # --- 內化的 AO (Awesome Oscillator) 計算邏輯 ---
    hl2 = (h + l) / 2
    fast_ao = sma(series=hl2, length=ao_fast)
    slow_ao = sma(series=hl2, length=ao_slow)
    ao_val = fast_ao - slow_ao

    # --- 內化的 RSI (Relative Strength Index) 計算邏輯 ---
    delta = rsi_source - prev(series=rsi_source, period=1)
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)
    avg_gain = wma(series=gain, length=rsi_period)
    avg_loss = wma(series=loss, length=rsi_period)
    rs = avg_gain / (avg_loss + epsilon)
    rsi_val = 100 - (100 / (1 + rs))

    # --- 內化的 CMF (Chaikin Money Flow) 計算邏輯 ---
    mfm = ((c - l) - (h - c)) / (h - l + epsilon)
    mfv = mfm * v
    cmf_val = rolling_sum(series=mfv, period=cmf_period) / (
        rolling_sum(series=v, period=cmf_period) + epsilon
    )

    # --- 核心因子融合邏輯 ---
    ao_z = (ao_val - sma(series=ao_val, length=z_window)) / (
        stddev(series=ao_val, period=z_window) + epsilon
    )
    rsi_z = (rsi_val - sma(series=rsi_val, length=z_window)) / (
        stddev(series=rsi_val, period=z_window) + epsilon
    )
    cmf_z = (cmf_val - sma(series=cmf_val, length=z_window)) / (
        stddev(series=cmf_val, period=z_window) + epsilon
    )

    factor1 = (ao_z + rsi_z + cmf_z) / 3.0
    factor2 = ema(series=factor1, length=signal_period)

    return {"type": "vector", "values": {"Factor1": factor1, "Factor2": factor2}}


def adapt_Fusion(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將融合後的共識因子進行 DL/ML 寬表特徵的高階萃取。
    這是一個天然的無尺度 (Scale-invariant) 特徵，正交分解為：
    共識絕對水位 (Position)、共識宏觀乖離 (Bias)、共識加速度 (Momentum)、共識分歧度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉黑天鵝) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，所有動能與變異數週期全面可由 YAML 配置。
    """
    factor1 = h_output["values"]["Factor1"]
    factor2 = h_output["values"]["Factor2"]

    # 1. 提取基礎參數
    z_window = params["z_window"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", z_window)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (共識絕對水位): 當前三大維度 (AO, RSI, CMF) 的綜合標準化力量
    # 語意補值: 0.0 (代表多空力量完全抵銷，市場毫無共識)
    # ---------------------------------------------------------
    # Stable 版：約束於 [-3.0, 3.0] (涵蓋絕大多數的統計區間，穩定 DL 權重)
    feat_fusion_position_stable = (
        factor1.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-6.0, 6.0]，捕捉如熔斷或極端狂熱時的三指標歷史性共振
    feat_fusion_position_sensitive = (
        factor1.fill_nan(0.0).fill_null(0.0).clip(-6.0, 6.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (共識宏觀乖離 / 發散): 短期共識 (Factor1) 相對於長期信號線 (Factor2) 的乖離
    # 語意補值: 0.0 (短期共識與長期趨勢完美貼合)
    # 本質上等於 MACD 的柱狀圖 (Histogram)，是極佳的頂底背離信號
    # ---------------------------------------------------------
    bias = factor1 - factor2

    # Stable 版：約束於 [-1.0, 1.0]，專注於常規的共識匯聚與發散
    feat_fusion_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-3.0, 3.0]，捕捉共識瞬間斷層反轉的極端信號
    feat_fusion_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (共識加速度): 共識匯聚或瓦解的速度 (一階導數正規化)
    # 語意補值: 0.0 (市場共識維持等速發展，無突然的加速或減速)
    # 降共線性處理: 減去自身的 EMA 並進行自適應標準化，凸顯突發利好/利空時的共識引爆點
    # ---------------------------------------------------------
    factor_ema_micro = factor1.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (factor1 - factor_ema_micro) / (factor_ema_micro.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_fusion_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極致捕捉市場共識瞬間反轉的動能峰值
    feat_fusion_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (共識分歧度 / 混沌度): 共識因子本身的歷史變異數
    # 語意補值: 0.0 (市場處於極度一致的單邊趨勢，無分歧)
    # 防禦處理: 強制套用 log1p 平滑極端的變異數爆炸
    # 若數值飆高，代表三大底層指標正在互相打架，市場處於極度混沌狀態
    # ---------------------------------------------------------
    fusion_volatility = factor1.rolling_std(window_size=adapt_vol_p)
    log_fusion_vol = fusion_volatility.log1p()

    # Stable 版：約束於 [0.0, 1.0]
    feat_fusion_volatility_stable = (
        log_fusion_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 2.0]，保留極端混沌時期的分歧特徵
    feat_fusion_volatility_sensitive = (
        log_fusion_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 2.0).cast(pl.Float64)
    )

    return {
        "feat_fusion_position_stable": feat_fusion_position_stable,
        "feat_fusion_position_sensitive": feat_fusion_position_sensitive,
        "feat_fusion_bias_stable": feat_fusion_bias_stable,
        "feat_fusion_bias_sensitive": feat_fusion_bias_sensitive,
        "feat_fusion_momentum_stable": feat_fusion_momentum_stable,
        "feat_fusion_momentum_sensitive": feat_fusion_momentum_sensitive,
        "feat_fusion_volatility_stable": feat_fusion_volatility_stable,
        "feat_fusion_volatility_sensitive": feat_fusion_volatility_sensitive,
    }
