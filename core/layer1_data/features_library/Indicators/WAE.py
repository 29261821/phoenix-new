# ==============================================================================
# § 指標 | 瓦達·阿塔爾爆炸指標 (Waddah Attar Explosion, WAE)
# 核心職責: 結合 MACD(方向動能)、布林帶(爆發力)與 ATR(死區濾網)的複合動能指標。
# v3.1 更新: [API 簽名修正] 徹底移除呼叫底層函數時的 kwargs 污染，回歸位置參數。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型  | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|-------|----------|-----------------|------|
| sensitivity   | H & G | float | 100~200  | 150             | MACD 動量變化的乘數敏感度 |
| fast          | H & G | int   | 10 ~ 20  | 20              | MACD 快速 EMA 週期 |
| slow          | H & G | int   | 30 ~ 50  | 40              | MACD 慢速 EMA 週期 |
| channel       | H & G | int   | 10 ~ 30  | 20              | 爆炸線 (布林帶) 週期 |
| mult          | H & G | float | 1.5 ~ 3.0| 2.0             | 爆炸線 (布林帶) 標準差乘數 |
| dz_p          | H & G | int   | 50 ~ 150 | 100             | 死區 (ATR) 週期 |
| dz_m          | H & G | float | 2.0 ~ 5.0| 3.7             | 死區 (ATR) 乘數 |
| adapt_micro_p | G 專用| int   | 3 ~ 10   | 5               | 用於 Momentum (動能加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int   | 21 ~ 55  | 34              | 用於 Volatility (動能混沌度) 的滾動標準差週期 |

【特徵工程說明】
- WAE 的 Trend 與 Explosion 帶有絕對價格尺度。G 接口會除以 close 進行無量綱化。
- Bias 被定義為「爆炸線 (Explosion)」超過「死區 (DeadZone)」的幅度，直接反映市場是否處於爆發狀態。
- Momentum 計算淨動能的加速度，能在 WAE 綠柱/紅柱縮短的瞬間提早給出反轉信號。
"""
from typing import Dict

import polars as pl

from src.features.functions.atr import calculate as atr
from src.features.functions.ema import calculate as ema
from src.features.functions.max import calculate as h_max
from src.features.functions.shift import calculate as prev
from src.features.functions.sma import calculate as sma
from src.features.functions.stddev import calculate as stddev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算 WAE 的四大絕對數值特徵 (TrendUp, TrendDown, Explosion, DeadZone)。
    並額外輸出 NetTrend 供 G 接口無尺度化使用。
    確保依賴「TrendUp > Explosion 且 TrendUp > DeadZone」的傳統量化策略可無縫對接。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含 'sensitivity', 'macd_fast', 'macd_slow',
      'bb_channel', 'bb_mult', 'dz_atr_len', 'dz_mult' 鍵。
    """
    sensitivity = params.get("sensitivity", 150.0)
    macd_fast, macd_slow = params["fast"], params["slow"]
    bb_channel, bb_mult = params["channel"], params["mult"]
    dz_atr_len, dz_mult = params["dz_p"], params["dz_m"]

    c = pl.col("close")

    # 1. 趨勢動能計算 (基於 MACD 差分)
    fast_ma = ema(c, macd_fast)
    slow_ma = ema(c, macd_slow)
    macd_val = fast_ma - slow_ma

    t1 = (macd_val - prev(macd_val, 1)) * sensitivity

    # 2. 爆炸線計算 (基於 Bollinger Bands 頻寬)
    basis = sma(c, bb_channel)
    dev = stddev(c, bb_channel)
    upper_band = basis + bb_mult * dev
    lower_band = basis - bb_mult * dev
    explosion = upper_band - lower_band

    # 3. 死區濾網計算 (基於 ATR)
    dead_zone = atr(df, dz_atr_len) * dz_mult

    return {
        "type": "vector",
        "values": {
            "TrendUp": h_max(t1, pl.lit(0)),
            "TrendDown": h_max(-t1, pl.lit(0)),
            "NetTrend": t1,  # 額外輸出淨趨勢供 G 接口使用
            "Explosion": explosion,
            "DeadZone": dead_zone,
        },
    }


def adapt_WAE(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將充滿絕對價格尺度與多變數的 WAE 轉換為無量綱的 DL/ML 特徵。
    正交分解為：淨動能水位 (Position)、爆炸突破乖離 (Bias)、動能加速度 (Momentum)、動能混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端爆發) 雙版本。
    """
    net_trend = h_output["values"]["NetTrend"].cast(pl.Float64)
    explosion = h_output["values"]["Explosion"].cast(pl.Float64)
    dead_zone = h_output["values"]["DeadZone"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    epsilon = 1e-6

    # 【核心工程：消除 Scale 污染】
    # 將淨動能除以收盤價，得到無量綱的百分比推力
    norm_trend = net_trend / (close + epsilon)

    # ---------------------------------------------------------
    # (A) Position (淨動能水位): 正規化後的多空趨勢推力
    # 語意補值: 0.0 (代表多空動能抵銷，或無動能)
    # ---------------------------------------------------------
    # Stable 版：約束於 [-0.1, 0.1]，專注於常規的動能振盪
    feat_wae_position_stable = (
        norm_trend.fill_nan(0.0).fill_null(0.0).clip(-0.1, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-0.5, 0.5]，捕捉黑天鵝級別的史詩級動能爆發
    feat_wae_position_sensitive = (
        norm_trend.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (爆炸突破乖離): Explosion 相對於 DeadZone 的溢出比例
    # 語意補值: 0.0 (代表爆炸線剛好等於死區線，市場處於爆發臨界點)
    # 正值代表市場正在爆發，負值代表市場處於死水盤整
    # ---------------------------------------------------------
    explosion_ratio = (explosion - dead_zone) / (dead_zone + epsilon)

    # Stable 版：約束於 [-1.0, 2.0] (允許捕捉兩倍於死區的爆發)
    feat_wae_bias_stable = (
        explosion_ratio.fill_nan(0.0).fill_null(0.0).clip(-1.0, 2.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 5.0] (捕捉波動率極度擴張的史詩級趨勢)
    feat_wae_bias_sensitive = (
        explosion_ratio.fill_nan(0.0).fill_null(0.0).clip(-1.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (動能加速度): 淨動能 (norm_trend) 的變化速度
    # 語意補值: 0.0 (動能維持現狀)
    # 降共線性處理: 減去短線 EMA 並自適應標準化，提早捕捉 WAE 柱狀圖變短的瞬間
    # ---------------------------------------------------------
    ema_norm_trend = norm_trend.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (norm_trend - ema_norm_trend) / (ema_norm_trend.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_wae_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極度凸顯動能瞬間抽離或點火的爆發力
    feat_wae_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (動能混沌度): 正規化淨動能的歷史變異數
    # 語意補值: 0.0 (動能維持單向推進，或處於絕對的平靜死水)
    # 防禦處理: 強制套用 log1p 平滑
    # ---------------------------------------------------------
    wae_vol = norm_trend.rolling_std(window_size=adapt_vol_p)
    log_wae_vol = wae_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_wae_volatility_stable = (
        log_wae_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留多空頻繁切換的絞肉機動能特徵
    feat_wae_volatility_sensitive = (
        log_wae_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_wae_position_stable": feat_wae_position_stable,
        "feat_wae_position_sensitive": feat_wae_position_sensitive,
        "feat_wae_bias_stable": feat_wae_bias_stable,
        "feat_wae_bias_sensitive": feat_wae_bias_sensitive,
        "feat_wae_momentum_stable": feat_wae_momentum_stable,
        "feat_wae_momentum_sensitive": feat_wae_momentum_sensitive,
        "feat_wae_volatility_stable": feat_wae_volatility_stable,
        "feat_wae_volatility_sensitive": feat_wae_volatility_sensitive,
    }
