# ==============================================================================
# § 指標 | 歸一化 MACD (Normalized MACD)
# 核心職責: 計算一個在 [-1, 1] 區間歸一化的 MACD 指標，天生具備抗尺度污染特性。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口極致正交特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| source        | H & G | str  | -        | 無 (必填)       | 價格來源 (如 'close') |
| fast_period   | H & G | int  | 8 ~ 15   | 無 (必填)       | 快線 EMA 週期 |
| slow_period   | H & G | int  | 21 ~ 34  | 無 (必填)       | 慢線 EMA 週期 |
| signal_period | H & G | int  | 5 ~ 13   | 無 (必填)       | 訊號線 WMA 週期 |
| norm_period   | H & G | int  | 30 ~ 100 | 無 (必填)       | 用於尋找歷史極值進行歸一化的視窗週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | slow_period 參數| (未直接使用，保留做系統對齊) |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | signal_period 參數| 用於 Momentum (柱狀圖加速度) 的短線平滑週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | norm_period 參數| 用於 Volatility (動能混沌度) 的滾動標準差週期 |

【特徵工程說明】
- N-MACD 天生已經歸一化至 [-1, 1] 區間，因此不需要依賴 close 進行無量綱化。
- 透過 adapt_micro_p 計算歸一化柱狀圖的加速度，極致敏銳地捕捉動能衰竭。
- 透過 adapt_vol_p 計算指標線波動率，識別大行情啟動前的動能極度收斂。
"""
from typing import Dict

import polars as pl

from src.features.functions.ema import calculate as ema
from src.features.functions.highest import calculate as highest
from src.features.functions.lowest import calculate as lowest
from src.features.functions.max import calculate as h_max
from src.features.functions.min import calculate as h_min
from src.features.functions.wma import calculate as wma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留歸一化後的 N-MACD 指標值 (Line, Signal, Hist)，數值嚴格落於 [-1, 1]。
    確保傳統量化策略能基於絕對門檻 (如 Line > 0.8 代表極端超買) 無縫對接。

    契約：
    - df 必須包含 params['source'] 指定的欄位。
    - params 必須包含所有基礎計算鍵。
    """
    source_col, fast_p, slow_p, signal_p, norm_p = (
        params["source"],
        params["fast_period"],
        params["slow_period"],
        params["signal_period"],
        params["norm_period"],
    )
    source = pl.col(source_col)
    epsilon = 1e-9

    fast_e = ema(source, fast_p)
    slow_e = ema(source, slow_p)
    ratio = h_min(fast_e, slow_e) / (h_max(fast_e, slow_e) + epsilon)
    mac_raw_base = pl.when(fast_e > slow_e).then(2 - ratio).otherwise(ratio)
    mac_raw = mac_raw_base - 1
    low_mac = lowest(mac_raw, norm_p)
    high_mac = highest(mac_raw, norm_p)
    line = ((mac_raw - low_mac) / (high_mac - low_mac + epsilon)) * 2 - 1
    sig = wma(line, signal_p)
    hist = line - sig

    return {"type": "vector", "values": {"Line": line, "Signal": sig, "Hist": hist}}


def adapt_N_MACD(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將天生無尺度的 N-MACD 進行二次高階時序特徵萃取。
    正交分解為：動能歷史水位 (Position)、柱狀圖發散乖離 (Bias)、動能翻轉加速度 (Momentum)、動能混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端反轉) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，動能與變異數週期全面可由 YAML 配置。
    """
    macd_line = h_output["values"]["Line"]
    macd_hist = h_output["values"]["Hist"]

    # 1. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_micro_p = params.get("adapt_micro_p", params["signal_period"])
    adapt_vol_p = params.get("adapt_vol_p", params["norm_period"])

    epsilon = 1e-6

    # N-MACD 的特點是它已經是 [-1, 1] 的無量綱特徵，無需除以 close。
    # 直接使用 macd_line 作為基礎指標進行特徵工程。

    # ---------------------------------------------------------
    # (A) Position (動能歷史水位): N-MACD 線的絕對位置
    # 語意補值: 0.0 (長短均線重合，無明顯動能方向)
    # ---------------------------------------------------------
    # Stable 版與 Sensitive 版先天已約束於 [-1.0, 1.0] (偶爾微小溢出)
    feat_nmacd_position_stable = (
        macd_line.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    feat_nmacd_position_sensitive = (
        macd_line.fill_nan(0.0).fill_null(0.0).clip(-1.5, 1.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (柱狀圖發散乖離): N-MACD 線相對於其信號線的乖離 (即 Hist)
    # 語意補值: 0.0 (MACD 線與信號線完美貼合，即將交叉)
    # ---------------------------------------------------------
    # Stable 版：約束於 [-0.5, 0.5]
    feat_nmacd_bias_stable = (
        macd_hist.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]，捕捉暴拉/暴跌產生的極端背離空間
    feat_nmacd_bias_sensitive = (
        macd_hist.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (動能翻轉加速度): 柱狀圖 (Hist) 的加速度
    # 語意補值: 0.0 (動能維持等速發展)
    # 降共線性處理: 減去自身的 EMA 並自適應標準化，敏銳捕捉紅綠柱縮短的反轉瞬間
    # ---------------------------------------------------------
    ema_hist = macd_hist.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (macd_hist - ema_hist) / (ema_hist.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_nmacd_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯變盤瞬間的強大加速度
    feat_nmacd_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (動能混沌度): N-MACD 線本身的歷史變異數
    # 語意補值: 0.0 (動能極度平穩或如死水般收斂)
    # 防禦處理: 強制套用 log1p 平滑
    # ---------------------------------------------------------
    nmacd_volatility = macd_line.rolling_std(window_size=adapt_vol_p)
    log_nmacd_vol = nmacd_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_nmacd_volatility_stable = (
        log_nmacd_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留市場瘋狂狀態下的極端特徵
    feat_nmacd_volatility_sensitive = (
        log_nmacd_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_nmacd_position_stable": feat_nmacd_position_stable,
        "feat_nmacd_position_sensitive": feat_nmacd_position_sensitive,
        "feat_nmacd_bias_stable": feat_nmacd_bias_stable,
        "feat_nmacd_bias_sensitive": feat_nmacd_bias_sensitive,
        "feat_nmacd_momentum_stable": feat_nmacd_momentum_stable,
        "feat_nmacd_momentum_sensitive": feat_nmacd_momentum_sensitive,
        "feat_nmacd_volatility_stable": feat_nmacd_volatility_stable,
        "feat_nmacd_volatility_sensitive": feat_nmacd_volatility_sensitive,
    }
