# ==============================================================================
# § 指標 | 驅動力上下文 (Force Context)
# 核心職責: 計算量價共振的強力指標 (Force Index)，並將其轉換為離散狀態或連續推力特徵。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| source        | H & G | str  | -        | 無 (必填)       | 價格來源 (如 'close') |
| period        | H & G | int  | 10 ~ 50  | 無 (必填)       | 計算 Raw Force 的基礎 SMA 週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 89  | 34              | 用於 Base Value 與 Bias (宏觀乖離) 的長線 EMA 週期 |
| adapt_micro_p | G 專用| int  | 5 ~ 14   | 9               | 用於 Momentum (推力加速度) 計算的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | 34              | 用於 Volatility (驅動力混沌度) 的滾動標準差週期 |

【特徵工程說明】
- Force Index 天生具有嚴重的成交量 Scale 污染。G 接口會使用 adapt_macro_p 作為基準週期來進行無量綱化。
- 透過 adapt_macro_p 觀察淨推力相對於長線均值的背離，捕捉量價背離。
- 透過 adapt_vol_p 衡量驅動力的混沌度，識別高換手率的絞肉機行情。
"""
from typing import Dict

import polars as pl

from src.features.functions.ema import calculate as ema
from src.features.functions.shift import calculate as prev
from src.features.functions.sma import calculate as sma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算基於價格變動與成交量的 Force Index，並將其離散化為狀態標籤。
    保留 1 (多方控盤), -1 (空方控盤), 0 (動能為零) 的絕對狀態碼。
    確保舊有基於此狀態作為趨勢方向濾網的策略腳本可無縫運行。

    同時，為了 G 接口能徹底消除成交量 Scale 污染，我們將原始的 Raw Force
    也一併輸出。

    契約：
    - df 必須包含 params['source'] 和 'volume' 欄位。
    - params 必須包含 'source', 'period' 鍵。
    """
    source_col = params["source"]
    period = params["period"]
    source = pl.col(source_col)

    # 1. 計算原始的 Force Index (絕對單位，包含嚴重的尺度污染)
    raw_force = sma(
        series=((source - prev(series=source, period=1)) * pl.col("volume")),
        length=period,
    )

    # 2. 轉換為離散的狀態碼供 H 接口使用
    force_state = (
        pl.when(raw_force > 0)
        .then(pl.lit(1, dtype=pl.Int8))
        .when(raw_force < 0)
        .then(pl.lit(-1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
    )

    return {
        "type": "event",
        "values": {
            "State": force_state,
            "RawForce": raw_force,  # 供 G 接口解碼使用
        },
    }


def adapt_Force_Context(
    h_output: Dict, close: pl.Expr, volume: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    放棄粗糙的二值化狀態碼，直接對 Raw Force 進行無量綱化與高階特徵萃取。
    我們透過計算歷史基準成交額 (Base Value) 來消除 Scale 污染。
    正交分解為：絕對淨推力 (Position)、推力宏觀乖離 (Bias)、推力加速度 (Momentum)、驅動混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉黑天鵝) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，基準消除與衰減週期全面可由 YAML 配置。
    """
    raw_force = h_output["values"]["RawForce"]

    # 1. 提取基礎參數
    period = params["period"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    # 由於量價基準的消除需要較穩定的長週期，給予預設常數 34 與 9
    adapt_macro_p = params.get("adapt_macro_p", 34)
    adapt_micro_p = params.get("adapt_micro_p", 9)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # --- 核心工程：消除 Scale 污染 ---
    # 計算基準價格與基準成交量 (使用 adapt_macro_p 以獲得穩定基準)
    base_price = ema(series=close, length=adapt_macro_p)
    base_volume = ema(series=volume, length=adapt_macro_p)
    base_value = base_price * base_volume

    # 將絕對 Force 轉換為「相對淨推力百分比」
    norm_force = raw_force / (base_value + epsilon)

    # ---------------------------------------------------------
    # (A) Position (絕對驅動力 / 淨推力): 當前量價推動佔基準成交額的比例
    # 語意補值: 0.0 (多空力道平衡，無實質推動)
    # ---------------------------------------------------------
    # Stable 版：約束於 [-1.0, 1.0]，最多關注相當於 1 倍均量的異常推力，穩定 DL 權重
    feat_force_context_position_stable = (
        norm_force.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-3.0, 3.0]，捕捉瞬間爆發 3 倍均量砸盤的史詩級異動
    feat_force_context_position_sensitive = (
        norm_force.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (驅動力宏觀乖離): 淨推力相對於其長線均線的背離
    # 語意補值: 0.0 (當前推力與過去基準力道一致)
    # ---------------------------------------------------------
    force_ema_macro = norm_force.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = norm_force - force_ema_macro

    # Stable 版：約束於 [-0.5, 0.5]，專注於常規的量價背離或推力衰竭
    feat_force_context_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.5, 1.5]，捕捉推力瞬間斷層反轉的強烈信號
    feat_force_context_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.5, 1.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (驅動力加速度 / 衝動): 淨推力的變化速度
    # 語意補值: 0.0 (推力維持等速，無突發的加速爆發)
    # 降共線性處理: 減去自身的短線 EMA 並自適應標準化，凸顯瞬間的籌碼瘋搶或踩踏
    # ---------------------------------------------------------
    ema_norm_force = norm_force.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (norm_force - ema_norm_force) / (ema_norm_force.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_force_context_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極致捕捉主力瞬間突襲的動能峰值
    feat_force_context_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (驅動力混沌度): 淨推力的歷史變異數
    # 語意補值: 0.0 (推力極度平穩，呈現無量陰跌或暗盤吸籌)
    # 防禦處理: 強制套用 log1p 平滑多空激烈互砸時產生的極端變異數
    # ---------------------------------------------------------
    force_volatility = norm_force.rolling_std(window_size=adapt_vol_p)
    log_force_vol = force_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_force_context_volatility_stable = (
        log_force_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.5]，保留高換手絞肉機行情下的極度混亂狀態
    feat_force_context_volatility_sensitive = (
        log_force_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.5).cast(pl.Float64)
    )

    return {
        "feat_force_context_position_stable": feat_force_context_position_stable,
        "feat_force_context_position_sensitive": feat_force_context_position_sensitive,
        "feat_force_context_bias_stable": feat_force_context_bias_stable,
        "feat_force_context_bias_sensitive": feat_force_context_bias_sensitive,
        "feat_force_context_momentum_stable": feat_force_context_momentum_stable,
        "feat_force_context_momentum_sensitive": feat_force_context_momentum_sensitive,
        "feat_force_context_volatility_stable": feat_force_context_volatility_stable,
        "feat_force_context_volatility_sensitive": feat_force_context_volatility_sensitive,
    }
