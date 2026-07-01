# ==============================================================================
# § 指標 | SMC 訂單塊 v2.0 (邏輯修正版)
# 核心職責: 識別市場大資金留下的訂單塊 (Order Block)，捕捉強勢突破前的最後洗盤區間。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| periods       | H & G | int  | 3 ~ 21   | 無 (必填)       | 強勢突破 (Displacement) 的觀察期 |
| threshold     | H     | float| -        | 0.0             | (保留參數) 突破的幅度門檻 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | 55              | 用於 Position (訂單塊政權) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 5 ~ 21   | 13              | 用於 Bias (短線記憶) 的 EMA 衰減週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | 34              | 用於 Volatility (事件群集度) 的滾動標準差週期 |

【特徵工程說明】
- SMC 訂單塊在 H 接口中是極度稀疏的事件脈衝 (0, 1, -1)。
- G 接口將其擴展為連續的時間面特徵，使得 DL/ML 模型能理解「訂單塊的餘波效應」。
- 透過 adapt_macro_p 觀察近期市場是由看漲 OB 還是看跌 OB 統治 (Market Regime)。
"""
from typing import Dict

import polars as pl

from src.features.functions.shift import calculate as prev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留精確的布林觸發信號。
    v2.0 核心升級:
    - [標準定義對齊] 嚴格遵循 SMC 定義：
        - 看漲 OB (Bullish OB): 在一輪強勢上漲前，最後一根下跌的 K 棒。
        - 看跌 OB (Bearish OB): 在一輪強勢下跌前，最後一根上漲的 K 棒。
    - [引入突破確認] 增加了對 Order Block 自身的突破作為確認條件，使信號更加明確。

    契約：
    - df 必須包含 'open', 'high', 'low', 'close' 欄位。
    - params 必須包含 'periods' 鍵。
    """
    # 參數 periods 現在定義了強勢突破的觀察期
    periods = params["periods"]
    # threshold 參數暫時保留，以備未來擴展 (例如要求突破幅度)
    # threshold = params.get('threshold', 0.0)

    o, h, l, c = pl.col("open"), pl.col("high"), pl.col("low"), pl.col("close")

    # --- 1. 識別潛在的訂單塊 K 棒 ---
    # 潛在看漲 OB: 一根下跌的 K 棒
    potential_bullish_ob_candle = prev(c, 1) < prev(o, 1)
    # 潛在看跌 OB: 一根上漲的 K 棒
    potential_bearish_ob_candle = prev(c, 1) > prev(o, 1)

    # --- 2. 識別強勢位移 (Displacement) ---
    # 強勢上漲：當前 K 棒的收盤價，高於過去 `periods` 根 K 棒的最高價
    strong_up_move = c > pl.col("high").shift(1).rolling_max(window_size=periods)
    # 強勢下跌：當前 K 棒的收盤價，低於過去 `periods` 根 K 棒的最低價
    strong_down_move = c < pl.col("low").shift(1).rolling_min(window_size=periods)

    # --- 3. 組合條件以確認訂單塊 ---
    # 看漲 OB 被確認: 當前是強勢上漲，且前一根 K 棒是下跌 K 棒
    is_bull = strong_up_move & potential_bullish_ob_candle
    # 看跌 OB 被確認: 當前是強勢下跌，且前一根 K 棒是上漲 K 棒
    is_bear = strong_down_move & potential_bearish_ob_candle

    return {"type": "event", "values": {"isBullish": is_bull, "isBearish": is_bear}}


def adapt_Order_Block(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將極度稀疏的 OB 脈衝事件，轉換為 DL/ML 可學習的連續時空特徵。
    正交分解為：訂單塊政權水位 (Position)、短線訂單塊記憶 (Bias)、瞬時脈衝 (Momentum)、事件群集度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，政權衰減與群集觀察週期全面可由 YAML 配置。
    """
    is_bull = h_output["values"]["isBullish"]
    is_bear = h_output["values"]["isBearish"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    # 由於基礎參數是突破觀察期，特徵工程需要較長週期來做事件衰減，故提供預設常數
    adapt_macro_p = params.get("adapt_macro_p", 55)
    adapt_micro_p = params.get("adapt_micro_p", 13)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    # ---------------------------------------------------------
    # (A) Momentum (瞬時脈衝): 當下 K 棒是否確認了 Order Block
    # 語意補值: 0.0 (無事件)
    # 看漲 OB 設為 1.0；看跌 OB 設為 -1.0
    # ---------------------------------------------------------
    impulse = (pl.when(is_bull).then(1.0).when(is_bear).then(-1.0).otherwise(0.0)).cast(
        pl.Float64
    )

    # Stable 版與 Sensitive 版先天已約束於 [-1.0, 1.0]
    feat_ob_momentum_stable = impulse
    feat_ob_momentum_sensitive = impulse

    # ---------------------------------------------------------
    # (B) Bias (短線記憶 / Event Decay): 事件的短期衰減餘波
    # 語意補值: 0.0 (近期無任何訂單塊形成)
    # ---------------------------------------------------------
    short_memory = impulse.ewm_mean(span=adapt_micro_p, ignore_nulls=True)

    # Stable 版：EMA 壓縮數值，約束於 [-0.5, 0.5]
    feat_ob_bias_stable = (
        short_memory.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]
    feat_ob_bias_sensitive = (
        short_memory.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Position (訂單塊政權水位): 長期訂單塊的統治方向
    # 語意補值: 0.0 (長期來看多空 OB 勢均力敵，或皆無 OB)
    # ---------------------------------------------------------
    long_memory = impulse.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：約束於 [-0.2, 0.2]
    feat_ob_position_stable = (
        long_memory.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.5, 0.5]
    feat_ob_position_sensitive = (
        long_memory.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (事件群集度): OB 形成事件的歷史變異數
    # 語意補值: 0.0 (市場無結構性破壞與重建，趨勢極度順暢)
    # 防禦處理: 強制套用 log1p 平滑
    # 若數值飆高，代表市場在頻繁洗盤，不斷形成新的多空 OB
    # ---------------------------------------------------------
    ob_vol = impulse.rolling_std(window_size=adapt_vol_p)
    log_ob_vol = ob_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_ob_volatility_stable = (
        log_ob_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]
    feat_ob_volatility_sensitive = (
        log_ob_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_ob_momentum_stable": feat_ob_momentum_stable,
        "feat_ob_momentum_sensitive": feat_ob_momentum_sensitive,
        "feat_ob_bias_stable": feat_ob_bias_stable,
        "feat_ob_bias_sensitive": feat_ob_bias_sensitive,
        "feat_ob_position_stable": feat_ob_position_stable,
        "feat_ob_position_sensitive": feat_ob_position_sensitive,
        "feat_ob_volatility_stable": feat_ob_volatility_stable,
        "feat_ob_volatility_sensitive": feat_ob_volatility_sensitive,
    }
