# ==============================================================================
# § 指標 | 背離探測引擎 v2.2 (邏輯暨錯誤修正版)
# 核心職責: 透過比對價格擺動點與指標值，探測趨勢反轉(常規背離)與趨勢延續(隱藏背離)。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| source        | H & G | str  | -        | 無 (必填)       | 計算 RSI 的價格來源 (如 'close') |
| rsi_period    | H & G | int  | 7 ~ 21   | 無 (必填)       | 內化 RSI 的計算週期 |
| p_left        | H & G | int  | 3 ~ 10   | 無 (必填)       | 擺動點 (Pivot) 左側所需的 K 棒數 |
| p_right       | H & G | int  | 3 ~ 10   | 無 (必填)       | 擺動點 (Pivot) 右側所需的 K 棒數 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | 55              | 用於長線背離中樞 (Position) 的 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 5 ~ 21   | 13              | 用於短線背離記憶 (Bias) 的 EMA 衰減週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | 34              | 用於結構混沌度 (Volatility) 的滾動標準差週期 |

【特徵工程說明】
- 背離事件為極度稀疏的離散脈衝 (-2, -1, 0, 1, 2)。
- 透過 adapt_micro_p 創造「事件衰減餘波 (Event Decay)」，將瞬間脈衝擴展為時間面。
- 透過 adapt_macro_p 形成長期的宏觀政權 (Market Regime)，衡量長線頂底結構的背離失衡。
"""
from typing import Dict

import polars as pl

from src.features.functions.pivots import calculate as pivots
from src.features.functions.wma import calculate as wma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的離散事件代碼 (1: 常規看漲, 2: 隱藏看漲, -1: 常規看跌, -2: 隱藏看跌, 0: 無)。
    確保依賴精準背離觸發點的傳統量化策略可無縫進場。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含 'source', 'rsi_period', 'p_left', 'p_right' 鍵。
    """
    source_col = params["source"]
    rsi_period = params["rsi_period"]
    p_left = params["p_left"]
    p_right = params["p_right"]

    source = pl.col(source_col)
    epsilon = 1e-9

    # --- 1. 內化的 RSI 計算邏輯 ---
    delta = source - source.shift(1)
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)

    avg_gain = wma(series=gain, length=rsi_period)
    avg_loss = wma(series=loss, length=rsi_period)

    # [v2.2 錯誤修正] 補上遺漏的 rs 變數定義
    rs = avg_gain / (avg_loss + epsilon)
    indicator = 100 - (100 / (1 + rs))

    # --- 2. 背離判斷邏輯 (v2.2 專業版) ---
    # 步驟 1: 只尋找價格的擺動高低點 (Price Pivots)
    price_pivots = pivots(series=pl.col("high"), left=p_left, right=p_right)
    is_ph = price_pivots == 1
    is_pl = price_pivots == -1

    # 步驟 2: 當價格擺動點形成時，記錄下當時的「價格」和「指標值」
    ph_price = pl.when(is_ph).then(pl.col("high")).otherwise(None)
    pl_price = pl.when(is_pl).then(pl.col("low")).otherwise(None)
    ph_indicator = pl.when(is_ph).then(indicator).otherwise(None)
    pl_indicator = pl.when(is_pl).then(indicator).otherwise(None)

    # 步驟 3: 取得「前一個」擺動點的價格和指標值
    prev_ph_price = ph_price.forward_fill().shift(1)
    prev_pl_price = pl_price.forward_fill().shift(1)
    prev_ph_indicator = ph_indicator.forward_fill().shift(1)
    prev_pl_indicator = pl_indicator.forward_fill().shift(1)

    # 步驟 4: 只在「新的」擺動點K棒上，進行背離條件判斷
    is_regular_bear = (
        is_ph & (ph_price > prev_ph_price) & (ph_indicator < prev_ph_indicator)
    )
    is_hidden_bear = (
        is_ph & (ph_price < prev_ph_price) & (ph_indicator > prev_ph_indicator)
    )

    is_regular_bull = (
        is_pl & (pl_price < prev_pl_price) & (pl_indicator > prev_pl_indicator)
    )
    is_hidden_bull = (
        is_pl & (pl_price > prev_pl_price) & (pl_indicator < prev_pl_indicator)
    )

    div_type = (
        pl.when(is_hidden_bull)
        .then(2)
        .when(is_regular_bull)
        .then(1)
        .when(is_regular_bear)
        .then(-1)
        .when(is_hidden_bear)
        .then(-2)
        .otherwise(0)
    ).cast(pl.Int8)

    return {"type": "event", "values": {"Event": div_type}}


def adapt_Divergence(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將極度稀疏的背離事件 (-2, -1, 0, 1, 2) 轉換為連續的時空特徵空間。
    正交分解為：瞬時脈衝 (Momentum)、短線背離記憶 (Bias)、長線背離中樞 (Position)、結構混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉群集背離) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，所有事件衰減週期全面可由 YAML 配置。
    """
    event = h_output["values"]["Event"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    # 由於基礎參數是極短週期的 pivots，特徵工程需要較長週期來做事件衰減，故提供預設常數
    adapt_macro_p = params.get("adapt_macro_p", 55)
    adapt_micro_p = params.get("adapt_micro_p", 13)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    # 防禦性常數
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Momentum (瞬時背離脈衝): 發生背離當下的標準化脈衝信號
    # 語意補值: 0.0 (無背離)
    # 將 -2 到 2 的事件碼除以 2.0，標準化至 [-1.0, 1.0] 空間
    # ---------------------------------------------------------
    impulse = event / 2.0

    # Stable 版：嚴格約束於 [-1.0, 1.0]
    feat_divergence_momentum_stable = (
        impulse.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束，允許樹模型識別極端異常
    feat_divergence_momentum_sensitive = (
        impulse.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (短線背離記憶 / Event Decay): 背離事件的短期衰減餘波
    # 語意補值: 0.0 (近期無任何背離信號，結構乾淨)
    # ---------------------------------------------------------
    short_memory = event.ewm_mean(span=adapt_micro_p, ignore_nulls=True)

    # Stable 版：EMA 會縮小峰值，約束於 [-0.5, 0.5]，穩定 Transformer 權重
    feat_divergence_bias_stable = (
        short_memory.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]，保留連續多次背離疊加產生的餘波峰值
    feat_divergence_bias_sensitive = (
        short_memory.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Position (長線背離中樞): 衡量宏觀的頂底結構政權 (Market Regime)
    # 語意補值: 0.0 (長線來看多空背離次數抵銷，或皆無背離)
    # ---------------------------------------------------------
    long_memory = event.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：長期 EMA 數值更小，約束於 [-0.2, 0.2]
    feat_divergence_position_stable = (
        long_memory.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.5, 0.5]，捕捉宏觀級別的結構失衡與頂底特徵
    feat_divergence_position_sensitive = (
        long_memory.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (結構混沌度): 背離信號群集的混亂程度
    # 語意補值: 0.0 (無頻繁背離，趨勢極度順暢)
    # 防禦處理: 強制套用 log1p 平滑滾動變異數，防止極端波動
    # ---------------------------------------------------------
    # 透過滾動標準差，識別當前是否處於「連環背離卻不反轉」的指標鈍化瘋狂期
    divergence_vol = event.rolling_std(window_size=adapt_vol_p)
    log_divergence_vol = divergence_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_divergence_volatility_stable = (
        log_divergence_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]，保留極端洗盤時的群集特徵 (Signal Clustering)
    feat_divergence_volatility_sensitive = (
        log_divergence_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_divergence_momentum_stable": feat_divergence_momentum_stable,
        "feat_divergence_momentum_sensitive": feat_divergence_momentum_sensitive,
        "feat_divergence_bias_stable": feat_divergence_bias_stable,
        "feat_divergence_bias_sensitive": feat_divergence_bias_sensitive,
        "feat_divergence_position_stable": feat_divergence_position_stable,
        "feat_divergence_position_sensitive": feat_divergence_position_sensitive,
        "feat_divergence_volatility_stable": feat_divergence_volatility_stable,
        "feat_divergence_volatility_sensitive": feat_divergence_volatility_sensitive,
    }
