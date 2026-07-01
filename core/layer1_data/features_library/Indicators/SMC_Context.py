# ==============================================================================
# § 指標 | SMC 上下文引擎 v3.0 (專業邏輯整合版)
# 核心職責: 將市場結構、失衡區、流動性與訂單塊打包成統一的 SMC 上下文狀態機。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口連續化降維特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型  | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|-------|----------|-----------------|------|
| p_left        | H     | int   | 3 ~ 10   | 無 (必填)       | 市場結構 Pivot 左側 K 棒數 |
| p_right       | H     | int   | 3 ~ 10   | 無 (必填)       | 市場結構 Pivot 右側 K 棒數 |
| bos_left      | H     | int   | -        | 5               | (相容性保留) BOS 觀察參數 |
| bos_right     | H     | int   | -        | 5               | (相容性保留) BOS 觀察參數 |
| liq_left      | H     | int   | 3 ~ 10   | 無 (必填)       | 流動性 Pivot 左側 K 棒數 |
| liq_right     | H     | int   | 3 ~ 10   | 無 (必填)       | 流動性 Pivot 右側 K 棒數 |
| liq_thresh_pct| H     | float | 0.0~0.005| 無 (必填)       | EQH/EQL 容許誤差百分比 |
| ob_p          | H     | int   | 3 ~ 10   | 無 (必填)       | 訂單塊位移觀察期 |
| ob_thresh     | H     | float | 1.5 ~ 3.0| 無 (必填)       | 訂單塊突破 ATR 乘數 |
| adapt_macro_p | G 專用| int   | 34 ~ 89  | 55              | 用於 Position (SMC 政權) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int   | 5 ~ 21   | 13              | 用於 Momentum (政權切換) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int   | 21 ~ 55  | 34              | 用於 Volatility (結構混沌度) 的滾動標準差週期 |

【特徵工程說明】
- SMC 原始狀態極度複雜且稀疏。G 接口將其融合為連續的「SMC 綜合政權評分」[-1.0, 1.0]。
- 透過加權組合 (結構40% + OB 40% + FVG 20%)，幫助 DL 模型一維度理解市場聰明錢方向。
"""
from typing import Dict

import numpy as np
import polars as pl

from src.features.functions.abs import calculate as abs_val
from src.features.functions.atr import calculate as atr
from src.features.functions.pivots import calculate as pivots
from src.features.functions.shift import calculate as prev
from src.features.functions.sum import calculate as rolling_sum


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    將四大 SMC 核心元素 (Structure, Imbalance, Liquidity, Order Block) 打包為 Struct。
    [契約修復]: 為了完美適配惰性計算，Eager 產生的 numpy array 被封裝為 pl.lit(pl.Series)。
    同時展開平坦化的特徵，供 G 接口直接取用。
    """
    # --- 參數提取 ---
    p_left, p_right = params["p_left"], params["p_right"]
    liq_left, liq_right = params["liq_left"], params["liq_right"]
    liq_thresh_pct = params["liq_thresh_pct"]
    ob_p = params["ob_p"]
    ob_atr_len = 14
    ob_atr_mult = params.get("ob_thresh", 1.5)

    h, l, o, c = pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close")
    epsilon = 1e-9

    # --- 1. 市場結構 (Structure) v2.0 ---
    pivots_val = pivots(series=h, left=p_left, right=p_right)
    is_ph = pivots_val == 1
    is_pl = pivots_val == -1

    hh = pl.when(is_ph).then(h).otherwise(None).forward_fill()
    ll = pl.when(is_pl).then(l).otherwise(None).forward_fill()
    is_uptrend = (
        (hh > hh.cum_max().shift(1)) & (ll > ll.cum_max().shift(1))
    ).fill_null(False)
    is_downtrend = (
        (hh < hh.cum_min().shift(1)) & (ll < ll.cum_min().shift(1))
    ).fill_null(False)

    last_ph = pl.when(is_ph).then(h).otherwise(None).forward_fill()
    last_pl = pl.when(is_pl).then(l).otherwise(None).forward_fill()
    prev_last_ph = prev(series=last_ph, period=1)
    prev_last_pl = prev(series=last_pl, period=1)

    break_high = h > prev_last_ph
    break_low = l < prev_last_pl

    bos_bull = is_uptrend & break_high
    choch_bear = is_uptrend & break_low
    bos_bear = is_downtrend & break_low
    choch_bull = is_downtrend & break_high

    structure_val = pl.struct(
        [
            is_uptrend.alias("isUptrend"),
            is_downtrend.alias("isDowntrend"),
            bos_bull.alias("bosBull"),
            choch_bear.alias("chochBear"),
            bos_bear.alias("bosBear"),
            choch_bull.alias("chochBull"),
        ]
    )

    # --- 2. 失衡區 (Imbalance / FVG) ---
    fvg_bull = l > prev(series=h, period=2)
    fvg_bear = h < prev(series=l, period=2)
    imbalance_val = pl.struct(
        [fvg_bull.alias("isBullish"), fvg_bear.alias("isBearish")]
    )

    # --- 3. 流動性 (Liquidity) v2.0 ---
    liq_pivots = pivots(series=h, left=liq_left, right=liq_right)
    is_liq_ph = liq_pivots == 1
    is_liq_pl = liq_pivots == -1

    liq_ph_price = pl.when(is_liq_ph).then(h).otherwise(None)
    liq_pl_price = pl.when(is_liq_pl).then(l).otherwise(None)

    prev_liq_ph_price = liq_ph_price.forward_fill().shift(1)
    prev_liq_pl_price = liq_pl_price.forward_fill().shift(1)

    high_diff = abs_val(series=(liq_ph_price - prev_liq_ph_price)) / (
        prev_liq_ph_price + epsilon
    )
    low_diff = abs_val(series=(liq_pl_price - prev_liq_pl_price)) / (
        prev_liq_pl_price + epsilon
    )

    is_eqh = is_liq_ph & (high_diff < liq_thresh_pct)
    is_eql = is_liq_pl & (low_diff < liq_thresh_pct)
    liquidity_val = pl.struct([is_eqh.alias("isEQH"), is_eql.alias("isEQL")])

    # --- 4. 訂單塊 (Order Block) v2.0 (Eager 部分) ---
    price_change = (c - prev(series=c, period=ob_p)).abs()
    atr_val = atr(df=df, period=ob_atr_len)
    is_displacement = price_change > (atr_val * ob_atr_mult)

    is_new_bull_ob_expr = prev(series=(c < o), period=1) & is_displacement & (c > o)
    is_new_bear_ob_expr = prev(series=(c > o), period=1) & is_displacement & (c < o)

    df_with_triggers = df.with_columns(
        is_new_bull_ob=is_new_bull_ob_expr.fill_null(False),
        is_new_bear_ob=is_new_bear_ob_expr.fill_null(False),
        potential_bull_ob_top=prev(series=o, period=1),
        potential_bull_ob_bottom=prev(series=l, period=1),
        potential_bear_ob_top=prev(series=h, period=1),
        potential_bear_ob_bottom=prev(series=o, period=1),
    )

    high_np = df_with_triggers["high"].to_numpy()
    low_np = df_with_triggers["low"].to_numpy()
    is_new_bull_ob_np = df_with_triggers["is_new_bull_ob"].to_numpy()
    is_new_bear_ob_np = df_with_triggers["is_new_bear_ob"].to_numpy()
    bull_ob_top_np = df_with_triggers["potential_bull_ob_top"].to_numpy()
    bull_ob_bottom_np = df_with_triggers["potential_bull_ob_bottom"].to_numpy()
    bear_ob_top_np = df_with_triggers["potential_bear_ob_top"].to_numpy()
    bear_ob_bottom_np = df_with_triggers["potential_bear_ob_bottom"].to_numpy()

    n = len(df)
    is_bullish_ob_np = np.zeros(n, dtype=bool)
    is_bearish_ob_np = np.zeros(n, dtype=bool)

    active_bull_ob = {"top": np.nan, "bottom": np.nan, "valid": False}
    active_bear_ob = {"top": np.nan, "bottom": np.nan, "valid": False}

    for i in range(n):
        if active_bull_ob["valid"]:
            if low_np[i] < active_bull_ob["bottom"]:
                active_bull_ob["valid"] = False
            else:
                is_bullish_ob_np[i] = True

        if active_bear_ob["valid"]:
            if high_np[i] > active_bear_ob["top"]:
                active_bear_ob["valid"] = False
            else:
                is_bearish_ob_np[i] = True

        if is_new_bull_ob_np[i]:
            active_bull_ob = {
                "top": bull_ob_top_np[i],
                "bottom": bull_ob_bottom_np[i],
                "valid": True,
            }
            is_bullish_ob_np[i] = True

        if is_new_bear_ob_np[i]:
            active_bear_ob = {
                "top": bear_ob_top_np[i],
                "bottom": bear_ob_bottom_np[i],
                "valid": True,
            }
            is_bearish_ob_np[i] = True

    # 將 numpy 結果轉為表達式
    bull_ob_expr = pl.lit(pl.Series("isBullish", is_bullish_ob_np))
    bear_ob_expr = pl.lit(pl.Series("isBearish", is_bearish_ob_np))

    order_block_val = pl.struct(
        [
            bull_ob_expr.alias("isBullish"),
            bear_ob_expr.alias("isBearish"),
        ]
    )

    context = pl.struct(
        [
            pivots_val.alias("pivots"),
            structure_val.alias("structure"),
            imbalance_val.alias("imbalance"),
            liquidity_val.alias("liquidity"),
            order_block_val.alias("order_block"),
        ]
    )

    return {
        "type": "vector",
        "values": {
            "Context": context,
            # 展開平坦化特徵供 G 接口直接調用
            "isUptrend": is_uptrend,
            "isDowntrend": is_downtrend,
            "fvgBull": fvg_bull,
            "fvgBear": fvg_bear,
            "obBull": bull_ob_expr,
            "obBear": bear_ob_expr,
        },
    }


def adapt_SMC_Context(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將龐大且稀疏的 SMC 狀態機，降維合成為連續的「SMC 綜合政權評分」[-1.0, 1.0]。
    正交分解為：SMC 政權中樞 (Position)、政權乖離 (Bias)、政權切換加速度 (Momentum)、結構混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    is_uptrend = h_output["values"]["isUptrend"].fill_null(False).cast(pl.Float64)
    is_downtrend = h_output["values"]["isDowntrend"].fill_null(False).cast(pl.Float64)
    fvg_bull = h_output["values"]["fvgBull"].fill_null(False).cast(pl.Float64)
    fvg_bear = h_output["values"]["fvgBear"].fill_null(False).cast(pl.Float64)
    ob_bull = h_output["values"]["obBull"].fill_null(False).cast(pl.Float64)
    ob_bear = h_output["values"]["obBear"].fill_null(False).cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", 55)
    adapt_micro_p = params.get("adapt_micro_p", 13)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    epsilon = 1e-6

    # 【核心：降維與連續化映射】
    # 將離散信號組合為 SMC Regime Score [-1.0, 1.0]
    # 權重配置：宏觀結構趨勢 40%，微觀訂單塊統治 40%，失衡區(FVG)動能 20%
    structure_score = (is_uptrend - is_downtrend) * 0.4
    ob_score = (ob_bull - ob_bear) * 0.4
    fvg_score = (fvg_bull - fvg_bear) * 0.2

    smc_score = structure_score + ob_score + fvg_score

    # ---------------------------------------------------------
    # (A) Position (SMC 政權中樞): 綜合評分的宏觀長期衰減水位
    # 語意補值: 0.0 (長期多空勢均力敵，或皆無明確聰明錢信號)
    # ---------------------------------------------------------
    regime = smc_score.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：約束於 [-0.8, 0.8]
    feat_smc_position_stable = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-0.8, 0.8).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]
    feat_smc_position_sensitive = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (政權微觀乖離): 當前 K 棒的綜合評分相對於長線政權的偏離
    # 語意補值: 0.0 (當前微觀行為完美符合近期宏觀聰明錢意圖)
    # ---------------------------------------------------------
    bias = smc_score - regime

    # Stable 版：約束於 [-1.0, 1.0]
    feat_smc_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-2.0, 2.0]，捕捉從死水瞬間切換至完美多頭共振的斷層
    feat_smc_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (政權切換加速度): SMC 綜合評分的變化速度
    # 語意補值: 0.0 (聰明錢狀態維持不變)
    # 降共線性處理: 減去短線 EMA 並自適應標準化
    # ---------------------------------------------------------
    score_ema_micro = smc_score.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (smc_score - score_ema_micro) / (score_ema_micro.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0]
    feat_smc_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極致捕捉市場環境巨變的動能
    feat_smc_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (結構混沌度): SMC 評分的歷史變異數
    # 語意補值: 0.0 (市場死心塌地維持單一環境)
    # 防禦處理: 強制套用 log1p 平滑
    # 數值飆高代表市場在「結構破壞」與「假突破獵殺」間頻繁洗盤
    # ---------------------------------------------------------
    score_volatility = smc_score.rolling_std(window_size=adapt_vol_p)
    log_score_vol = score_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_smc_volatility_stable = (
        log_score_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]
    feat_smc_volatility_sensitive = (
        log_score_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_smc_position_stable": feat_smc_position_stable,
        "feat_smc_position_sensitive": feat_smc_position_sensitive,
        "feat_smc_bias_stable": feat_smc_bias_stable,
        "feat_smc_bias_sensitive": feat_smc_bias_sensitive,
        "feat_smc_momentum_stable": feat_smc_momentum_stable,
        "feat_smc_momentum_sensitive": feat_smc_momentum_sensitive,
        "feat_smc_volatility_stable": feat_smc_volatility_stable,
        "feat_smc_volatility_sensitive": feat_smc_volatility_sensitive,
    }
