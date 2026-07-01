# ==============================================================================
# § 指標 | SMC 流動性 (Liquidity)
# 核心職責: 識別 EQH/EQL，標記市場流動性池與潛在的獵殺目標 (Stop Hunt)。
# v2.0 更新: [狀態機修正版] 修正擺動點比較邏輯。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| left          | H & G | int  | 3 ~ 10   | 無 (必填)       | 擺動點 (Pivot) 左側所需的 K 棒數 |
| right         | H & G | int  | 3 ~ 10   | 無 (必填)       | 擺動點 (Pivot) 右側所需的 K 棒數 |
| threshold_pct | H & G | float| 0.0~0.005| 無 (必填)       | 容許的等高/等低價格誤差百分比 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | 55              | 用於 Position (流動性政權) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 5 ~ 21   | 13              | 用於 Bias (短線流動性記憶) 的 EMA 衰減週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | 34              | 用於 Volatility (流動性群集度) 的滾動標準差週期 |

【特徵工程說明】
- EQH/EQL 是極度稀疏的事件脈衝。G 接口將其擴展為連續的時間面特徵。
- 透過 adapt_macro_p 觀察流動性池的宏觀堆積方向 (上方或下方)。
- 透過 adapt_vol_p 衡量流動性事件的群集程度，識別即將發生大型獵殺的混沌期。
"""
from typing import Dict

import polars as pl

from src.features.functions.abs import calculate as abs_val
from src.features.functions.pivots import calculate as pivots
from src.features.functions.shift import calculate as prev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留精確的布林觸發信號。
    v2.0 核心升級:
    - [根本性設計修正] 廢除了 v1.0 中要求在「同一個」pivot high/low K線上
      進行比較的嚴苛邏輯。
    - [引入狀態機] 新邏輯改為當一個「新的」擺動點形成時，它會去跟
      「前一個」已確立的擺動點進行比較，這才是 EQH/EQL 的正確定義。

    契約：
    - df 必須包含 'high', 'low' 欄位。
    - params 必須包含 'left', 'right', 'threshold_pct' 鍵。
    """
    left = params["left"]
    right = params["right"]
    threshold_pct = params["threshold_pct"]

    # --- 1. 識別擺動點 (Pivots) ---
    pivots_high = pivots(series=pl.col("high"), left=left, right=right)
    pivots_low = pivots(series=pl.col("low"), left=left, right=right)
    is_pivot_high = pivots_high == 1
    is_pivot_low = pivots_low == -1

    # --- 2. 獲取最近的擺動點「值」 ---
    # 當出現新的擺動點時，記錄其值，否則沿用上一個值
    last_ph_price = (
        pl.when(is_pivot_high).then(pl.col("high")).otherwise(None).forward_fill()
    )
    last_pl_price = (
        pl.when(is_pivot_low).then(pl.col("low")).otherwise(None).forward_fill()
    )

    # --- 3. 獲取「上一個」已確立的擺動點值 ---
    prev_swing_ph_price = prev(series=last_ph_price, period=1)
    prev_swing_pl_price = prev(series=last_pl_price, period=1)

    # --- 4. 計算價格差異百分比 ---
    # 當一個新的 pivot high 出現時，用它的價格去和前一個 pivot high 的價格比較
    high_diff_pct = (
        abs_val(series=(pl.col("high") - prev_swing_ph_price)) / prev_swing_ph_price
    )
    low_diff_pct = (
        abs_val(series=(pl.col("low") - prev_swing_pl_price)) / prev_swing_pl_price
    )

    # --- 5. 組合條件 ---
    # 條件觸發於「新的」擺動點形成的那一根 K 棒
    is_eqh = is_pivot_high & (high_diff_pct < threshold_pct)
    is_eql = is_pivot_low & (low_diff_pct < threshold_pct)

    return {"type": "event", "values": {"isEQH": is_eqh, "isEQL": is_eql}}


def adapt_Liquidity(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將極度稀疏的 EQH/EQL 脈衝事件，轉換為 DL/ML 可學習的連續時空特徵。
    正交分解為：流動性政權中樞 (Position)、短線流動性記憶 (Bias)、瞬時脈衝 (Momentum)、流動性群集度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端獵殺) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，政權衰減與群集觀察週期全面可由 YAML 配置。
    """
    is_eqh = h_output["values"]["isEQH"]
    is_eql = h_output["values"]["isEQL"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    # 由於基礎參數是極短週期的 pivots，特徵工程需要較長週期來做事件衰減，故提供預設常數
    adapt_macro_p = params.get("adapt_macro_p", 55)
    adapt_micro_p = params.get("adapt_micro_p", 13)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    # ---------------------------------------------------------
    # (A) Momentum (瞬時脈衝): 當下 K 棒發生 EQH 還是 EQL
    # 語意補值: 0.0 (無事件)
    # EQH 代表上方累積流動性 (多頭目標) 設為 1.0；EQL 代表下方累積流動性 (空頭目標) 設為 -1.0
    # ---------------------------------------------------------
    impulse = (pl.when(is_eqh).then(1.0).when(is_eql).then(-1.0).otherwise(0.0)).cast(
        pl.Float64
    )

    # Stable 版與 Sensitive 版先天已約束於 [-1.0, 1.0]
    feat_liquidity_momentum_stable = impulse
    feat_liquidity_momentum_sensitive = impulse

    # ---------------------------------------------------------
    # (B) Bias (短線流動性記憶): 事件的短期衰減餘波
    # 語意補值: 0.0 (近期無任何流動性堆積)
    # ---------------------------------------------------------
    short_memory = impulse.ewm_mean(span=adapt_micro_p, ignore_nulls=True)

    # Stable 版：EMA 壓縮數值，約束於 [-0.5, 0.5]
    feat_liquidity_bias_stable = (
        short_memory.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0]
    feat_liquidity_bias_sensitive = (
        short_memory.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Position (流動性政權中樞): 長期流動性堆積方向
    # 語意補值: 0.0 (長期來看上方與下方流動性勢均力敵)
    # ---------------------------------------------------------
    long_memory = impulse.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：約束於 [-0.2, 0.2]
    feat_liquidity_position_stable = (
        long_memory.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.5, 0.5]
    feat_liquidity_position_sensitive = (
        long_memory.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (流動性群集度): 流動性事件的歷史變異數
    # 語意補值: 0.0 (無頻繁的等高/等低點形成)
    # 防禦處理: 強制套用 log1p 平滑
    # ---------------------------------------------------------
    liquidity_vol = impulse.rolling_std(window_size=adapt_vol_p)
    log_liquidity_vol = liquidity_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_liquidity_volatility_stable = (
        log_liquidity_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]
    feat_liquidity_volatility_sensitive = (
        log_liquidity_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_liquidity_momentum_stable": feat_liquidity_momentum_stable,
        "feat_liquidity_momentum_sensitive": feat_liquidity_momentum_sensitive,
        "feat_liquidity_bias_stable": feat_liquidity_bias_stable,
        "feat_liquidity_bias_sensitive": feat_liquidity_bias_sensitive,
        "feat_liquidity_position_stable": feat_liquidity_position_stable,
        "feat_liquidity_position_sensitive": feat_liquidity_position_sensitive,
        "feat_liquidity_volatility_stable": feat_liquidity_volatility_stable,
        "feat_liquidity_volatility_sensitive": feat_liquidity_volatility_sensitive,
    }
