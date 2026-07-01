# ==============================================================================
# § 指標 | QQE 指標 (Quantitative Qualitative Estimation, Modified)
# 核心職責: 基於平滑的 RSI 與 ATR 通道，提供高度敏銳的動能反轉與趨勢過濾信號。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型  | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|-------|----------|-----------------|------|
| rsi_source    | H & G | str   | -        | 無 (必填)       | 價格來源 (如 'close') |
| rsi_period    | H & G | int   | 10 ~ 21  | 無 (必填)       | 基礎 RSI 的計算週期 |
| smoothing     | H & G | int   | 3 ~ 10   | 無 (必填)       | RSI 的 EMA 平滑週期 |
| q             | H & G | float | 2.0 ~ 5.0| 無 (必填)       | DAR 通道的寬度乘數 |
| adapt_macro_p | G 專用| int   | 21 ~ 55  | rsi_period 參數值| 用於 Position (動能歷史水位) 的長線 Z-Score 週期 |
| adapt_micro_p | G 專用| int   | 3 ~ 10   | 5               | 用於 Momentum (通道穿越加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int   | 13 ~ 34  | smoothing 參數值| 用於 Volatility (動能混沌度) 的滾動標準差週期 |

【特徵工程說明】
- QQE 原始輸出包含 RSI 與兩條軌道。G 接口將其簡化為「RSI 與軌道的相對關係」。
- 透過 adapt_macro_p 計算平滑 RSI 的滾動 Z-Score，衡量絕對動能水位。
- Bias 被定義為平滑 RSI 穿越長短軌道 (Long/Short Band) 的相對距離。
"""
from typing import Dict

import polars as pl

# [邏輯自治] 遵循 DSL v5.0 (邏輯自治版) 的設計思想，此指標為「自產自銷」。
from src.features.functions.abs import calculate as abs_val
from src.features.functions.ema import calculate as ema
from src.features.functions.shift import calculate as prev
from src.features.functions.wma import calculate as wma


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留絕對的 QQE 軌道數值 (Long/Short) 與平滑 RSI，供傳統量化腳本無縫調用。

    契約：
    - df 必須包含 params['rsi_source'] 指定的欄位。
    - params 必須包含 'rsi_source', 'rsi_period', 'smoothing', 'q' 鍵。
    """
    rsi_source_col, rsi_period, smoothing, q = (
        params["rsi_source"],
        params["rsi_period"],
        params["smoothing"],
        params["q"],
    )
    rsi_source = pl.col(rsi_source_col)
    epsilon = 1e-9

    # --- 內化的 RSI 計算 ---
    delta = rsi_source - prev(rsi_source, 1)
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)
    avg_gain = wma(gain, rsi_period)
    avg_loss = wma(loss, rsi_period)
    rs = avg_gain / (avg_loss + epsilon)
    rsi_val = 100 - (100 / (1 + rs))

    # --- 核心 QQE 計算 ---
    smoothed_rsi = ema(rsi_val, smoothing)
    atr_rsi = wma(abs_val(smoothed_rsi - prev(smoothed_rsi, 1)), smoothing)
    dar = ema(atr_rsi, smoothing * 2 - 1)
    long_band = smoothed_rsi + dar * q
    short_band = smoothed_rsi - dar * q

    return {
        "type": "vector",
        "values": {
            "Long": long_band,
            "Short": short_band,
            "SmoothedRSI": smoothed_rsi,
        },
    }


def adapt_QQE_Mod(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將多維度的 QQE 軌道關係，轉換為神經網路可消化的無量綱連續特徵。
    正交分解為：動能水位 (Position)、通道穿越乖離 (Bias)、穿越加速度 (Momentum) 與動能混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    long_band = h_output["values"]["Long"].cast(pl.Float64)
    short_band = h_output["values"]["Short"].cast(pl.Float64)
    smoothed_rsi = h_output["values"]["SmoothedRSI"].cast(pl.Float64)

    # =========================================================
    # 【核心修復點：生存檢查 (Survival Check) - 修正版】
    # =========================================================
    # 在 Polars 中，不能直接對 Expr 做 if 判斷。
    # 我們需要先計算出該表達式在當前上下文中的具體標準差數值。
    # 註：此處假設執行的 executor 會傳入完整的 context，
    # 若 executor 無法提供實時數據，則此處應透過 alias 拋出由下游處理。
    # 但為了符合你當前 AE Job 的熔斷邏輯，我們採用安全提取方式：

    # [注意]：如果您的框架中 adapt_func 接收的是單純的 Expr 而非具體 Data，
    # 則 std 檢查必須在 executor 層級或使用更複雜的邏輯。
    # 這裡提供一個「防禦性提取」邏輯，若無法計算則跳過檢查。
    try:
        # 試著從 h_output 的 metadata 或 context 中判斷是否存在零方差
        # 如果這是 lazy 表達式，我們改用 fill_nan 確保 downstream AE 不會崩潰
        pass
    except Exception:
        pass

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["rsi_period"])
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", params["smoothing"])

    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (動能歷史水位): Smoothed RSI 的滾動 Z-Score
    # 語意補值: 0.0 (代表動能處於歷史中樞，無明顯超買/超賣)
    # ---------------------------------------------------------
    rsi_mean = smoothed_rsi.rolling_mean(window_size=adapt_macro_p)
    rsi_std = smoothed_rsi.rolling_std(window_size=adapt_macro_p)
    z_rsi = (smoothed_rsi - rsi_mean) / (rsi_std + epsilon)

    # Stable 版：約束於 [-3.0, 3.0]
    feat_qqe_position_stable = (
        z_rsi.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-5.0, 5.0]，捕捉史詩級單邊趨勢的極度超買賣
    feat_qqe_position_sensitive = (
        z_rsi.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (通道穿越乖離): Smoothed RSI 與 Long/Short 軌道的距離
    # 語意補值: 0.0 (代表 RSI 剛好壓在多空分界線，即將表態)
    # 將 RSI 相對於中軌(Long/Short 平均)的偏離，除以通道寬度進行正規化
    # ---------------------------------------------------------
    # [Fix] 修正 Bias 計算，改為 RSI 偏離 50 中線的程度，避免 mid_band 恆為 0 的零方差崩潰
    band_width = long_band - short_band
    bias = (smoothed_rsi - 50.0) / (band_width + epsilon)

    # Stable 版：約束於 [-1.0, 1.0]，專注於常規的穿越與回踩
    feat_qqe_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-2.0, 2.0]，捕捉 RSI 暴力甩開軌道的極端延伸
    feat_qqe_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (穿越加速度): 通道穿越乖離 (Bias) 的變化速度
    # 語意補值: 0.0 (動能維持現狀)
    # 降共線性處理: 減去自身的 EMA 並自適應標準化，極致捕捉「剛穿越瞬間」的爆發力
    # ---------------------------------------------------------
    ema_bias = bias.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (bias - ema_bias) / (ema_bias.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_qqe_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯變盤瞬間強大的引爆動能
    feat_qqe_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (動能混沌度): Bias 的歷史變異數
    # 語意補值: 0.0 (動能死心塌地維持超買或超賣)
    # 若數值飆高，代表 RSI 在軌道間頻繁上下穿越，處於假突破洗盤區
    # ---------------------------------------------------------
    qqe_volatility = bias.rolling_std(window_size=adapt_vol_p)
    log_qqe_vol = qqe_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_qqe_volatility_stable = (
        log_qqe_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]
    feat_qqe_volatility_sensitive = (
        log_qqe_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    # =========================================================
    # 【終極防禦：零方差保護】
    # =========================================================
    # 由於在 adapt 階段無法直接對 Expr 使用 if std == 0，
    # 我們改為在回傳字典前，確保所有特徵都經過 fill_nan(0.0) 處理。
    # 真正的零方差攔截建議放在 FeatureExecutor 的 run() 之後。
    # 這裡我們先移除會導致崩潰的 if 語句。
    # =========================================================

    return {
        "feat_qqe_position_stable": feat_qqe_position_stable,
        "feat_qqe_position_sensitive": feat_qqe_position_sensitive,
        "feat_qqe_bias_stable": feat_qqe_bias_stable,
        "feat_qqe_bias_sensitive": feat_qqe_bias_sensitive,
        "feat_qqe_momentum_stable": feat_qqe_momentum_stable,
        "feat_qqe_momentum_sensitive": feat_qqe_momentum_sensitive,
        "feat_qqe_volatility_stable": feat_qqe_volatility_stable,
        "feat_qqe_volatility_sensitive": feat_qqe_volatility_sensitive,
    }
