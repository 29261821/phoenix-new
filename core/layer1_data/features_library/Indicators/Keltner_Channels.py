# ==============================================================================
# § 指標 | 肯特納通道 (Keltner Channels)
# 核心職責: 以 EMA 為中軌，ATR 為寬度，衡量市場真實波動邊界與趨勢突破。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型  | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|-------|----------|-----------------|------|
| ema_period    | H & G | int   | 10 ~ 50  | 無 (必填)       | 中軌 EMA 的計算週期 |
| atr_len       | H & G | int   | 10 ~ 22  | 無 (必填)       | ATR 波動率的計算週期 |
| multiplier    | H     | float | 1.5 ~ 3.0| 無 (必填)       | 決定通道寬度的 ATR 乘數 |
| adapt_micro_p | G 專用| int   | 5 ~ 14   | ema_period 參數 | 用於 Momentum (穿透動能) 計算時的 EMA 平滑週期 |

【特徵工程說明】
- 原始通道為絕對價格位準，G 接口將其轉換為無量綱的相對位置 (%K)、頻寬與乖離。
- 透過 adapt_micro_p 決定模型對「撞擊軌道或突破軌道加速度」的敏感度。
"""
from typing import Dict

import polars as pl

from src.features.functions.atr import calculate as atr
from src.features.functions.ema import calculate as ema


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留原始絕對價格指標，不做任何無量綱化，
    確保舊有腳本、自然語言策略可無縫取得 Upper, Middle, Lower 價格位準作為支撐壓力。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含 'ema_period', 'atr_len', 'multiplier' 鍵。
    """
    ema_period = params["ema_period"]
    atr_len = params["atr_len"]
    multiplier = params["multiplier"]

    middle = ema(series=pl.col("close"), length=ema_period)
    atr_val = atr(df=df, period=atr_len)

    upper = middle + (atr_val * multiplier)
    lower = middle - (atr_val * multiplier)

    return {
        "type": "vector",
        "values": {"Upper": upper, "Middle": middle, "Lower": lower},
    }


def adapt_Keltner_Channels(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將絕對價格轉換為供 DL/ML 使用的無量綱穩定特徵。
    正交分解為：相對位置 (%K, Position)、通道頻寬 (Volatility)、中樞乖離 (Bias)、穿透動能 (Momentum)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端行情) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，動能平滑週期可由 YAML 配置。
    """
    upper = h_output["values"]["Upper"]
    middle = h_output["values"]["Middle"]
    lower = h_output["values"]["Lower"]

    # 1. 提取基礎參數
    ema_period = params["ema_period"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_micro_p = params.get("adapt_micro_p", ema_period)

    # 數值穩定性防護常數
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (位置特徵): 衡量價格在上下軌間的相對座標 (%K)
    # 語意補值: 0.5 (代表處於中樞 EMA 中立位置)
    # ---------------------------------------------------------
    pct_k = (close - lower) / (upper - lower + epsilon)

    # Stable 版：約束於 [0.0, 1.0] 內，防止模型 Activation 偏移
    feat_keltner_position_stable = (
        pct_k.fill_nan(0.5).fill_null(0.5).clip(0.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-1.0, 2.0]，保留價格嚴重刺穿軌道的突破資訊
    feat_keltner_position_sensitive = (
        pct_k.fill_nan(0.5).fill_null(0.5).clip(-1.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Volatility (波動特徵): 衡量通道相對頻寬 (ATR 佔價格的百分比)
    # 語意補值: 0.0 (極度收斂、無波動)
    # ---------------------------------------------------------
    bandwidth = (upper - lower) / (middle + epsilon)
    log_bandwidth = bandwidth.log1p()

    # Stable 版：約束於 [0.0, 0.2] (容許最多 20% 的通道寬度)
    feat_keltner_volatility_stable = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.5] (允許捕捉黑天鵝級別的 ATR 擴張)
    feat_keltner_volatility_sensitive = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Bias (偏離特徵): 價格相對於中軌 EMA 的乖離率
    # 語意補值: 0.0 (完美貼合中軌)
    # ---------------------------------------------------------
    bias = (close / (middle + epsilon)) - 1.0

    # Stable 版：約束於 [-0.1, 0.1]，代表最多 10% 的偏離
    feat_keltner_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.1, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.3, 0.3]，保留高波動資產的超限偏離
    feat_keltner_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.3, 0.3).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Momentum (動量特徵): 位置特徵 (%K) 的加速度
    # 語意補值: 0.0 (無動能方向)
    # 降共線性處理: 減去自身的 EMA 並標準化
    # ---------------------------------------------------------
    ema_pct_k = pct_k.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    pct_k_osc = (pct_k - ema_pct_k) / (ema_pct_k.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_keltner_momentum_stable = (
        pct_k_osc.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束，捕捉極強的瞬間突破或回馬槍洗盤動能
    feat_keltner_momentum_sensitive = (
        pct_k_osc.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    return {
        "feat_keltner_position_stable": feat_keltner_position_stable,
        "feat_keltner_position_sensitive": feat_keltner_position_sensitive,
        "feat_keltner_volatility_stable": feat_keltner_volatility_stable,
        "feat_keltner_volatility_sensitive": feat_keltner_volatility_sensitive,
        "feat_keltner_bias_stable": feat_keltner_bias_stable,
        "feat_keltner_bias_sensitive": feat_keltner_bias_sensitive,
        "feat_keltner_momentum_stable": feat_keltner_momentum_stable,
        "feat_keltner_momentum_sensitive": feat_keltner_momentum_sensitive,
    }
