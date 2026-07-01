# ==============================================================================
# § 指標 | 布林帶 (Bollinger Bands)
# 核心職責: 透過移動平均線與標準差構建市場價格的動態邊界與波動率觀測器。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型  | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|-------|----------|-----------------|------|
| source        | H & G | str   | -        | 無 (必填)       | 計算價格來源 (如 'close') |
| period        | H & G | int   | 10 ~ 50  | 無 (必填)       | 布林帶 SMA 與標準差計算週期 |
| std           | H     | float | 1.5 ~ 3.0| 無 (必填)       | 決定通道寬度的標準差乘數 |
| adapt_micro_p | G 專用| int   | 5 ~ 14   | period 參數的值 | 用於 Momentum (動量) 計算時的 EMA 平滑週期，隔離共線性 |

【特徵工程說明】
- 原始布林帶為絕對價格位準，G 接口將其轉換為無量綱的相對位置 (%B)、頻寬與乖離。
- 透過 adapt_micro_p 決定模型對「撞擊軌道加速度」的敏感度。
"""
from typing import Dict

import polars as pl

from src.features.functions.sma import calculate as sma
from src.features.functions.stddev import calculate as stddev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留原始絕對價格指標，不做任何無量綱化或 Clip，
    確保舊有腳本、自然語言策略可無縫取得 Upper, Middle, Lower 價格位準。

    契約：
    - df 必須包含 'close' 欄位 (或 params['source'] 指定的欄位)。
    - params 必須包含 'source', 'period', 'std' 鍵。
    """
    source_col = params["source"]
    period = params["period"]
    std_mult = params["std"]

    source = pl.col(source_col)
    middle = sma(series=source, length=period)
    stdev_val = stddev(series=source, period=period)

    upper = middle + stdev_val * std_mult
    lower = middle - stdev_val * std_mult

    return {
        "type": "vector",
        "values": {"Upper": upper, "Middle": middle, "Lower": lower},
    }


def adapt_Bollinger_Bands(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將 H 接口的絕對價格轉換為供 DL/ML 使用的無量綱穩定特徵。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端行情) 雙版本。
    天然支援多時間框架 (Multi-timeframe) 融合，所有輸出皆具備數值防護。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，動能平滑週期可由 YAML 配置。
    """
    upper = h_output["values"]["Upper"]
    middle = h_output["values"]["Middle"]
    lower = h_output["values"]["Lower"]

    # 1. 提取基礎參數
    period = params["period"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_micro_p = params.get("adapt_micro_p", period)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN，避免梯度崩潰
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (位置特徵): 衡量價格在上下軌間的相對座標 (%B)
    # 語意補值: 0.5 (代表處於中樞中立位置)
    # ---------------------------------------------------------
    pct_b = (close - lower) / (upper - lower + epsilon)

    # Stable 版：嚴格約束於 [0.0, 1.0] 內，防止模型 Activation 偏移，適合 Transformer
    feat_bb_position_stable = (
        pct_b.fill_nan(0.5).fill_null(0.5).clip(0.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-1.0, 2.0]，保留價格嚴重刺穿軌道的極端厚尾資訊，適合 Tree-based
    feat_bb_position_sensitive = (
        pct_b.fill_nan(0.5).fill_null(0.5).clip(-1.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Volatility (波動特徵): 衡量通道相對頻寬，捕捉波動擴張/收斂
    # 語意補值: 0.0 (代表極度收斂、無波動)
    # 防禦處理: 強制套用 log1p 平滑極端變異數爆炸
    # ---------------------------------------------------------
    bandwidth = (upper - lower) / (middle + epsilon)
    log_bandwidth = bandwidth.log1p()

    # Stable 版：約束於 [0.0, 3.0]
    feat_bb_volatility_stable = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 3.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 10.0]，允許捕捉黑天鵝級別的波動擴張
    feat_bb_volatility_sensitive = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 10.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Bias (偏離特徵): 價格相對於中軌的乖離率
    # 語意補值: 0.0 (代表完美貼合中軌)
    # ---------------------------------------------------------
    bias = (close / (middle + epsilon)) - 1.0

    # Stable 版：約束於 [-0.5, 0.5]，代表最多 50% 的偏離
    feat_bb_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-3.0, 3.0]，保留如 Crypto 等高波動資產的超限偏離
    feat_bb_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Momentum (動量特徵): 位置特徵 (%B) 的加速度
    # 語意補值: 0.0 (代表無動能方向)
    # 降共線性處理: 減去自身的 EMA 並標準化，隔離與 Bias 的特徵重疊
    # ---------------------------------------------------------
    ema_pct_b = pct_b.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    pct_b_osc = (pct_b - ema_pct_b) / (ema_pct_b.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_bb_momentum_stable = (
        pct_b_osc.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束，捕捉極強的瞬間爆發動能
    feat_bb_momentum_sensitive = (
        pct_b_osc.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    return {
        "feat_bb_position_stable": feat_bb_position_stable,
        "feat_bb_position_sensitive": feat_bb_position_sensitive,
        "feat_bb_volatility_stable": feat_bb_volatility_stable,
        "feat_bb_volatility_sensitive": feat_bb_volatility_sensitive,
        "feat_bb_bias_stable": feat_bb_bias_stable,
        "feat_bb_bias_sensitive": feat_bb_bias_sensitive,
        "feat_bb_momentum_stable": feat_bb_momentum_stable,
        "feat_bb_momentum_sensitive": feat_bb_momentum_sensitive,
    }
