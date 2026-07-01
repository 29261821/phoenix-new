# ==============================================================================
# § 指標 | 唐奇安通道 (Donchian Channels)
# 核心職責: 透過擷取指定週期內的絕對最高價與最低價，建立價格的極值邊界與突破通道。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| period        | H & G | int  | 10 ~ 50  | 無 (必填)       | 唐奇安通道的極值觀察週期 |
| adapt_micro_p | G 專用| int  | 5 ~ 14   | period 參數的值 | 用於 Momentum (突破加速度) 計算時的 EMA 平滑週期，隔離共線性 |

【特徵工程說明】
- 原始唐奇安通道為絕對價格極值，G 接口將其轉換為無量綱的相對位置 (%D)、極值頻寬與中樞乖離。
- 透過 adapt_micro_p 決定模型對「撞擊極值邊界加速度」的敏感度。
"""
from typing import Dict

import polars as pl

from src.features.functions.highest import calculate as highest
from src.features.functions.lowest import calculate as lowest


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留原始的絕對價格軌道 (Upper, Middle, Lower)。
    確保依賴「價格穿越唐奇安上軌」作為突破信號的海龜交易等傳統策略能無縫對接。

    契約：
    - df 必須包含 'high', 'low' 欄位。
    - params 必須包含 'period' 鍵。
    """
    period = params["period"]

    upper = highest(series=pl.col("high"), period=period)
    lower = lowest(series=pl.col("low"), period=period)
    middle = (upper + lower) / 2.0

    return {
        "type": "vector",
        "values": {"Upper": upper, "Middle": middle, "Lower": lower},
    }


def adapt_Donchian(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將具備階梯狀 (Staircase) 特性的唐奇安通道轉換為供 DL/ML 使用的無尺度連續特徵。
    正交分解為：相對位置 (%D, Position)、極值頻寬 (Volatility)、極值乖離 (Bias)、突破動能 (Momentum)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端擴張) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，動能平滑週期可由 YAML 配置。
    """
    upper = h_output["values"]["Upper"]
    middle = h_output["values"]["Middle"]
    lower = h_output["values"]["Lower"]

    # 1. 提取基礎參數
    period = params["period"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_micro_p = params.get("adapt_micro_p", period)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (通道相對位置): 價格在極值上下軌之間的相對座標 (%D)
    # 語意補值: 0.5 (代表處於歷史高低點的正中央，處於盤整或方向不明)
    # 由於唐奇安通道包含當下 K 棒，理論上數值必落於 [0, 1] 之間。1.0 代表創新高。
    # ---------------------------------------------------------
    pct_d = (close - lower) / (upper - lower + epsilon)

    # Stable 版：嚴格約束於 [0.0, 1.0] 內，穩定 Transformer 注意力機制
    feat_donchian_position_stable = (
        pct_d.fill_nan(0.5).fill_null(0.5).clip(0.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：略微放寬至 [-0.1, 1.1]，容許資料偏移或前置處理造成的微小越界
    feat_donchian_position_sensitive = (
        pct_d.fill_nan(0.5).fill_null(0.5).clip(-0.1, 1.1).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Volatility (極值頻寬 / 突破空間): 歷史最高與最低點的距離佔價格的百分比
    # 語意補值: 0.0 (極度收斂、無波動)
    # 防禦處理: 強制套用 log1p 平滑極端跳空時的邊界擴張
    # ---------------------------------------------------------
    bandwidth = (upper - lower) / (middle + epsilon)
    log_bandwidth = bandwidth.log1p()

    # Stable 版：約束於 [0.0, 0.3] (容許最多 30% 的極值通道寬度)
    feat_donchian_volatility_stable = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.3).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0] (捕捉如 Crypto 史詩級拉升時的通道暴增)
    feat_donchian_volatility_sensitive = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Bias (極值中樞乖離): 價格相對於唐奇安中軌 (剛性中樞) 的偏離
    # 語意補值: 0.0 (貼合歷史高低點平衡中樞)
    # ---------------------------------------------------------
    bias = (close / (middle + epsilon)) - 1.0

    # Stable 版：約束於 [-0.15, 0.15]，代表最多 15% 的中樞偏離
    feat_donchian_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.15, 0.15).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.5, 0.5]，保留高波動資產的超限偏離資訊
    feat_donchian_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Momentum (突破加速度): 位置逼近率 (%D) 的變化速度
    # 語意補值: 0.0 (價格在通道內平穩遊走，無撞擊邊界的動能)
    # 降共線性處理: 減去自身的 EMA 並標準化，凸顯瞬間撞擊上下軌的突破爆發力
    # ---------------------------------------------------------
    ema_pct_d = pct_d.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    pct_d_osc = (pct_d - ema_pct_d) / (ema_pct_d.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_donchian_momentum_stable = (
        pct_d_osc.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，捕捉突破瞬間的恐怖加速度
    feat_donchian_momentum_sensitive = (
        pct_d_osc.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    return {
        "feat_donchian_position_stable": feat_donchian_position_stable,
        "feat_donchian_position_sensitive": feat_donchian_position_sensitive,
        "feat_donchian_volatility_stable": feat_donchian_volatility_stable,
        "feat_donchian_volatility_sensitive": feat_donchian_volatility_sensitive,
        "feat_donchian_bias_stable": feat_donchian_bias_stable,
        "feat_donchian_bias_sensitive": feat_donchian_bias_sensitive,
        "feat_donchian_momentum_stable": feat_donchian_momentum_stable,
        "feat_donchian_momentum_sensitive": feat_donchian_momentum_sensitive,
    }
