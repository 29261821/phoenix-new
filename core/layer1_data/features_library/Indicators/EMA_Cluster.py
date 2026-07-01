# ==============================================================================
# § 指標 | 指數移動平均線簇 (EMA Cluster)
# 核心職責: 一次性計算並輸出一組不同週期的 EMA，構成多空均線帶 (Ribbon)。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| source        | H & G | str  | -        | 無 (必填)       | 價格來源 (如 'close') |
| len1          | H & G | int  | 5 ~ 13   | 無 (必填)       | EMA 簇的最短週期 (快線) |
| len2          | H & G | int  | 13 ~ 21  | 無 (必填)       | EMA 簇的中短週期 |
| len3          | H & G | int  | 21 ~ 55  | 無 (必填)       | EMA 簇的中長週期 |
| len4          | H & G | int  | 55 ~ 200 | 無 (必填)       | EMA 簇的最長週期 (慢線) |
| adapt_micro_p | G 專用| int  | 5 ~ 14   | len1 參數的值   | 用於 Momentum (均線扭轉動能) 計算時的 EMA 平滑週期，隔離共線性 |

【特徵工程說明】
- 四條絕對均線會被轉換為無量綱的相對位置 (%Ribbon)、發散頻寬與重心乖離。
- 透過 adapt_micro_p 決定模型對「價格穿透或跌破均線帶加速度」的敏感度。
"""
from typing import Dict

import polars as pl

from src.features.functions.ema import calculate as ema


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留各週期 EMA 的絕對價格。
    確保舊有基於均線交叉 (如 EMA_8 穿過 EMA_21) 的策略，或是依賴絕對價格位準
    作為動態支撐壓力的邏輯可以無縫調用。

    契約：
    - df 必須包含 params['source'] 指定的欄位。
    - params 必須包含 'source', 'len1', 'len2', 'len3', 'len4' 鍵。
    """
    source_col = params["source"]
    source = pl.col(source_col)

    len1 = params["len1"]
    len2 = params["len2"]
    len3 = params["len3"]
    len4 = params["len4"]

    return {
        "type": "vector",
        "values": {
            "EMA_8": ema(series=source, length=len1),
            "EMA_13": ema(series=source, length=len2),
            "EMA_21": ema(series=source, length=len3),
            "EMA_55": ema(series=source, length=len4),
        },
    }


def adapt_EMA_Cluster(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將四條絕對價格的均線轉換為 DL/ML 寬表特徵，消除尺度污染與共線性。
    正交分解為：均線帶相對位置 (Position)、均線發散度 (Volatility)、重心乖離 (Bias)、均線扭轉動能 (Momentum)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端乖離與發散) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，動能平滑週期可由 YAML 配置。
    """
    # 提取四組 EMA 均線
    ema1 = h_output["values"]["EMA_8"]
    ema2 = h_output["values"]["EMA_13"]
    ema3 = h_output["values"]["EMA_21"]
    ema4 = h_output["values"]["EMA_55"]

    # 1. 提取基礎參數
    len1 = params["len1"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    # 均線扭轉動能的平滑週期預設與最敏感的快線 (len1) 對齊
    adapt_micro_p = params.get("adapt_micro_p", len1)

    # 防禦性常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # 建立均線帶 (Ribbon) 的上、下邊界與重心
    ribbon_top = pl.max_horizontal([ema1, ema2, ema3, ema4])
    ribbon_bottom = pl.min_horizontal([ema1, ema2, ema3, ema4])
    center_of_gravity = (ema1 + ema2 + ema3 + ema4) / 4.0

    # ---------------------------------------------------------
    # (A) Position (均線帶相對位置): 價格在均線簇中或均線簇外的相對座標
    # 語意補值: 0.5 (代表價格剛好被困在均線帶正中央，處於極度糾結期)
    # > 1.0 代表噴出均線帶(多頭)，< 0.0 代表跌破均線帶(空頭)
    # ---------------------------------------------------------
    pct_ribbon = (close - ribbon_bottom) / (ribbon_top - ribbon_bottom + epsilon)

    # Stable 版：約束於 [-0.5, 1.5]，允許一定程度的突破，但壓制極端值以穩定 Transformer
    feat_ema_cluster_position_stable = (
        pct_ribbon.fill_nan(0.5).fill_null(0.5).clip(-0.5, 1.5).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-3.0, 4.0]，允許捕捉價格將均線遠遠甩在後面的主升/主跌段
    feat_ema_cluster_position_sensitive = (
        pct_ribbon.fill_nan(0.5).fill_null(0.5).clip(-3.0, 4.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Volatility (均線發散度 / 頻寬): 均線帶的寬度，衡量均線是糾纏還是呈扇形發散
    # 語意補值: 0.0 (四條均線完美重合，極度變盤前夕)
    # 防禦處理: 強制套用 log1p 平滑極端的趨勢爆發擴張
    # ---------------------------------------------------------
    ribbon_width = (ribbon_top - ribbon_bottom) / (center_of_gravity + epsilon)
    log_ribbon_width = ribbon_width.log1p()

    # Stable 版：約束於 [0.0, 0.2] (容許最多 20% 的均線發散寬度)
    feat_ema_cluster_volatility_stable = (
        log_ribbon_width.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.6] (捕捉史詩級單邊趨勢時的恐怖均線扇形發散)
    feat_ema_cluster_volatility_sensitive = (
        log_ribbon_width.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.6).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Bias (均線簇重心乖離): 價格相對於四條均線共識重心的偏離
    # 語意補值: 0.0 (代表價格完美貼合共識價值，無乖離)
    # ---------------------------------------------------------
    bias = (close / (center_of_gravity + epsilon)) - 1.0

    # Stable 版：約束於 [-0.2, 0.2]，過濾掉過度極端的均值回歸偏離
    feat_ema_cluster_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.8, 0.8]，捕捉市場極度超買/超賣時強大的均值拉扯力道
    feat_ema_cluster_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.8, 0.8).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Momentum (均線扭轉動能): 價格穿透均線帶的速度 (一階導數正規化)
    # 語意補值: 0.0 (價格與均線帶的相對位置維持平穩，無突破或跌破動作)
    # 降共線性處理: 減去自身的 EMA 並自適應標準化，凸顯瞬間改變均線結構的爆發力
    # ---------------------------------------------------------
    ema_pct_ribbon = pct_ribbon.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (pct_ribbon - ema_pct_ribbon) / (ema_pct_ribbon.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_ema_cluster_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，捕捉變盤瞬間暴力摜破所有均線的動能
    feat_ema_cluster_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    return {
        "feat_ema_cluster_position_stable": feat_ema_cluster_position_stable,
        "feat_ema_cluster_position_sensitive": feat_ema_cluster_position_sensitive,
        "feat_ema_cluster_volatility_stable": feat_ema_cluster_volatility_stable,
        "feat_ema_cluster_volatility_sensitive": feat_ema_cluster_volatility_sensitive,
        "feat_ema_cluster_bias_stable": feat_ema_cluster_bias_stable,
        "feat_ema_cluster_bias_sensitive": feat_ema_cluster_bias_sensitive,
        "feat_ema_cluster_momentum_stable": feat_ema_cluster_momentum_stable,
        "feat_ema_cluster_momentum_sensitive": feat_ema_cluster_momentum_sensitive,
    }
