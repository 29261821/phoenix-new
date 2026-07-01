# ==============================================================================
# § 指標 | 趨勢健康度引擎 (Trend Health Engine)
# 核心職責: 從多個維度診斷趨勢的健康狀況 (方向、排列與發散度)。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| fast_period   | H & G | int  | 8 ~ 21   | 無 (必填)       | 快速 EMA 計算週期 |
| mid_period    | H & G | int  | 21 ~ 55  | 無 (必填)       | 中速 EMA 計算週期 |
| slow_period   | H & G | int  | 55 ~ 200 | 無 (必填)       | 慢速 EMA 計算週期 |
| adapt_macro_p | G 專用| int  | 34 ~ 89  | slow_period 參數| 用於 Position (趨勢健康度中樞) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | fast_period 參數| 用於 Momentum (趨勢發散加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | mid_period 參數 | 用於 Volatility (趨勢結構混沌度) 的滾動標準差週期 |

【特徵工程說明】
- Trend Health 綜合了均線的排列 (Alignment) 與發散 (Dispersion)。
- 原始的 Alignment 是離散狀態 (-1, 0, 1)，Dispersion 則帶有絕對價格尺度。
- G 接口將 Dispersion 無量綱化轉為動能，並將 Alignment 連續化，提取宏觀健康度政權。
"""
from typing import Dict

import polars as pl

from src.features.functions.abs import calculate as abs_val
from src.features.functions.ema import calculate as ema
from src.features.functions.shift import calculate as prev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留原有的 Struct 封裝，同時解構出平坦化的維度特徵 (Direction, Alignment, Dispersion)，
    以兼顧舊有腳本調用與 G 接口的無縫特徵萃取。

    契約：
    - df 必須包含 'close' 欄位。
    - params 必須包含 'fast_period', 'mid_period', 'slow_period' 鍵。
    """
    fast_p, mid_p, slow_p = (
        params["fast_period"],
        params["mid_period"],
        params["slow_period"],
    )
    c = pl.col("close")

    fast = ema(series=c, length=fast_p)
    mid = ema(series=c, length=mid_p)
    slow = ema(series=c, length=slow_p)

    direction = pl.when(fast > slow).then(1).when(fast < slow).then(-1).otherwise(0)
    alignment = (
        pl.when((fast > mid) & (mid > slow))
        .then(1)
        .when((fast < mid) & (mid < slow))
        .then(-1)
        .otherwise(0)
    )
    # Dispersion: 快慢線距離的變化量 (正值代表發散，負值代表收斂)
    dispersion = abs_val(series=(fast - slow)) - abs_val(
        series=(prev(series=fast, period=1) - prev(series=slow, period=1))
    )

    health_context = pl.struct(
        [
            direction.alias("direction"),
            alignment.alias("alignment"),
            dispersion.alias("dispersion"),
        ]
    )

    return {
        "type": "vector",
        "values": {
            "Health": health_context,
            "Direction": direction,
            "Alignment": alignment,
            "Dispersion": dispersion,
        },
    }


def adapt_Trend_Health(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將包含狀態與絕對距離的混合特徵，轉換為無量綱的 DL/ML 特徵。
    正交分解為：趨勢健康度水位 (Position)、健康度乖離 (Bias)、趨勢發散加速度 (Momentum)、結構混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    alignment = h_output["values"]["Alignment"].cast(pl.Float64)
    dispersion = h_output["values"]["Dispersion"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["slow_period"])
    adapt_micro_p = params.get("adapt_micro_p", params["fast_period"])
    adapt_vol_p = params.get("adapt_vol_p", params["mid_period"])

    epsilon = 1e-6

    # 【核心：無量綱化 Dispersion】
    # 將價格絕對變化量除以股價，轉換為「百分比發散/收斂率」
    norm_dispersion = dispersion / (close + epsilon)

    # ---------------------------------------------------------
    # (A) Position (趨勢健康度水位): 均線排列 (Alignment) 的宏觀長期記憶
    # 語意補值: 0.0 (長期來看趨勢方向不明，處於均線糾纏狀態)
    # ---------------------------------------------------------
    regime = alignment.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：約束於 [-0.8, 0.8]，過濾極端單邊造成的絕對固化
    feat_th_position_stable = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-0.8, 0.8).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0] 的完整理論空間
    feat_th_position_sensitive = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (健康度微觀乖離): 當前 K 棒的排列狀態相對於長線政權的偏離
    # 語意補值: 0.0 (當前微觀行為完美符合近期宏觀趨勢健康度)
    # ---------------------------------------------------------
    bias = alignment - regime

    # Stable 版：約束於 [-1.0, 1.0]
    feat_th_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-2.0, 2.0]，捕捉從均線空頭排列 (-1) 瞬間突破至多頭 (+1) 的極端斷層
    feat_th_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (趨勢發散加速度): 正規化發散度 (norm_dispersion) 的變化速度
    # 語意補值: 0.0 (均線距離維持等距，無加速發散或收斂)
    # 降共線性處理: 減去自身的短線 EMA 並自適應標準化
    # ---------------------------------------------------------
    ema_disp = norm_dispersion.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (norm_dispersion - ema_disp) / (ema_disp.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_th_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極致捕捉變盤瞬間均線突然爆發張口的動能
    feat_th_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (趨勢結構混沌度): Alignment 排列狀態的歷史變異數
    # 語意補值: 0.0 (市場死心塌地維持單邊趨勢，極度順暢)
    # 防禦處理: 強制套用 log1p 平滑
    # 若數值飆高，代表市場均線在多頭與空頭之間頻繁扭轉打結
    # ---------------------------------------------------------
    align_vol = alignment.rolling_std(window_size=adapt_vol_p)
    log_align_vol = align_vol.log1p()

    # Stable 版：約束於 [0.0, 0.5]
    feat_th_volatility_stable = (
        log_align_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 1.0]
    feat_th_volatility_sensitive = (
        log_align_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    return {
        "feat_th_position_stable": feat_th_position_stable,
        "feat_th_position_sensitive": feat_th_position_sensitive,
        "feat_th_bias_stable": feat_th_bias_stable,
        "feat_th_bias_sensitive": feat_th_bias_sensitive,
        "feat_th_momentum_stable": feat_th_momentum_stable,
        "feat_th_momentum_sensitive": feat_th_momentum_sensitive,
        "feat_th_volatility_stable": feat_th_volatility_stable,
        "feat_th_volatility_sensitive": feat_th_volatility_sensitive,
    }
