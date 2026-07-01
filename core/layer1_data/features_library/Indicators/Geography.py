# ==============================================================================
# § 指標 | 價格地理位置 (Price Geography)
# 核心職責: 判斷價格相對於快慢均線的絕對地理位置，建立動態市場環境狀態機。
# v2.0 更新: [健壯性修正] 採用 .get() 方法讀取參數，並提供與 DSL 100%
#             對齊的預設值，以應對作戰計畫中的契約不匹配問題。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| fast_period   | H & G | int  | 10 ~ 30  | 20              | 快線 EMA 週期 |
| slow_period   | H & G | int  | 30 ~ 100 | 50              | 慢線 EMA 週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 89  | 21              | 用於 Position (地理政權中樞) 的長線 EMA 衰減週期 |
| adapt_micro_p | G 專用| int  | 5 ~ 14   | 9               | 用於 Momentum (政權穿透動能) 的短線 EMA 平滑週期 |

【特徵工程說明】
- 原始狀態機為離散地理位置 (1:強勢, -1:弱勢, 0:游離)，G 接口將其擴展為連續政權。
- 透過 adapt_macro_p 觀察政權的長期固化程度 (Market Regime)。
- 透過 adapt_micro_p 決定模型對「地理板塊瞬間切換」的動能敏感度，隔離共線性。
"""
from typing import Dict

import polars as pl

from src.features.functions.ema import calculate as ema


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留離散的地理狀態碼 (1: 強勢區, -1: 弱勢區, 0: 游離區)。
    確保依賴「狀態必須為 1 才允許做多」等傳統量化濾網能無縫執行。
    同時輸出 FastLine 與 SlowLine，供 G 接口進行無尺度化轉換。

    契約：
    - df 必須包含 'close' 欄位。
    - params 應包含 'fast_period', 'slow_period' 鍵。
    - [健壯性] 若參數未提供，則回退至 DSL 藍圖的預設值 (20, 50)。
    """
    # [健壯性修正] 使用 .get() 並提供 DSL 預設值
    fast_period = params.get("fast_period", 20)
    slow_period = params.get("slow_period", 50)
    c = pl.col("close")

    fast_line = ema(c, fast_period)
    slow_line = ema(c, slow_period)

    geo_state = (
        pl.when(c > fast_line)
        .then(pl.lit(1, dtype=pl.Int8))
        .when(c < slow_line)
        .then(pl.lit(-1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
    )

    return {
        "type": "event",
        "values": {"State": geo_state, "FastLine": fast_line, "SlowLine": slow_line},
    }


def adapt_Geography(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將粗糙的地理狀態與絕對價格均線，轉換為 DL/ML 寬表特徵。
    正交分解為：地理政權中樞 (Position)、地理環境發散度 (Volatility)、地理重心乖離 (Bias)、政權穿透動能 (Momentum)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端行情) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，政權衰減與動能平滑週期全面可由 YAML 配置。
    """
    state = h_output["values"]["State"].cast(pl.Float64)
    fast_line = h_output["values"]["FastLine"]
    slow_line = h_output["values"]["SlowLine"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", 21)
    adapt_micro_p = params.get("adapt_micro_p", 9)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (地理政權中樞): 長期地理狀態的衰減重心
    # 語意補值: 0.0 (代表市場頻繁穿梭於強弱區，無明確地理優勢)
    # ---------------------------------------------------------
    regime = state.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable 版：約束於 [-0.8, 0.8]，過濾極端絕對固化，穩定 Transformer 權重
    feat_geography_position_stable = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-0.8, 0.8).cast(pl.Float64)
    )
    # Sensitive 版：保留 [-1.0, 1.0] 的完整理論極限，捕捉極端單邊行情
    feat_geography_position_sensitive = (
        regime.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Volatility (地理環境發散度 / 頻寬): 快慢線的絕對距離佔慢線的百分比
    # 語意補值: 0.0 (快慢線完美重合，極度變盤前夕)
    # 防禦處理: 強制套用 log1p 平滑趨勢爆發時的極端發散
    # ---------------------------------------------------------
    bandwidth = (fast_line - slow_line).abs() / (slow_line + epsilon)
    log_bandwidth = bandwidth.log1p()

    # Stable 版：約束於 [0.0, 0.2] (容許最多 20% 的均線發散寬度)
    feat_geography_volatility_stable = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.5] (捕捉史詩級單邊趨勢時的恐怖發散)
    feat_geography_volatility_sensitive = (
        log_bandwidth.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Bias (地理中樞乖離): 價格相對於快慢線重心的偏離
    # 語意補值: 0.0 (代表價格完美貼合地理重心，無乖離)
    # ---------------------------------------------------------
    midpoint = (fast_line + slow_line) / 2.0
    bias = (close / (midpoint + epsilon)) - 1.0

    # Stable 版：約束於 [-0.15, 0.15]，過濾掉過度極端的均值回歸偏離
    feat_geography_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.15, 0.15).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.5, 0.5]，捕捉市場恐慌暴跌/狂熱暴漲時的極端乖離
    feat_geography_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Momentum (政權穿透動能): 地理狀態發生轉換的速度 (一階導數正規化)
    # 語意補值: 0.0 (地理狀態維持穩定，無突破或跌破動作)
    # 降共線性處理: 減去自身的 EMA 並自適應標準化，凸顯瞬間改變地理結構的爆發力
    # ---------------------------------------------------------
    state_ema_micro = state.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (state - state_ema_micro) / (state_ema_micro.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_geography_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，捕捉變盤瞬間暴力貫穿地理板塊的強大動能
    feat_geography_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    return {
        "feat_geography_position_stable": feat_geography_position_stable,
        "feat_geography_position_sensitive": feat_geography_position_sensitive,
        "feat_geography_volatility_stable": feat_geography_volatility_stable,
        "feat_geography_volatility_sensitive": feat_geography_volatility_sensitive,
        "feat_geography_bias_stable": feat_geography_bias_stable,
        "feat_geography_bias_sensitive": feat_geography_bias_sensitive,
        "feat_geography_momentum_stable": feat_geography_momentum_stable,
        "feat_geography_momentum_sensitive": feat_geography_momentum_sensitive,
    }
