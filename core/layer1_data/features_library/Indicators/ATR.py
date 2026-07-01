# ==============================================================================
# § 指標 | 平均真實波幅 (Average True Range)
# 核心職責: 衡量市場的絕對波動點數，反映市場真實的交投熱度與風險邊界。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| Len           | H & G | int  | 7 ~ 21   | 無 (必填)       | ATR 的基礎計算週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | Len 參數的值    | 用於計算 Position (Z-Score) 的歷史觀測期 |
| adapt_micro_p | G 專用| int  | 5 ~ 14   | Len 參數的值    | 用於計算 Momentum (動能) 時的 EMA 平滑週期 |

【特徵工程說明】
- 原始 ATR 是絕對價格點數，G 接口先將其除以 Close 轉為百分比。
- 透過 adapt_macro_p 觀察波動率的長期分位狀態 (如 34 或 55 期)。
- 透過 adapt_micro_p 捕捉波動率擴張的瞬間加速度，隔離共線性。
"""
from typing import Dict

import polars as pl

from src.features.functions.atr import calculate as atr_func


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算平均真實波幅 (Average True Range)。
    保留原始絕對價格尺度的波動點數，不做無量綱化。
    確保舊有策略能直接使用此絕對數值進行動態止損 (Trailing Stop) 與部位控管 (Position Sizing)。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含 'Len' 鍵。
    """
    period = params["Len"]
    atr_val = atr_func(df=df, period=period)

    return {"type": "scalar", "values": {"ATR": atr_val}}


def adapt_ATR(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將絕對波動點數 (ATR) 轉換為供 DL/ML 使用的無尺度、穩定特徵。
    將單一波動率正交分解為：常態波動 (Volatility)、歷史分位 (Position)、乖離 (Bias)、擴張動能 (Momentum)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉極端行情) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，所有滾動週期全面可由 YAML 配置。
    """
    atr_val = h_output["values"]["ATR"]

    # 1. 提取基礎參數
    period = params["Len"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", period)
    adapt_micro_p = params.get("adapt_micro_p", period)

    # 防禦性常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # 首先將絕對 ATR 無量綱化，轉為「百分比真實波幅 (Normalized ATR)」
    norm_atr = atr_val / (close + epsilon)

    # ---------------------------------------------------------
    # (A) Volatility (常態波動特徵): 每日平均波幅佔股價的百分比
    # 語意補值: 0.0 (無波動)
    # 防禦處理: 強制套用 log1p 壓抑極端波動造成的長尾分佈
    # ---------------------------------------------------------
    log_norm_atr = norm_atr.log1p()

    # Stable 版：約束於 [0.0, 0.1]，代表最多只關注 10% 以內的常規波動
    feat_atr_norm_stable = (
        log_norm_atr.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.1).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.5]，允許捕捉如 Crypto 閃崩時 50% 的單日極端波動
    feat_atr_norm_sensitive = (
        log_norm_atr.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Position (波動率歷史分位): 百分比波幅的滾動 Z-Score
    # 語意補值: 0.0 (處於歷史常態均值)
    # ---------------------------------------------------------
    atr_rolling_mean = norm_atr.rolling_mean(window_size=adapt_macro_p)
    atr_rolling_std = norm_atr.rolling_std(window_size=adapt_macro_p)
    z_score = (norm_atr - atr_rolling_mean) / (atr_rolling_std + epsilon)

    # Stable 版：嚴格約束 [-3.0, 3.0] 的常態分佈範圍
    feat_atr_position_stable = (
        z_score.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-5.0, 5.0]，捕捉暴風雨前夕的極度收斂或恐慌拋售的極端擴張
    feat_atr_position_sensitive = (
        z_score.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Bias (波動率乖離): 當前波動率相對於長期均線的擴張倍數
    # 語意補值: 0.0 (完美貼合長期均值)
    # ---------------------------------------------------------
    bias = (norm_atr / (atr_rolling_mean + epsilon)) - 1.0

    # Stable 版：約束於 [-0.5, 1.0] (波動收斂至多 50%，擴張至多 100%)
    feat_atr_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.8, 3.0] (容許觀察到波動率瞬間放大 3 倍的極端乖離)
    feat_atr_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.8, 3.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Momentum (波動率動能): 波動率擴張的加速度 (一階導數)
    # 語意補值: 0.0 (波動率無加速現象)
    # 降共線性處理: 減去自身的 EMA 並進行自適應標準化
    # ---------------------------------------------------------
    ema_norm_atr = norm_atr.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (norm_atr - ema_norm_atr) / (ema_norm_atr.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_atr_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束，捕捉波動率的瞬間核彈級爆發
    feat_atr_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    return {
        "feat_atr_norm_stable": feat_atr_norm_stable,
        "feat_atr_norm_sensitive": feat_atr_norm_sensitive,
        "feat_atr_position_stable": feat_atr_position_stable,
        "feat_atr_position_sensitive": feat_atr_position_sensitive,
        "feat_atr_bias_stable": feat_atr_bias_stable,
        "feat_atr_bias_sensitive": feat_atr_bias_sensitive,
        "feat_atr_momentum_stable": feat_atr_momentum_stable,
        "feat_atr_momentum_sensitive": feat_atr_momentum_sensitive,
    }
