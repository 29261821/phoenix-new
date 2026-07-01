# ==============================================================================
# § 指標 | 卡爾曼濾波器引擎 (Kalman Filter Engine)
# 核心職責: 根據【第三邊：統計套利】作戰計畫，實現狀態空間動量模型，過濾序列雜訊。
# v3.0 更新: [架構升級] H 接口封裝為 pl.Expr 確保契約一致，新增極致無尺度 G 接口。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱          | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|-----------------|-------|------|----------|-----------------|------|
| input_feature   | H & G | str  | -        | 無 (必填)       | 要進行降噪濾波的目標欄位 (如 'close', 'RSI') |
| process_noise   | H & G | float| 1e-5~1e-2| 無 (必填)       | 過程噪音協方差 (Q)，決定對真實變化的信任度 |
| measurement_noise| H & G| float| 1e-3~1.0 | 無 (必填)       | 測量噪音協方差 (R)，決定對觀測雜訊的抗性 |
| adapt_macro_p   | G 專用| int  | 34 ~ 89  | 55              | 用於 Position (濾波 Z-Score) 的歷史觀測週期 |
| adapt_micro_p   | G 專用| int  | 3 ~ 10   | 5               | 用於 Momentum (濾波加速度) 的短線 EMA 週期 |
| adapt_vol_p     | G 專用| int  | 21 ~ 55  | 34              | 用於 Volatility (濾波混沌度) 的滾動標準差週期 |

【特徵工程說明】
- Kalman Filter 可應用於任何特徵(包含振盪器或絕對價格)，因此 G 接口必須「絕對無尺度化」。
- G 接口不依賴外部 close，而是直接對 Filtered 序列計算其自身的動態 Z-Score、發散乖離與變動率。
"""
from typing import Any, Dict

import numpy as np
import polars as pl


def calculate(df: pl.DataFrame, params: Dict[str, Any], **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    對指定的單一特徵序列應用一維卡爾曼濾波器進行降噪。
    本指標屬於「 eager (non-expression-based) 」類型，因其遞歸和有狀態特性，
    必須在 Python/NumPy 層完成計算。

    [契約修復]: 為了與【惰性閃擊軍團】完全相容，產出的 pl.Series 會透過 pl.lit()
    包裝為 pl.Expr，確保後續特徵工程能無縫銜接。
    """
    # --- 1. 契約驗證與參數提取 ---
    feature_col: str = params.get("input_feature")
    q: float = params.get("process_noise")
    r: float = params.get("measurement_noise")

    if not feature_col or q is None or r is None:
        raise ValueError(
            "Kalman_Filter_Engine 的參數 'input_feature', 'process_noise', 'measurement_noise' 必須被提供。"
        )

    if feature_col not in df.columns:
        raise ValueError(f"DataFrame 中缺少卡爾曼濾波器所需的欄位: {feature_col}")

    # --- 2. 數據準備 ---
    # 使用前向填充處理缺失值，確保數據連續性
    input_series = df.get_column(feature_col).forward_fill()

    # 處理完全為空的 Series 的邊界情況
    if input_series.is_null().all():
        return {
            "type": "scalar",
            "values": {"Filtered": pl.lit(input_series.alias("Filtered"))},
        }

    input_values = input_series.to_numpy()

    # --- 3. 核心卡爾曼濾波器計算 ---
    n = len(input_values)
    x_hat = np.zeros(n)  # 後驗狀態估計 A posteriori state estimate
    p = np.zeros(n)  # 後驗誤差協方差 A posteriori estimate error covariance

    # 初始化第一個值
    x_hat[0] = input_values[0]
    p[0] = 1.0

    for k in range(1, n):
        # --- 預測步驟 ---
        # 狀態預測 (我們假設狀態是隨機遊走的，所以預測等於上一個狀態)
        x_hat_minus = x_hat[k - 1]
        # 誤差協方差預測
        p_minus = p[k - 1] + q

        # --- 更新步驟 ---
        # 卡爾曼增益
        kalman_gain = p_minus / (p_minus + r)
        # 更新狀態估計
        x_hat[k] = x_hat_minus + kalman_gain * (input_values[k] - x_hat_minus)
        # 更新誤差協方差
        p[k] = (1 - kalman_gain) * p_minus

    # --- 4. 格式化輸出 ---
    # 將 NumPy array 轉為 Series，並透過 pl.lit() 偽裝成 Expr 滿足惰性合約
    filtered_series = pl.Series(name="Filtered", values=x_hat)

    return {"type": "scalar", "values": {"Filtered": pl.lit(filtered_series)}}


def adapt_Kalman_Filter(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    由於 Kalman Filter 可作用於任意尺度特徵 (如價格 60000 或 RSI 50)，
    特徵工程採用「絕對無尺度 (Scale-invariant)」的自身正交分解。
    正交分解為：濾波水位 (Position)、濾波乖離 (Bias)、濾波加速度 (Momentum)、濾波混沌度 (Volatility)。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，動能與變異數週期全面可由 YAML 配置。
    """
    filtered = h_output["values"]["Filtered"].cast(pl.Float64)

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    # 由於目標欄位不可預知，賦予通用的穩健預設常數
    adapt_macro_p = params.get("adapt_macro_p", 55)
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", 34)

    # 數值穩定性防護常數
    epsilon = 1e-6

    # 計算濾波特徵的宏觀歷史均值與標準差
    kf_mean = filtered.rolling_mean(window_size=adapt_macro_p)
    kf_std = filtered.rolling_std(window_size=adapt_macro_p)

    # ---------------------------------------------------------
    # (A) Position (濾波信號歷史水位): 濾波後信號的滾動 Z-Score
    # 語意補值: 0.0 (濾波信號處於歷史常態平均)
    # ---------------------------------------------------------
    z_score = (filtered - kf_mean) / (kf_std + epsilon)

    # Stable 版：嚴格約束 [-3.0, 3.0]
    feat_kalman_position_stable = (
        z_score.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-5.0, 5.0]，捕捉 5 個標準差外的極端異動
    feat_kalman_position_sensitive = (
        z_score.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (濾波信號發散乖離): 濾波信號相對於其長期均線的無量綱偏離率
    # 語意補值: 0.0 (信號發展平穩，無極端偏離)
    # ---------------------------------------------------------
    bias = (filtered - kf_mean) / (kf_mean.abs() + epsilon)

    # Stable 版：約束於 [-0.2, 0.2]
    feat_kalman_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.5, 0.5]
    feat_kalman_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (濾波信號加速度): 濾波信號的變化率
    # 語意補值: 0.0 (信號維持等速，無突發加速度)
    # 降共線性處理: 減去短線 EMA 並自適應標準化
    # ---------------------------------------------------------
    kf_ema_micro = filtered.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (filtered - kf_ema_micro) / (kf_ema_micro.abs() + epsilon)

    # Stable 版：約束於 [-1.0, 1.0] 的正規化震盪空間
    feat_kalman_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]
    feat_kalman_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (濾波信號混沌度): 濾波信號的相對標準差 (變異係數 CV)
    # 語意補值: 0.0 (信號波動極小)
    # 防禦處理: 強制套用 log1p 平滑
    # ---------------------------------------------------------
    volatility = kf_std / (kf_mean.abs() + epsilon)
    log_vol = volatility.log1p()

    # Stable 版：約束於 [0.0, 0.2]
    feat_kalman_volatility_stable = (
        log_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.5]
    feat_kalman_volatility_sensitive = (
        log_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )

    return {
        "feat_kalman_position_stable": feat_kalman_position_stable,
        "feat_kalman_position_sensitive": feat_kalman_position_sensitive,
        "feat_kalman_bias_stable": feat_kalman_bias_stable,
        "feat_kalman_bias_sensitive": feat_kalman_bias_sensitive,
        "feat_kalman_momentum_stable": feat_kalman_momentum_stable,
        "feat_kalman_momentum_sensitive": feat_kalman_momentum_sensitive,
        "feat_kalman_volatility_stable": feat_kalman_volatility_stable,
        "feat_kalman_volatility_sensitive": feat_kalman_volatility_sensitive,
    }
