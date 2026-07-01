# ==============================================================================
# § 指標 | 資金流上下文 (Flow Context)
# 版本: v3.0 (150分典範版)
# 核心職責: 將牛頓物理學定律應用於 OBV，提取資金流的慣性、動量與加速度。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| inertia_p     | H & G | int  | 20 ~ 50  | 無 (必填)       | 資金流慣性 (慢線) 的 EMA 週期 |
| momentum_p    | H & G | int  | 5 ~ 20   | 無 (必填)       | 資金流動量 (快線) 的 EMA 週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | inertia_p 參數值| 用於 Bias (宏觀乖離) 計算的長線 EMA 週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 13   | momentum_p 參數值| 用於 Momentum (加速度) 計算的短線 EMA 週期，隔離共線性 |
| adapt_vol_p   | G 專用| int  | 21 ~ 55  | inertia_p 參數值| 用於 Volatility (籌碼混沌度) 的滾動標準差週期 |

【特徵工程說明】
- Flow Context 將絕對成交量加速度轉換為無尺度的「淨壓力比率」。
- 透過 adapt_macro_p 觀察淨資金流壓力相對於長線均值的背離，捕捉資金池底層背離。
- 透過 adapt_vol_p 衡量籌碼混亂程度，識別暗盤吸籌或激烈互砸的絞肉機行情。
"""
from typing import Any, Dict

import numpy as np
import polars as pl

from src.features.functions.ema import calculate as ema


def _calculate_obv_vectorized(
    close_np: np.ndarray, volume_np: np.ndarray
) -> np.ndarray:
    """
    【v3.0 聖杯級重構】
    使用純 NumPy 向量化操作，高效且絕對穩定地計算 OBV。
    此實現取代了舊有的、存在 Numba JIT 編譯風險的迭代迴圈。
    """
    # 1. 計算相鄰收盤價的差值符號 (+1, -1, 0)
    #    使用 np.diff 計算差值，並用 prepend 在開頭補上第一個元素以維持長度一致
    price_diff = np.diff(close_np, prepend=close_np[0])
    price_diff_sign = np.sign(price_diff)

    # 2. 根據符號為成交量賦予方向
    signed_volume = price_diff_sign * volume_np

    # 3. 累加帶符號的成交量，得到 OBV 序列
    #    此操作由 NumPy 底層的高度優化 C 函數執行，性能卓越。
    obv_np = np.cumsum(signed_volume)

    return obv_np


def calculate(df: pl.DataFrame, params: Dict[str, Any], **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算資金流的物理學特徵。
    保留絕對的 OBV 單位尺度 (Inertia, Momentum, Acceleration)。
    為了後續 G 接口的無量綱化特徵轉換，我們在此額外輸出 BaseVolEMA。

    契約：
    - df 必須包含 'close', 'volume' 欄位。
    - params 必須包含 'inertia_p' (慣性週期) 和 'momentum_p' (動量週期) 鍵。
    """
    # --- 1. 契約驗證與參數提取 ---
    inertia_p: int = params.get("inertia_p")
    momentum_p: int = params.get("momentum_p")

    if not all([inertia_p, momentum_p]):
        raise ValueError("Flow_Context 的參數 'inertia_p' 和 'momentum_p' 必須被提供。")
    if momentum_p >= inertia_p:
        raise ValueError(
            "Flow_Context 的契約要求 'momentum_p' (快線) 必須小於 'inertia_p' (慢線)。"
        )

    for col in ["close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"輸入 DataFrame 缺少 Flow_Context 所需的欄位: {col}")

    # --- 2. Eager 部分: 高性能、絕對穩定的 OBV 計算 ---
    try:
        # 嘗試零拷貝以獲得最佳性能
        close_np = df["close"].to_numpy(zero_copy_only=True)
        volume_np = df["volume"].to_numpy(zero_copy_only=True)
    except Exception:
        # 如果內存不連續，則允許複製
        close_np = df["close"].to_numpy()
        volume_np = df["volume"].to_numpy()

    obv_np = _calculate_obv_vectorized(close_np=close_np, volume_np=volume_np)
    obv_series = pl.Series("obv_target", obv_np)

    # --- 3. Lazy 部分: 基於 OBV 序列，進行無狀態表達式構建 ---
    inertia = ema(series=obv_series, length=inertia_p)
    momentum = ema(series=obv_series, length=momentum_p)
    acceleration = momentum - inertia

    # 【追加特徵】為了 G 接口能消除成交量的絕對 Scale 污染，輸出基礎平均成交量
    base_vol_ema = ema(series=pl.col("volume"), length=inertia_p)

    return {
        "type": "vector",
        "values": {
            "Inertia": inertia,
            "Momentum": momentum,
            "Acceleration": acceleration,
            "BaseVolEMA": base_vol_ema,
        },
    }


def adapt_Flow_Context(
    h_output: Dict, close: pl.Expr, params: Dict
) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將具備絕對成交量單位的資金流加速度 (Acceleration)，轉換為無尺度的淨壓力比率。
    正交分解為：絕對水位/淨壓力 (Position)、宏觀乖離 (Bias)、資金衝動/Jerk (Momentum)、籌碼混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉巨鯨異動) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，所有動能、乖離與變異數週期全面可由 YAML 配置。
    """
    acceleration = h_output["values"]["Acceleration"]
    base_vol_ema = h_output["values"]["BaseVolEMA"]

    # 1. 提取基礎參數
    inertia_p = params["inertia_p"]
    momentum_p = params["momentum_p"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", inertia_p)
    adapt_micro_p = params.get("adapt_micro_p", momentum_p)
    adapt_vol_p = params.get("adapt_vol_p", inertia_p)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # 將絕對成交量加速度，除以基礎成交量，得到無量綱的「淨資金流壓力比 (Net Flow Pressure)」
    norm_accel = acceleration / (base_vol_ema + epsilon)

    # ---------------------------------------------------------
    # (A) Position (資金流絕對水位 / 淨壓力): 當前異常淨流入/流出的倍數
    # 語意補值: 0.0 (多空動能平衡，無異常淨流入/流出)
    # ---------------------------------------------------------
    # Stable 版：約束於 [-2.0, 2.0]，即最多關注 2 倍均量的淨資金流，穩定 DL 注意力
    feat_flow_context_position_stable = (
        norm_accel.fill_nan(0.0).fill_null(0.0).clip(-2.0, 2.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬至 [-5.0, 5.0]，捕捉巨鯨拋售或史詩級軋空時的 5 倍均量異動
    feat_flow_context_position_sensitive = (
        norm_accel.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (資金流宏觀乖離): 淨資金流壓力相對於其長線均線的背離
    # 語意補值: 0.0 (資金流入/流出的力道與過去基準一致)
    # ---------------------------------------------------------
    accel_ema_macro = norm_accel.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = norm_accel - accel_ema_macro

    # Stable 版：約束於 [-1.0, 1.0]，專注於常規的資金流衰退或增強背離
    feat_flow_context_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-3.0, 3.0]，捕捉資金流向瞬間斷層反轉的極端信號
    feat_flow_context_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-3.0, 3.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (資金流衝動 / Jerk): 加速度的變化率 (物理學上的衝動)
    # 語意補值: 0.0 (資金流壓力維持等速發展，無突發的加速爆發)
    # 降共線性處理: 減去自身的短線 EMA 並自適應標準化，凸顯瞬間的籌碼瘋搶或踩踏
    # ---------------------------------------------------------
    ema_norm_accel = norm_accel.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (norm_accel - ema_norm_accel) / (ema_norm_accel.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_flow_context_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，極致捕捉主力瞬間突襲的動能峰值
    feat_flow_context_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (資金籌碼混沌度): 資金流壓力的歷史變異數
    # 語意補值: 0.0 (籌碼流動極度穩定，呈現暗盤吸籌/派發)
    # 防禦處理: 強制套用 log1p 平滑多空激烈互砸時產生的極端變異數
    # ---------------------------------------------------------
    flow_volatility = norm_accel.rolling_std(window_size=adapt_vol_p)
    log_flow_vol = flow_volatility.log1p()

    # Stable 版：約束於 [0.0, 1.0]
    feat_flow_context_volatility_stable = (
        log_flow_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 3.0]，保留高換手絞肉機行情下的極度混亂狀態
    feat_flow_context_volatility_sensitive = (
        log_flow_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 3.0).cast(pl.Float64)
    )

    return {
        "feat_flow_context_position_stable": feat_flow_context_position_stable,
        "feat_flow_context_position_sensitive": feat_flow_context_position_sensitive,
        "feat_flow_context_bias_stable": feat_flow_context_bias_stable,
        "feat_flow_context_bias_sensitive": feat_flow_context_bias_sensitive,
        "feat_flow_context_momentum_stable": feat_flow_context_momentum_stable,
        "feat_flow_context_momentum_sensitive": feat_flow_context_momentum_sensitive,
        "feat_flow_context_volatility_stable": feat_flow_context_volatility_stable,
        "feat_flow_context_volatility_sensitive": feat_flow_context_volatility_sensitive,
    }
