# ==============================================================================
# § 指標 | 壓力引擎 (TTM Squeeze) v3.0
# 核心職責: 根據【第四邊：系統動力學】作戰計畫，實現波動率壓縮與爆發指標。
# v3.0 更新: [架構升級] 導入 H 接口合約標準與 G 接口降維連續化特徵工程。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型  | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|-------|----------|-----------------|------|
| bb_period     | H & G | int   | 10 ~ 50  | 無 (必填)       | 布林帶 SMA 計算週期 |
| bb_std        | H     | float | 1.5 ~ 3.0| 無 (必填)       | 布林帶標準差倍數 |
| kc_period     | H     | int   | 10 ~ 50  | 無 (必填)       | 肯特納通道 EMA 計算週期 |
| atr_period    | H     | int   | 10 ~ 30  | 無 (必填)       | 肯特納通道 ATR 計算週期 |
| adapt_macro_p | G 專用| int   | 21 ~ 55  | bb_period 參數值| 用於 Position (壓縮政權水位) 的長線衰減週期 |
| adapt_micro_p | G 專用| int   | 3 ~ 10   | 5               | 用於 Momentum (壓縮釋放加速度) 的短線 EMA 週期 |
| adapt_vol_p   | G 專用| int   | 13 ~ 34  | kc_period 參數值| 用於 Volatility (政權混沌度) 的滾動標準差週期 |

【特徵工程說明】
- Squeeze 的原始輸出只有二元狀態 (0 關閉 / 1 開啟)，神經網路難以感知壓縮的「深度」。
- H 接口額外輸出了 BBW (布林頻寬) 與 KCW (肯特納頻寬)，供 G 接口還原連續性。
- G 接口將其正交分解為：壓縮政權機率 (Position)、真實壓縮深度 (Bias)、壓縮釋放加速度 (Momentum) 與狀態混沌度 (Volatility)。
"""
from typing import Dict

import polars as pl

from src.features.functions.atr import calculate as atr
from src.features.functions.ema import calculate as ema
from src.features.functions.sma import calculate as sma
from src.features.functions.stddev import calculate as stddev


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    保留核心的 0/1 狀態，並額外導出 BBW 與 KCW 頻寬。
    確保舊有策略能直接使用 Squeeze==1 作為動能積蓄濾網，同時為 G 接口提供連續化素材。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - params 必須包含 'bb_period', 'bb_std', 'kc_period', 'atr_period' 鍵。
    """
    # --- 1. 契約驗證與參數提取 ---
    bb_period: int = params.get("bb_period")
    bb_std: float = params.get("bb_std")
    kc_period: int = params.get("kc_period")
    atr_period: int = params.get("atr_period")

    if not all([bb_period, bb_std, kc_period, atr_period]):
        raise ValueError(
            "Pressure_Engine 的參數 'bb_period', 'bb_std', 'kc_period', 'atr_period' 必須被提供。"
        )

    epsilon = 1e-9
    h, l, c = pl.col("high"), pl.col("low"), pl.col("close")

    # --- 2. 計算布林帶 (Bollinger Bands) ---
    bb_middle = sma(series=c, length=bb_period)
    bb_stdev = stddev(series=c, period=bb_period)
    bb_upper = bb_middle + bb_stdev * bb_std
    bb_lower = bb_middle - bb_stdev * bb_std

    # 計算布林頻寬 (Bollinger Band Width)
    bbw = (bb_upper - bb_lower) / (bb_middle + epsilon)

    # --- 3. 計算肯特納通道 (Keltner Channels) ---
    typical_price = (h + l + c) / 3
    kc_middle = ema(series=typical_price, length=kc_period)
    atr_val = atr(df=df, period=atr_period)

    kc_atr_mult = 1.5
    kc_upper = kc_middle + (atr_val * kc_atr_mult)
    kc_lower = kc_middle - (atr_val * kc_atr_mult)

    # 計算肯特納頻寬 (Keltner Channel Width)
    kcw = (kc_upper - kc_lower) / (kc_middle + epsilon)

    # --- 4. 核心 Squeeze 判斷邏輯 ---
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    return {
        "type": "vector",
        "values": {"Squeeze": squeeze_on.cast(pl.Int8), "BBW": bbw, "KCW": kcw},
    }


def adapt_Pressure(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將單純的 0/1 壓縮狀態，還原為具有物理學「彈簧深度」意義的連續特徵。
    正交分解為：壓縮政權機率 (Position)、真實壓縮深度 (Bias)、壓縮釋放加速度 (Momentum)、狀態混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束) 雙版本。
    """
    squeeze = h_output["values"]["Squeeze"].cast(pl.Float64)
    bbw = h_output["values"]["BBW"]
    kcw = h_output["values"]["KCW"]

    # 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", params["bb_period"])
    adapt_micro_p = params.get("adapt_micro_p", 5)
    adapt_vol_p = params.get("adapt_vol_p", params["kc_period"])

    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (壓縮政權機率): 市場長期處於 Squeeze 狀態的衰減中樞
    # 語意補值: 0.0 (市場處於長期的釋放擴張期，毫無壓縮跡象)
    # ---------------------------------------------------------
    regime = squeeze.ewm_mean(span=adapt_macro_p, ignore_nulls=True)

    # Stable & Sensitive 版：先天約束於 [0.0, 1.0] 區間
    feat_pressure_position_stable = (
        regime.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )
    feat_pressure_position_sensitive = (
        regime.fill_nan(0.0).fill_null(0.0).clip(0.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (真實壓縮深度 / Squeeze Delta): 布林頻寬相對於肯特納頻寬的偏離
    # 語意補值: 0.0 (兩者寬度相等，處於臨界點)
    # 當數值為負，代表布林帶鑽入肯特納內部，壓縮越深數值越負；數值為正代表波動釋放。
    # ---------------------------------------------------------
    squeeze_delta = (bbw - kcw) / (kcw + epsilon)

    # Stable 版：約束於 [-0.5, 0.5]，專注於常規的能量壓縮與初步釋放
    feat_pressure_bias_stable = (
        squeeze_delta.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 2.0]，捕捉極致壓縮 (-1.0) 以及史詩級爆發導致頻寬暴增兩倍的極端異動
    feat_pressure_bias_sensitive = (
        squeeze_delta.fill_nan(0.0).fill_null(0.0).clip(-1.0, 2.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (壓縮釋放加速度): 壓縮深度 (Bias) 的變化速度
    # 語意補值: 0.0 (波動率維持現狀)
    # 降共線性處理: 減去自身的 EMA 並自適應標準化，極致捕捉「彈簧剛鬆開瞬間」的爆發力
    # ---------------------------------------------------------
    ema_delta = squeeze_delta.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (squeeze_delta - ema_delta) / (ema_delta.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_pressure_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯變盤瞬間強大的波動率引爆動能
    feat_pressure_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (政權切換混沌度): Squeeze 狀態 (0/1) 的歷史變異數
    # 語意補值: 0.0 (市場死心塌地維持壓縮或擴張)
    # 若數值飆高，代表市場處於「假突破」邊緣，頻繁進出壓縮狀態
    # ---------------------------------------------------------
    state_volatility = squeeze.rolling_std(window_size=adapt_vol_p)

    # Stable & Sensitive 版：0/1 序列的標準差理論極限為 0.5
    feat_pressure_volatility_stable = (
        state_volatility.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )
    feat_pressure_volatility_sensitive = (
        state_volatility.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )

    return {
        "feat_pressure_position_stable": feat_pressure_position_stable,
        "feat_pressure_position_sensitive": feat_pressure_position_sensitive,
        "feat_pressure_bias_stable": feat_pressure_bias_stable,
        "feat_pressure_bias_sensitive": feat_pressure_bias_sensitive,
        "feat_pressure_momentum_stable": feat_pressure_momentum_stable,
        "feat_pressure_momentum_sensitive": feat_pressure_momentum_sensitive,
        "feat_pressure_volatility_stable": feat_pressure_volatility_stable,
        "feat_pressure_volatility_sensitive": feat_pressure_volatility_sensitive,
    }
