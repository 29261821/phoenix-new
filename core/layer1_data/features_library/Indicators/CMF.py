# ==============================================================================
# § 指標 | 蔡金資金流 (Chaikin Money Flow)
# 核心職責: 綜合價格區間與成交量，衡量特定週期內市場資金的淨流入(吸籌)與淨流出(派發)。
# ==============================================================================
"""
【YAML 參數合約清單】
| 參數名稱        | 層級  | 類型 | 建議範圍 | 預設值/Fallback | 說明 |
|---------------|-------|------|----------|-----------------|------|
| period        | H & G | int  | 10 ~ 40  | 無 (必填)       | CMF 的基礎計算與滾動加總週期 |
| adapt_macro_p | G 專用| int  | 21 ~ 55  | period 參數的值 | 用於 Bias (乖離) 計算時的長期中樞 EMA 週期 |
| adapt_micro_p | G 專用| int  | 3 ~ 10   | period 參數的值 | 用於 Momentum (動量) 計算時的短期 EMA 平滑週期，隔離共線性 |
| adapt_vol_p   | G 專用| int  | 20 ~ 55  | period 參數的值 | 用於 Volatility (籌碼混沌度) 計算的滾動標準差週期 |

【特徵工程說明】
- CMF 天生已是 [-1.0, 1.0] 的無量綱指標，代表資金淨流入/流出的強度。
- 透過 adapt_macro_p 決定模型觀察資金流背離 (Bias) 的歷史長度。
- 透過 adapt_micro_p 決定模型對主力資金突然湧入/撤退 (Momentum) 的敏感度。
"""
from typing import Dict

import polars as pl

# 遵照 DSL v6.0 (邏輯自治版) 的設計思想，此指標為「自產自銷」。
# 它不依賴 functions/cmf.py，而是將邏輯內化，以確保指標的獨立性和可讀性。
from src.features.functions.sum import calculate as rolling_sum


def calculate(df: pl.DataFrame, params: Dict, **kwargs) -> Dict[str, pl.Expr]:
    """
    【H 接口：人類與策略庫語意】
    計算蔡金資金流 (Chaikin Money Flow)。
    保留原始的資金流比例數值 (理論範圍 [-1, 1])。
    確保舊有量化腳本可直接利用 CMF > 0 或頂底背離等絕對門檻進行濾網判斷。

    契約：
    - df 必須包含 'high', 'low', 'close', 'volume' 欄位。
    - params 必須包含 'period' 鍵。
    """
    period = params["period"]

    h, l, c, v = pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")

    epsilon = 1e-9
    # money_flow_multiplier 的計算公式與 function 版本略有不同，嚴格遵循 CMF.pl_ind
    money_flow_multiplier = ((c - l) - (h - c)) / (h - l + epsilon)
    money_flow_volume = money_flow_multiplier * v

    cmf_value = rolling_sum(series=money_flow_volume, period=period) / (
        rolling_sum(series=v, period=period) + epsilon
    )

    return {"type": "scalar", "values": {"value": cmf_value}}


def adapt_CMF(h_output: Dict, close: pl.Expr, params: Dict) -> Dict[str, pl.Expr]:
    """
    【G 接口：深度學習寬表特徵】
    將原本的 CMF 資金流數值進行高階時序特徵萃取，轉化為 DL/ML 寬表特徵。
    正交分解為：絕對水位 (Position)、宏觀乖離 (Bias)、加速度 (Momentum) 與 籌碼混沌度 (Volatility)。
    包含 Stable (嚴格約束、適合神經網路) 與 Sensitive (放寬約束、捕捉恐慌/狂熱) 雙版本。

    [架構升級] 採用參數解耦與智慧 Fallback 系統，所有滾動週期全面可由 YAML 配置。
    """
    cmf = h_output["values"]["value"]

    # 1. 提取基礎參數
    period = params["period"]

    # 2. 提取 Adapter 專用參數 (智慧 Fallback 機制)
    adapt_macro_p = params.get("adapt_macro_p", period)
    adapt_micro_p = params.get("adapt_micro_p", period)
    adapt_vol_p = params.get("adapt_vol_p", period)

    # 數值穩定性防護常數：杜絕除零導致的 Inf/NaN
    epsilon = 1e-6

    # ---------------------------------------------------------
    # (A) Position (資金流絕對水位): 當前市場的吸籌/派發總量比例
    # 語意補值: 0.0 (代表資金多空平衡，無淨流入/流出)
    # ---------------------------------------------------------
    # Stable 版：約束於 [-0.5, 0.5]，過濾極端單邊爆量，穩定 Transformer 注意力
    feat_cmf_position_stable = (
        cmf.fill_nan(0.0).fill_null(0.0).clip(-0.5, 0.5).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-1.0, 1.0] 理論極限，允許捕捉史詩級的籌碼清洗
    feat_cmf_position_sensitive = (
        cmf.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (B) Bias (資金流宏觀乖離): CMF 相對於其長線均線的背離
    # 語意補值: 0.0 (當前資金流力道與歷史均值一致)
    # ---------------------------------------------------------
    cmf_ema_macro = cmf.ewm_mean(span=adapt_macro_p, ignore_nulls=True)
    bias = cmf - cmf_ema_macro

    # Stable 版：約束於 [-0.2, 0.2]，關注資金流的微觀動態背離
    feat_cmf_bias_stable = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.2, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [-0.8, 0.8]，捕捉資金流突然斷層式反轉的極端背離信號
    feat_cmf_bias_sensitive = (
        bias.fill_nan(0.0).fill_null(0.0).clip(-0.8, 0.8).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (C) Momentum (資金流加速度): 買盤或賣盤湧入的加速度 (一階導數正規化)
    # 語意補值: 0.0 (資金流速度保持等速，無加速/減速)
    # 降共線性處理: 減去短線 EMA 並進行自適應標準化
    # ---------------------------------------------------------
    cmf_ema_micro = cmf.ewm_mean(span=adapt_micro_p, ignore_nulls=True)
    momentum = (cmf - cmf_ema_micro) / (cmf_ema_micro.abs() + epsilon)

    # Stable 版：嚴格約束 [-1.0, 1.0] 的正規化震盪空間
    feat_cmf_momentum_stable = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0).cast(pl.Float64)
    )
    # Sensitive 版：放寬約束至 [-5.0, 5.0]，凸顯主力資金瞬間倒貨或瘋搶的爆發力
    feat_cmf_momentum_sensitive = (
        momentum.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).cast(pl.Float64)
    )

    # ---------------------------------------------------------
    # (D) Volatility (籌碼混沌度): CMF 自身的歷史變異數
    # 語意補值: 0.0 (資金流入/流出極度平穩，呈現單邊暗盤交易)
    # 防禦處理: 強制套用 log1p 平滑多空激烈交戰時產生的變異數爆炸
    # ---------------------------------------------------------
    cmf_volatility = cmf.rolling_std(window_size=adapt_vol_p)
    log_cmf_vol = cmf_volatility.log1p()

    # Stable 版：約束於 [0.0, 0.2]
    feat_cmf_volatility_stable = (
        log_cmf_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.2).cast(pl.Float64)
    )
    # Sensitive 版：約束於 [0.0, 0.5]，保留高頻換手導致的籌碼極度不穩定狀態
    feat_cmf_volatility_sensitive = (
        log_cmf_vol.fill_nan(0.0).fill_null(0.0).clip(0.0, 0.5).cast(pl.Float64)
    )

    return {
        "feat_cmf_position_stable": feat_cmf_position_stable,
        "feat_cmf_position_sensitive": feat_cmf_position_sensitive,
        "feat_cmf_bias_stable": feat_cmf_bias_stable,
        "feat_cmf_bias_sensitive": feat_cmf_bias_sensitive,
        "feat_cmf_momentum_stable": feat_cmf_momentum_stable,
        "feat_cmf_momentum_sensitive": feat_cmf_momentum_sensitive,
        "feat_cmf_volatility_stable": feat_cmf_volatility_stable,
        "feat_cmf_volatility_sensitive": feat_cmf_volatility_sensitive,
    }
