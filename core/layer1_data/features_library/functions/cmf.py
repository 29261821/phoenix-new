# ==============================================================================
# § 公式 | 蔡金資金流 (Chaikin Money Flow)
# ==============================================================================
import polars as pl

from .sum import calculate as rolling_sum


def calculate(df: pl.DataFrame, period: int, **kwargs) -> pl.Expr:
    """
    計算蔡金資金流 (Chaikin Money Flow, CMF)。

    契約：
    - df 必須包含 'high', 'low', 'close', 'volume' 欄位。
    - period: int, 計算資金流的週期。
    """
    if period <= 0:
        raise ValueError("CMF 的週期 (period) 必須是正整數。")

    h, l, c, v = pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")

    epsilon = 1e-9
    # 修改處：加入 fill_null(0) 防止因 high==low 產生的 NaN 擴散
    money_flow_multiplier = (((c - l) - (h - c)) / (h - l + epsilon)).fill_null(0)
    money_flow_volume = money_flow_multiplier * v

    # 修改處：對分母與結果進行防護，並 clip 在 [-1.0, 1.0] 之間
    v_sum = rolling_sum(series=v, period=period).fill_null(0)
    cmf = rolling_sum(series=money_flow_volume, period=period).fill_null(0) / (
        v_sum + epsilon
    )

    return cmf.clip(-1.0, 1.0)
