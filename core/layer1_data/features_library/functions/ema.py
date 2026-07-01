# ==============================================================================
# § 公式 | 指數移動平均 (Exponential Moving Average)
# ==============================================================================
import polars as pl


def calculate(series: pl.Expr, length: int, **kwargs) -> pl.Expr:
    """
    計算指數移動平均 (EMA)。

    契約：
    - series: pl.Expr, 一個 Polars 表達式，代表要計算的序列。
    - length: int, 移動平均的週期 (span)。

    返回：
    一個 Polars 表達式，代表 EMA 序列。
    """
    if length <= 0:
        raise ValueError("EMA 的週期 (length) 必須是正整數。")
    return series.ewm_mean(span=length, adjust=False).fill_nan(0)
