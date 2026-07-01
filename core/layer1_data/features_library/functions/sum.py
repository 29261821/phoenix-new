# ==============================================================================
# § 公式 | 滾動求和 (Rolling Sum)
# ==============================================================================
import polars as pl


def calculate(series: pl.Expr, period: int, **kwargs) -> pl.Expr:
    """
    計算滾動求和。

    契約：
    - series: pl.Expr, 一個 Polars 表達式，代表要計算的序列。
    - period: int, 週期。

    返回：
    一個 Polars 表達式，代表滾動求和序列。
    """
    if period <= 0:
        raise ValueError("Sum 的週期 (period) 必須是正整數。")
    return series.rolling_sum(window_size=period).fill_nan(0).fill_null(0)
