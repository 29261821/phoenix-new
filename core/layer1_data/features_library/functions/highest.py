# ==============================================================================
# § 公式 | 週期最高價 (Highest Value)
# ==============================================================================
import polars as pl


def calculate(series: pl.Expr, period: int, **kwargs) -> pl.Expr:
    """
    計算 N 週期內的最高值。

    契約：
    - series: pl.Expr, 一個 Polars 表達式，代表要計算的序列。
    - period: int, 週期。

    返回：
    一個 Polars 表達式，代表週期最高值序列。
    """
    if period <= 0:
        raise ValueError("Highest 的週期 (period) 必須是正整數。")
    return series.rolling_max(window_size=period)
