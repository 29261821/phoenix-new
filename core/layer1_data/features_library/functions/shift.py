# ==============================================================================
# § 公式 | 移位 (Shift / Prev)
# ==============================================================================
import polars as pl


def calculate(series: pl.Expr, period: int, **kwargs) -> pl.Expr:
    """
    對序列進行向前移位（獲取先前的值）。

    契約：
    - series: pl.Expr, 一個 Polars 表達式。
    - period: int, 向前看的週期數。

    返回：
    一個 Polars 表達式，代表移位後的序列。
    """
    if period < 0:
        raise ValueError("Shift 的週期 (period) 必須是非負整數。")
    return series.shift(period)
