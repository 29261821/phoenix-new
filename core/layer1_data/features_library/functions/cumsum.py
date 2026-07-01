# ==============================================================================
# § 公式 | 累積求和 (Cumulative Sum)
# ==============================================================================
import polars as pl


def calculate(series: pl.Expr, **kwargs) -> pl.Expr:
    """
    計算序列的累積求和。

    契約：
    - series: pl.Expr, 一個 Polars 表達式。

    返回：
    一個 Polars 表達式，代表累積求和序列。
    """
    return series.fill_null(0).fill_nan(0).cum_sum()
