# ==============================================================================
# § 公式 | 絕對值 (Absolute Value)
# ==============================================================================
import polars as pl


def calculate(series: pl.Expr, **kwargs) -> pl.Expr:
    """
    計算序列的絕對值。

    契約：
    - series: pl.Expr, 一個 Polars 表達式。

    返回：
    一個 Polars 表達式，代表絕對值序列。
    """
    return series.abs()
