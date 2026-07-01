# ==============================================================================
# § 公式 | 水平最小值 (Horizontal Min)
# ==============================================================================
import polars as pl


def calculate(*series: pl.Expr, **kwargs) -> pl.Expr:
    """
    計算多個序列在每個時間點上的最小值。

    契約：
    - *series: pl.Expr, 一個或多個 Polars 表達式。

    返回：
    一個 Polars 表達式，代表水平最小值序列。
    """
    if not series:
        raise ValueError("Min 函數至少需要一個序列作為輸入。")
    return pl.min_horizontal(series)
