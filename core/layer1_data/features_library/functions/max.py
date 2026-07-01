# ==============================================================================
# § 公式 | 水平最大值 (Horizontal Max)
# ==============================================================================
import polars as pl


def calculate(*series: pl.Expr, **kwargs) -> pl.Expr:
    """
    計算多個序列在每個時間點上的最大值。

    契約：
    - *series: pl.Expr, 一個或多個 Polars 表達式。

    返回：
    一個 Polars 表達式，代表水平最大值序列。
    """
    if not series:
        raise ValueError("Max 函數至少需要一個序列作為輸入。")
    return pl.max_horizontal(series)
