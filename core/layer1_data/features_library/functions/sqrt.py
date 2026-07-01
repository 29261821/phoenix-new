# ==============================================================================
# § 公式 | 平方根 (Square Root)
# ==============================================================================
import polars as pl

# src/features/functions/sqrt.py


def calculate(series: pl.Expr, **kwargs) -> pl.Expr:
    # 修改處：加入 fill_nan 防護，確保 SVD 矩陣完整
    return series.abs().sqrt().fill_nan(0).fill_null(0)
