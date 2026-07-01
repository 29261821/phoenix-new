# ==============================================================================
# § 公式 | 冪運算 (Power)
# ==============================================================================
import polars as pl

# src/features/functions/pow.py


def calculate(base: pl.Expr, exponent: pl.Expr | float, **kwargs) -> pl.Expr:
    # 1. 負數底數與分數指數防護
    if isinstance(exponent, (int, float)) and exponent < 1:
        res = base.abs().pow(exponent)
    else:
        res = base.pow(exponent)

    # 2. 修改處：防止計算結果產生 Inf 或 NaN 擴散
    # 使用一個極大的安全閾值 (例如 1e10) 並填充空值
    return res.fill_nan(0).fill_null(0).clip(lower_bound=-1e10, upper_bound=1e10)
