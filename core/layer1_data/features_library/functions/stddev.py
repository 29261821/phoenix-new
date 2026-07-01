# ==============================================================================
# § 公式 | 滾動標準差 (Rolling Standard Deviation)
# ==============================================================================
import polars as pl


def calculate(series: pl.Expr, period: int, **kwargs) -> pl.Expr:
    """
    計算滾動標準差。

    契約：
    - series: pl.Expr, 一個 Polars 表達式，代表要計算的序列。
    - period: int, 週期。

    返回：
    一個 Polars 表達式，代表滾動標準差序列。
    """
    if period <= 1:
        raise ValueError("StdDev 的週期 (period) 必須大於 1。")
    return series.rolling_std(window_size=period).fill_nan(0).fill_null(0)
