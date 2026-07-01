# ==============================================================================
# § 公式 | 威爾德平滑 (Wilder's Smoothing)
# ==============================================================================
import polars as pl


def calculate(series: pl.Expr, period: int, **kwargs) -> pl.Expr:
    """
    計算威爾德平滑 (Wilder's Smoothing)。
    這是一種特殊的 EMA，alpha = 1 / period。

    契約：
    - series: pl.Expr, 一個 Polars 表達式，代表要計算的序列。
    - period: int, 平滑週期。

    返回：
    一個 Polars 表達式，代表平滑後的序列。
    """
    if period <= 0:
        raise ValueError("Wilder's Smoothing 的週期 (period) 必須是正整數。")
    return series.ewm_mean(com=period - 1, adjust=False).fill_nan(0).fill_null(0)
