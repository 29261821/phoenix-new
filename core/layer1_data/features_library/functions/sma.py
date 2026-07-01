# ==============================================================================
# § 公式 | 簡單移動平均 (Simple Moving Average)
# ==============================================================================
import polars as pl


def calculate(series: pl.Expr, length: int, **kwargs) -> pl.Expr:
    """
    計算簡單移動平均 (SMA)。

    契約：
    - series: pl.Expr, 一個 Polars 表達式，代表要計算的序列。
    - length: int, 移動平均的週期。

    返回：
    一個 Polars 表達式，代表 SMA 序列。
    """
    if length <= 0:
        raise ValueError("SMA 的週期 (length) 必須是正整數。")
    return series.rolling_mean(window_size=length).fill_nan(0).fill_null(0)
