# ==============================================================================
# § 公式 | Z-Score
# ==============================================================================
import polars as pl

from .sma import calculate as sma
from .stddev import calculate as stddev

# src/features/functions/zscore.py


def calculate(series: pl.Expr, period: int, **kwargs) -> pl.Expr:
    """
    計算 Z-Score。
    """
    if period <= 1:
        raise ValueError("Z-Score 的週期 (period) 必須大於 1。")

    # 確保輸入序列無空值
    safe_series = series.fill_null(strategy="forward").fill_null(0)

    mean = sma(safe_series, length=period)
    std = stddev(safe_series, period=period)

    epsilon = 1e-9
    z = (safe_series - mean) / (std + epsilon)

    # 修改處：
    # 1. fill_nan(0) 防止 SVD 崩潰
    # 2. clip 限制在 [-10, 10] 之間。在統計學上，Z > 10 已經是極端異常值。
    # 限制極值能讓模型更關注常態分佈區域，並防止 SVD 被單一離群點主導。
    return z.fill_nan(0).fill_null(0).clip(lower_bound=-10.0, upper_bound=10.0)
