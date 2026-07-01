# ==============================================================================
# § 公式 | 平均真實波幅 (Average True Range)
# ==============================================================================
import polars as pl

from .tr import calculate as tr
from .wilder_smooth import calculate as wilder_smooth


def calculate(df: pl.DataFrame, period: int, **kwargs) -> pl.Expr:
    """
    計算平均真實波幅 (ATR)。
    這是 TR 的 Wilder's Smoothing 平滑移動平均。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。
    - period: int, ATR 的計算週期。

    返回：
    一個 Polars 表達式，代表 ATR 序列。
    """
    if period <= 0:
        raise ValueError("ATR 的週期 (period) 必須是正整數。")

    # 修改處：確保 TR 序列沒有負值（TR 理論上恆正，但為了預防浮點數誤差）
    tr_series = tr(df).clip(lower_bound=0)
    return wilder_smooth(series=tr_series, period=period).fill_nan(0)
