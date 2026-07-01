# ==============================================================================
# § 公式 | 樞軸點 (Pivots)
# ==============================================================================
import polars as pl


def calculate(series: pl.Expr, left: int, right: int, **kwargs) -> pl.Expr:
    """
    計算高低樞軸點 (Swing Highs/Lows)。
    - 返回 1 代表高點 (Pivot High)。
    - 返回 -1 代表低點 (Pivot Low)。
    - 返回 0 代表不是樞軸點。

    契約：
    - series: pl.Expr, 要尋找樞軸點的序列 (例如 high 或 low)。
    - left: int, 左側需要比較的 K 棒數量。
    - right: int, 右側需要比較的 K 棒數量。
    - [150分健壯性] 繼承 DSL 系統的 .fill_null(False) 契約。
    """
    if left < 1 or right < 1:
        raise ValueError("pivots 的 left 和 right 參數必須至少為 1。")

    is_higher_than_left = pl.all_horizontal(
        [(series > series.shift(i)).fill_null(False) for i in range(1, left + 1)]
    )
    is_higher_than_right = pl.all_horizontal(
        [(series >= series.shift(-i)).fill_null(False) for i in range(1, right + 1)]
    )
    is_pivot_high = is_higher_than_left & is_higher_than_right

    is_lower_than_left = pl.all_horizontal(
        [(series < series.shift(i)).fill_null(False) for i in range(1, left + 1)]
    )
    is_lower_than_right = pl.all_horizontal(
        [(series <= series.shift(-i)).fill_null(False) for i in range(1, right + 1)]
    )
    is_pivot_low = is_lower_than_left & is_lower_than_right

    return (
        pl.when(is_pivot_high)
        .then(pl.lit(1, dtype=pl.Int8))
        .when(is_pivot_low)
        .then(pl.lit(-1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
    ).fill_null(
        0
    )  # 確保最終輸出不帶 Null
