# ==============================================================================
# § 公式 | 真實波幅 (True Range)
# ==============================================================================
import polars as pl


def calculate(df: pl.DataFrame, **kwargs) -> pl.Expr:
    """
    計算真實波幅 (True Range)。

    契約：
    - df 必須包含 'high', 'low', 'close' 欄位。

    返回：
    一個 Polars 表達式，代表每個時間點的 TR 值。
    """
    h, l, c = pl.col("high"), pl.col("low"), pl.col("close")
    # 增加 fill_null(c) 以確保在序列開頭不會因 prev(c, 1) 為空而產生 null
    # 使用當前的 close 作為填充值，是比 0 更合理的預設
    prev_c = c.shift(1).fill_null(c)

    # 修改處：加上 fill_nan 並確保數值不為負 (雖然 abs 已處理，但多一層保險防止浮點誤差)
    tr_expr = (
        pl.max_horizontal((h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs())
        .fill_nan(0)
        .fill_null(0)
        .clip(lower_bound=0)
    )

    return tr_expr
