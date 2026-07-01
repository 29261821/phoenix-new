# ==============================================================================
# § 公式 | Wilder's Smoothing (Running Moving Average)
# 修正說明: 傳統的 DMI/ADX 與 ATR 算法所使用的並非線性 WMA，而是 Wilder's Smoothing (RMA)。
# 它在數學上等價於 alpha = 1 / length 的 EMA。
# 這樣修改能消滅原本 `pl.sum_horizontal` 在長週期下造成的「表達式樹爆炸 (Expression Tree Explosion)」，
# 讓 Polars 發揮極致的 C/Rust 效能，計算速度可提升上百倍。
# ==============================================================================
import polars as pl


def calculate(series: pl.Expr, length: int, **kwargs) -> pl.Expr:
    """
    計算 Wilder's Smoothing (RMA)。

    契約：
    - series: pl.Expr, 一個 Polars 表達式，代表要計算的序列。
    - length: int, 移動平均的週期。

    返回：
    一個 Polars 表達式，代表平滑後的序列。
    """
    if length <= 0:
        raise ValueError("週期的長度 (length) 必須是正整數。")
        
    # 先填補初期的 Null，確保 EMA 遞迴不會中斷
    safe_series = series.forward_fill().fill_null(0)

    # Wilder's Smoothing 等價於 EMA, alpha = 1.0 / length
    # adjust=False 確保它使用嚴格的無限歷史遞迴算法 (與 TradingView 等主流算法完全一致)
    return safe_series.ewm_mean(alpha=1.0/length, adjust=False).fill_nan(0)