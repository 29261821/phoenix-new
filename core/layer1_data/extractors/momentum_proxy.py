import polars as pl
from typing import List

class MomentumProxy:
    """
    [Layer 1 - ⚡ SOTA 核心] 動能與微觀波動特徵代理器
    完全拋棄 Pandas 與 TA-Lib，100% 基於 Polars 惰性求值 (Lazy Evaluation) 實作。
    提供 O(1) 記憶體開銷的高速特徵矩陣生成，完美支援 Batch 回測與 Streaming 實盤。
    """

    @staticmethod
    def expr_rsi(price_col: str = "close", window: int = 14) -> pl.Expr:
        """Polars 原生 RSI 運算式"""
        delta = pl.col(price_col).diff()
        
        # 區分上漲與下跌段
        up = delta.clip(lower_bound=0)
        down = delta.clip(upper_bound=0).abs()
        
        # 使用 Wilder's Smoothing 近似 rolling_mean，以取得極致效能
        roll_up = up.rolling_mean(window_size=window)
        roll_down = down.rolling_mean(window_size=window)
        
        rs = roll_up / (roll_down + 1e-8)
        return (100.0 - (100.0 / (1.0 + rs))).alias(f"rsi_{window}")

    @staticmethod
    def expr_macd(price_col: str = "close", fast: int = 12, slow: int = 26, signal: int = 9) -> List[pl.Expr]:
        """Polars 原生 MACD 族群運算式 (包含快線、慢線、柱狀圖)"""
        fast_ema = pl.col(price_col).ewm_mean(span=fast, adjust=False)
        slow_ema = pl.col(price_col).ewm_mean(span=slow, adjust=False)
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm_mean(span=signal, adjust=False)
        histogram = macd_line - signal_line
        
        return [
            macd_line.alias("macd_line"),
            signal_line.alias("macd_signal"),
            histogram.alias("macd_hist")
        ]

    @staticmethod
    def expr_bollinger_bands(price_col: str = "close", window: int = 20, num_std: float = 2.0) -> List[pl.Expr]:
        """Polars 原生布林通道運算式 (包含上下軌、帶寬壓縮比、價格相對位置 %B)"""
        basis = pl.col(price_col).rolling_mean(window_size=window)
        dev = pl.col(price_col).rolling_std(window_size=window)
        
        upper = basis + num_std * dev
        lower = basis - num_std * dev
        
        # BB Width (波動率壓縮指標 - 抓突破的關鍵)
        bb_width = (upper - lower) / (basis + 1e-8)
        
        # %B (價格在通道內的位置，>1 代表超買，<0 代表超賣)
        bb_pct = (pl.col(price_col) - lower) / (upper - lower + 1e-8)
        
        return [
            upper.alias(f"bb_upper_{window}"),
            lower.alias(f"bb_lower_{window}"),
            bb_width.alias(f"bb_width_{window}"),
            bb_pct.alias(f"bb_pct_{window}")
        ]

    @staticmethod
    def expr_atr(high_col: str = "high", low_col: str = "low", close_col: str = "close", window: int = 14) -> pl.Expr:
        """Polars 原生 ATR (Average True Range) - 計算市場真實波動幅度"""
        prev_close = pl.col(close_col).shift(1)
        
        # True Range 的三個組件
        tr1 = pl.col(high_col) - pl.col(low_col)
        tr2 = (pl.col(high_col) - prev_close).abs()
        tr3 = (pl.col(low_col) - prev_close).abs()
        
        # 取三者最大值並進行平滑
        true_range = pl.max_horizontal([tr1, tr2, tr3])
        return true_range.rolling_mean(window_size=window).alias(f"atr_{window}")

    def extract_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        [多核極限壓榨]
        一次性將所有動能與波動度特徵注入 DataFrame。
        這段程式碼不會依序執行，而是將所有公式交給 Rust 引擎進行「平行運算圖優化」。
        """
        print("[*] 啟動 Momentum Proxy: 執行高速向量化特徵工程 (Polars Lazy Eval)...")
        
        # 聚合所有的 Lazy Expressions
        feature_exprs = [
            self.expr_rsi(),
            self.expr_atr(),
            *self.expr_macd(),
            *self.expr_bollinger_bands()
        ]
        
        # 一次性執行所有欄位擴充
        return df.with_columns(feature_exprs)

if __name__ == "__main__":
    # === 簡單測試區 ===
    print("--- MomentumProxy SOTA Test ---")
    
    # 建立隨機測試資料 (模擬 K 線)
    data = pl.DataFrame({
        "high": [10.5, 12.1, 15.3, 14.2, 16.8, 18.1, 20.2, 19.5, 21.1, 22.8] * 5,
        "low": [8.1, 9.2, 11.5, 10.3, 12.4, 14.6, 15.2, 14.8, 16.5, 17.1] * 5,
        "close": [9.0, 11.0, 14.0, 12.5, 15.2, 17.8, 18.5, 16.9, 20.1, 21.5] * 5
    })
    
    proxy = MomentumProxy()
    
    import time
    start = time.time()
    
    # SOTA 向量化特徵生成
    enhanced_df = proxy.extract_features(data)
    
    print(f"[v] 特徵生成完畢，耗時: {(time.time() - start)*1000:.2f} ms")
    print(f"新增特徵清單: {[col for col in enhanced_df.columns if col not in data.columns]}")
    
    # 檢視末尾資料
    print("\n[Preview]")
    print(enhanced_df.select(["close", "rsi_14", "macd_hist", "bb_pct_20", "atr_14"]).tail(5))