import polars as pl
from typing import Dict
import logging

logger = logging.getLogger("TimeframeAligner")

class TimeframeAligner:
    """
    [Layer 1 - ⚡ SOTA 核心] 多時間框架對齊縫合器
    職責：將基準 K 線與更高階的尺度進行嚴格的 Backward 對齊。
    
    SOTA 升級亮點 (End-Time Indexing)：
    1. 結束時間錨定: 依賴上游 DataLake 預先算好的 close_time，徹底消滅所有時間戳重疊的前視偏差。
    2. 極致輕量化: 廢除動態 offset_by 計算，大幅降低實盤 CPU 開銷。
    3. 資訊衰減追蹤: 延遲時間改為「距離上級 K 線收盤過了多久」，物理意義更精確。
    """

    @staticmethod
    def align_timeframes(base_df: pl.DataFrame, macro_dfs: Dict[str, pl.DataFrame], base_timeframe: str = "1m") -> pl.DataFrame:
        """
        將基準與其他尺度進行 Join 縫合，並計算延遲。
        """
        print(f"[*] 啟動 Timeframe Aligner: 執行多尺度對齊縫合與衰減計算 (End-Time Backward Join, Base: {base_timeframe})...")
        
        df_aligned = base_df
        
        for tf, macro_df in macro_dfs.items():
            if tf == base_timeframe:
                continue
                
            # 🚨 [防爆紅線] 嚴格斷言上游資料必須提供物理結束時間
            if "close_time" not in macro_df.columns:
                raise ValueError(f"[{tf}] 宏觀數據缺失 'close_time' 欄位！請檢查 Data Lake 或實盤推播邏輯。")
                
            # 1. 確保宏觀數據按「物理結束時間」嚴格排序
            macro_df = macro_df.sort("close_time")
            
            # 2. 重新命名 close_time，避免與基準表潛在的欄位衝突
            macro_time_col = f"close_time_macro_{tf}"
            macro_df = macro_df.rename({"close_time": macro_time_col})
            
            # 3. 選取需要縫合的特徵 (排除已經存在的 mask, reason 等)，並動態加上 _tf 後綴
            cols_to_keep = [macro_time_col, "open", "high", "low", "close", "volume"]
            macro_select = macro_df.select([
                pl.col(c).alias(f"{c}_{tf}") if c != macro_time_col else pl.col(c)
                for c in macro_df.columns if c in cols_to_keep
            ])
            
            # 4. [防爆紅線] 執行 End-Time Backward Join
            # 邏輯：用基準的 datetime 去往回找巨觀的 close_time
            df_aligned = df_aligned.join_asof(
                macro_select,
                left_on="datetime",
                right_on=macro_time_col,
                strategy="backward"
            )
            
            # 5. 計算資料衰減 (Staleness Tracking)
            # 公式: (當前開盤時間 - 宏觀 K 線物理收盤時間) / 60000 毫秒 = 延遲的分鐘數
            staleness_col = f"staleness_{tf}_mins"
            df_aligned = df_aligned.with_columns(
                ((pl.col("datetime") - pl.col(macro_time_col)).dt.total_milliseconds() / 60000.0)
                .alias(staleness_col)
            )
            
            # 6. 拋棄中介的 macro 時間戳，保持矩陣乾淨
            df_aligned = df_aligned.drop(macro_time_col)
            
        # 7. 全局向前填充 (處理開盤初期還沒有宏觀 K 線的空窗期)
        df_aligned = df_aligned.fill_null(strategy="forward").fill_null(0.0)
        
        return df_aligned

if __name__ == "__main__":
    from datetime import datetime
    print("--- TimeframeAligner SOTA Test (End-Time Indexing Proof) ---")
    
    # 建立基準 1m K線 (10:00 到 10:05)
    base = pl.DataFrame({
        "datetime": pl.datetime_range(datetime(2026, 6, 1, 10, 0), datetime(2026, 6, 1, 10, 5), "1m", eager=True),
        "close_1m": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    })
    
    # 建立 5m 巨觀 K 線 (包含 09:55 與 10:00 兩根)
    # 注意：我們直接掛載 Data Lake 寫死的 close_time
    macro_5m = pl.DataFrame({
        "datetime": [datetime(2026, 6, 1, 9, 55), datetime(2026, 6, 1, 10, 0)],
        "close_time": [
            datetime(2026, 6, 1, 9, 59, 59, 999000),  # 09:55 結束時間
            datetime(2026, 6, 1, 10, 4, 59, 999000)   # 10:00 結束時間
        ],
        "open": [90.0, 99.0],
        "high": [95.0, 105.0],
        "low": [88.0, 98.0],
        "close": [94.0, 103.0], 
        "volume": [500.0, 1000.0]
    })
    
    macros = {"5m": macro_5m}
    aligned_df = TimeframeAligner.align_timeframes(base, macros)
    
    print("\n[v] 縫合成功！請檢驗前視偏差防護結果：")
    print("邏輯斷言：10:04 的 1m 線，只能配對到 09:55 的 5m 收盤價 (94.0)。直到 10:05 才能拿到 103.0！")
    print("-" * 75)
    print(aligned_df.select(["datetime", "close_1m", "close_5m", "staleness_5m_mins"]))