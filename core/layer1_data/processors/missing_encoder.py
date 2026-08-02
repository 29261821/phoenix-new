import polars as pl
from typing import Optional
from enum import IntEnum

class MissingReason(IntEnum):
    """
    [嚴格型別] 缺失原因狀態碼
    確保全系統對於「沒有資料」的認知具備統一且唯一的物理意義。
    """
    NORMAL = 0     # 正常交易
    CLOSED = 1     # 市場休市 (例: 週末的美股)
    HALTED = 2     # 熔斷或暫停交易
    API_DOWN = 3   # 應該要有資料，但 API 斷線或流動性徹底枯竭 (0 成交量)

class MissingnessEncoder:
    """
    [Layer 1 - ⚡ SOTA 核心] 三維遺失編碼器 (Value, Mask, Reason)
    職責：處理休市與斷線導致的資料缺失，確保狀態空間連續且具有物理意義。
    升級亮點：引入 IntEnum、極限 NaN 防護、單次 Rust Pass 效能最佳化。
    """

    def encode(self, df: pl.DataFrame, is_market_open_map: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        將原始 DataFrame 轉譯為三維遺失編碼結構。
        
        :param df: 原始 K 線資料
        :param is_market_open_map: 選填，提供特定時間點的開盤狀態查詢表
        """
        print("[*] 執行三維遺失編碼 (Missingness Encoding)...")
        
        # 1. 定義 Mask 條件運算式 (Lazy Expression)
        # 條件：close 存在 且 volume > 0 才算有效交易
        mask_cond = pl.col("close").is_not_null() & (pl.col("volume") > 0)
        
        # 2. 定義 Reason 運算式
        if is_market_open_map is not None:
            # 高效 Left Join 日曆資訊
            df = df.join(is_market_open_map, on="datetime", how="left")
            reason_expr = (
                pl.when(mask_cond).then(MissingReason.NORMAL.value)
                .when(pl.col("is_open") == False).then(MissingReason.CLOSED.value)
                .otherwise(MissingReason.API_DOWN.value)
                .cast(pl.Int8)
                .alias("features_reason")
            )
        else:
            reason_expr = (
                pl.when(mask_cond).then(MissingReason.NORMAL.value)
                .otherwise(MissingReason.API_DOWN.value)
                .cast(pl.Int8)
                .alias("features_reason")
            )
            
        # 3. 定義 Value 填充運算式 (極限 NaN 防護)
        # 策略: 先做 Forward Fill，但如果第一筆就是 NaN，ffill 會無效。
        # 因此加上第二層 fill_null(0.0) 絕對防線，確保神經網路輸入層 100% 不會吃到 NaN 拋錯。
        fill_exprs = [
            pl.col("open").fill_null(strategy="forward").fill_null(0.0).alias("open"),
            pl.col("high").fill_null(strategy="forward").fill_null(0.0).alias("high"),
            pl.col("low").fill_null(strategy="forward").fill_null(0.0).alias("low"),
            pl.col("close").fill_null(strategy="forward").fill_null(0.0).alias("close"),
            pl.col("volume").fill_null(0.0).alias("volume") # 成交量缺失一律補 0
        ]
        
        # 4. [效能榨取] 單次 Pass 執行所有運算
        df = df.with_columns([
            mask_cond.alias("features_mask"),
            reason_expr,
            *fill_exprs
        ])
        
        # 5. 清理暫存欄位，保持特徵矩陣純淨
        if is_market_open_map is not None and "is_open" in df.columns:
            df = df.drop("is_open")
            
        return df

if __name__ == "__main__":
    from datetime import datetime
    print("--- MissingnessEncoder SOTA Test ---")
    
    # 建立極端測試資料：第一筆就是 NaN (模擬 API 一開機就沒抓到)
    data = pl.DataFrame({
        "datetime": pl.datetime_range(datetime(2026, 6, 1), datetime(2026, 6, 1, 0, 5), "1m", eager=True),
        "close": [None, 101.0, None, None, 102.0],
        "volume": [None, 5.0, 0.0, 0.0, 8.0]
    })
    
    encoder = MissingnessEncoder()
    encoded_df = encoder.encode(data)
    
    print("\n[v] SOTA 編碼結果 (留意第一筆的絕對 0.0 防護與 Reason 標記)：")
    print(encoded_df)