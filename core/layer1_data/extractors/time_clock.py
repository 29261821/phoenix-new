import polars as pl
import numpy as np
from typing import Optional

class TimeClockExtractor:
    """
    [Layer 1 - ⚡ SOTA 核心] 事件時鐘與時間變形萃取器
    職責：將均勻的「物理時間 (Physical Time)」轉換為非線性的「市場感知時間 (Market Time)」。
    """
    def __init__(self, vol_window: int = 60, baseline_window: int = 1440):
        """
        :param vol_window: 計算短期波動率的視窗長度 (預設 60 根 1m K 線 = 1 小時)
        :param baseline_window: 計算歷史基準波動與成交量的視窗長度 (預設 1440 根 = 24 小時)
        """
        self.vol_window = vol_window
        self.baseline_window = baseline_window
        self.epsilon = 1e-8  # 防止除以零的微小值

    def apply_time_deformation(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        計算時間變形係數 (Time Deformation / d_tau)
        公式: dτ_t = (σ_t * V_t) / mean(σ * V)
        當數值 > 1：代表進入「超密集時間」(如 CPI 發布、大戶砸盤)
        當數值 < 1：代表進入「壓縮時間」(如流動性枯竭、休市)
        """
        # 1. 計算對數報酬率 (Log Return)
        # 注意：我們使用向前填充的 close，防止因為休市出現 NaN 破壞運算
        df = df.with_columns(
            pl.col("close").log().diff().fill_null(0.0).alias("log_ret")
        )
        
        # 2. 計算短期波動率 σ_t (Rolling Standard Deviation)
        df = df.with_columns(
            pl.col("log_ret").rolling_std(window_size=self.vol_window).fill_null(0.0).alias("sigma_t")
        )
        
        # 3. 計算市場活躍度 (Activity = 波動率 x 成交量)
        df = df.with_columns(
            (pl.col("sigma_t") * pl.col("volume")).alias("activity_t")
        )
        
        # 4. 計算基準活躍度 (Baseline Activity)
        df = df.with_columns(
            pl.col("activity_t").rolling_mean(window_size=self.baseline_window).alias("baseline_activity")
        )
        
        # 5. 計算時間變形係數 dτ_t，並對休市/無資料區間進行強制凍結
        df = df.with_columns(
            pl.when(pl.col("features_mask") == False)
            .then(0.0) # 休市或斷線時，市場時間流逝嚴格為 0
            .otherwise(
                pl.col("activity_t") / (pl.col("baseline_activity") + self.epsilon)
            )
            .fill_nan(1.0).fill_null(1.0) # 初期尚未滿 baseline_window 時，假設時間是均勻的 (=1.0)
            .alias("time_delta_tau")
        )
        
        # 移除運算過程中的中介欄位，保持特徵矩陣乾淨
        return df.drop(["log_ret", "sigma_t", "activity_t", "baseline_activity"])

    def apply_event_clock(self, df: pl.DataFrame, session_start_mins: int = 0, session_duration_mins: int = 1440) -> pl.DataFrame:
        """
        計算事件時鐘 (Event Clock)
        將距離開盤/收盤的時間標準化至 [-1, 1] 區間，幫助神經網路學習週期性。
        (預設以 Crypto 的 24 小時/ 1440 分鐘週期為例。若是美股，可透過 market_calendar 傳入 390 分鐘)
        """
        # 提取當天自午夜起算的總分鐘數
        df = df.with_columns(
            (pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute()).alias("mins_from_midnight")
        )
        
        # 計算距離開盤的相對分鐘數 (考慮跨日情況)
        df = df.with_columns(
            ((pl.col("mins_from_midnight") - session_start_mins) % 1440).alias("mins_since_open_raw")
        )
        
        # 轉換為神經網路友好的 [-1, 1] 空間
        half_session = session_duration_mins / 2.0
        
        df = df.with_columns([
            # -1 代表剛開盤，1 代表即將收盤
            ((pl.col("mins_since_open_raw") / half_session) - 1.0)
            .clip(-1.0, 1.0) # 強制約束邊界，防止異常值
            .alias("minutes_since_open_norm"),
            
            # 1 代表距離收盤還很久，-1 代表馬上收盤 (與上面反向，提供不同相位的資訊)
            (1.0 - (pl.col("mins_since_open_raw") / half_session))
            .clip(-1.0, 1.0)
            .alias("minutes_until_close_norm")
        ])
        
        return df.drop(["mins_from_midnight", "mins_since_open_raw"])

    def apply_physical_dt(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        計算真實物理時間差 (dt)，專供連續時間的 Mamba ODE 使用。
        這是 Red Team Phase 9 (Shot 01 & 04) 的核心防爆機制。
        精確捕捉包含週末跳空、API 斷線所流失的真實秒數。
        """
        # dt = 該筆 K 線的 datetime 減去前一筆 K 線的 datetime，單位轉為秒
        # 如果是第一筆，填補 0.0
        df = df.with_columns(
            (pl.col("datetime").diff().dt.total_milliseconds() / 1000.0)
            .fill_null(0.0)
            .alias("physical_dt_sec")
        )
        return df

    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        主介面：一次性套用所有的時間語意特徵
        """
        print("[*] 啟動 Time Clock Extractor: 萃取時間變形與事件時鐘...")
        df = self.apply_physical_dt(df)
        df = self.apply_time_deformation(df)
        df = self.apply_event_clock(df)
        return df

if __name__ == "__main__":
    # === 簡單測試區 ===
    # 模擬讀取我們先前 Data Lake 產出的 1m 數據
    print("--- TimeClockExtractor Test ---")
    try:
        # 替換為您本地真實的 parquet 檔案路徑
        test_df = pl.read_parquet("../../infra/data_lake/parquet_store/BTC_USDT/BTC_USDT_1m.parquet")
        
        extractor = TimeClockExtractor()
        enhanced_df = extractor.extract(test_df)
        
        print("\n[v] 成功萃取時間特徵！以下為包含 d_tau (time_delta_tau) 的最新資料片段：")
        print(enhanced_df.select(["datetime", "close", "volume", "time_delta_tau", "minutes_since_open_norm"]).tail(5))
        
    except FileNotFoundError:
        print("[!] 找不到 Parquet 測試檔案。請先執行 data_lake_builder.py。")