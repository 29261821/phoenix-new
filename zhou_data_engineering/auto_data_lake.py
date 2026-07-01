import os
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import polars as pl
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from dotenv import load_dotenv

# 載入 .env 檔案中的環境變數
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ZeroTouchDataLake")

class ZeroTouchDataLake:
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化非同步資料湖建構器
        """
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent / "market_data_lake"
        else:
            self.data_dir = Path(data_dir)
            
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        tv_username = os.getenv("TV_USERNAME")
        tv_password = os.getenv("TV_PASSWORD")
        
        if tv_username and tv_password:
            logger.info("[i] 已偵測到 API 金鑰/帳密，將啟用驗證連線 (Authenticated Mode)。")
            self.tv = TvDatafeed(username=tv_username, password=tv_password)
        else:
            logger.info("[i] 未偵測到帳密，將使用公開連線抓取數據 (Anonymous Mode)。")
            self.tv = TvDatafeed() 

    async def fetch_historical_async(self, symbol: str, exchange: str, interval: Interval, n_bars: int) -> pl.DataFrame:
        """
        [非同步抓取] 具備指數退避重試的歷史拉取 (通用型)
        """
        interval_name = interval.name
        logger.info(f"[*] 開始抓取 {symbol}:{exchange} {interval_name} 數據 ({n_bars} 根)...")
        
        max_retries = 5
        df_pd = None
        
        for attempt in range(1, max_retries + 1):
            try:
                # tvDatafeed 是同步的，我們使用 asyncio.to_thread 讓它不阻塞 Event Loop
                df_pd = await asyncio.to_thread(
                    self.tv.get_hist,
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    n_bars=n_bars
                )
                
                if df_pd is not None and not df_pd.empty:
                    logger.info(f"[{symbol}:{exchange}] Successfully fetched {interval_name} data on attempt {attempt}.")
                    break
                else:
                    logger.warning(f"[{symbol}:{exchange}] Attempt {attempt}: API returned empty data for {interval_name}.")
                    
            except Exception as e:
                logger.error(f"[{symbol}:{exchange}] 嘗試 {attempt} 失敗! 錯誤: {e}")
                
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.info(f"[{symbol}:{exchange}] 網路或速率限制，等待 {wait_time} 秒後重試...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"[{symbol}:{exchange}] 放棄抓取 {interval_name}：達到最大重試次數。")

        if df_pd is None or df_pd.empty:
            raise ValueError(f"No data returned for {symbol} on {exchange} ({interval_name}).")
            
        # Reset index and clean up Pandas DataFrame
        df_pd = df_pd.reset_index()
        if 'symbol' in df_pd.columns:
            df_pd = df_pd.drop(columns=['symbol'])
            
        # ---------------------------------------------------------------------
        # 防禦機制 1: Timezone Naïveté (時區裸奔與 8 小時未來視)
        # 強制標記 Asia/Taipei 並轉換為 UTC
        # ---------------------------------------------------------------------
        if df_pd['datetime'].dt.tz is None:
            df_pd['datetime'] = df_pd['datetime'].dt.tz_localize('Asia/Taipei').dt.tz_convert('UTC')
        else:
            df_pd['datetime'] = df_pd['datetime'].dt.tz_convert('UTC')
            
        df = pl.from_pandas(df_pd)
        df = df.rename({col: col.lower() for col in df.columns})
        
        # 剃除最後一根未收盤的 K 線，防止未來視
        df = df.slice(0, df.height - 1)
        df = df.sort("datetime")
        return df

    def apply_intraday_interpolation(self, df_15m: pl.DataFrame) -> pl.DataFrame:
        """
        防禦機制 2: 動態日曆萃取與盤中插值引擎 (Ghost Gap Interpolator)
        針對台灣股市: 09:00 - 13:15, 每 15 分鐘一根, 共 18 根
        """
        # 萃取有交易的營業日 (透過 Asia/Taipei 來判定確實的日期，避開跨日問題)
        unique_dates = df_15m.select(
            pl.col("datetime").dt.convert_time_zone("Asia/Taipei").dt.date().alias("trading_date")
        ).unique().sort("trading_date")

        # 生成該營業日的絕對刻度 (09:00 到 13:15 Asia/Taipei，然後轉回 UTC)
        master_timeline = unique_dates.select([
            pl.datetime_ranges(
                pl.col("trading_date").cast(pl.Datetime).dt.replace_time_zone("Asia/Taipei").dt.offset_by("9h"),
                pl.col("trading_date").cast(pl.Datetime).dt.replace_time_zone("Asia/Taipei").dt.offset_by("13h15m"),
                interval="15m",
                eager=False
            ).alias("datetime")
        ]).explode("datetime").with_columns(
            pl.col("datetime").dt.convert_time_zone("UTC")
        )

        # 進行 Left Join，找出缺口
        df_merged = master_timeline.join(df_15m, on="datetime", how="left")
        
        # 填補特徵與標記 (自首機制)
        df_filled = df_merged.with_columns([
            (pl.col("close").is_not_null() & (pl.col("volume") > 0)).alias("features_mask"),
            pl.when(pl.col("close").is_null()).then(3).otherwise(0).cast(pl.Int8).alias("features_reason"),
        ]).with_columns([
            pl.col("close").fill_null(strategy="forward"),
            pl.col("volume").fill_null(0.0)
        ]).with_columns([
            pl.col("open").fill_null(pl.col("close")),
            pl.col("high").fill_null(pl.col("close")),
            pl.col("low").fill_null(pl.col("close"))
        ]).drop_nulls(subset=["close"])  # 防呆：如果當天第一筆剛好遺失，移除之
        
        return df_filled

    def generate_micro_timeframes(self, df_15m: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """
        微觀多時間尺度重採樣 (30m, 60m, 4h)
        """
        timeframes = ["30m", "60m", "4h"]
        
        # 賦予 15m 雙軌時間與 close_time
        df_15m = df_15m.with_columns([
            (pl.col("datetime").dt.offset_by("15m") - pl.duration(milliseconds=1)).alias("close_time"),
            pl.col("datetime").dt.timestamp("ms").alias("datetime_ms"),
            (pl.col("datetime").dt.offset_by("15m") - pl.duration(milliseconds=1)).dt.timestamp("ms").alias("close_time_ms")
        ])
        results = {"15m": df_15m}
        
        for tf in timeframes:
            # Polars group_by_dynamic 在 UTC 時區下進行聚合非常完美
            df_tf = df_15m.group_by_dynamic("datetime", every=tf, closed="left", label="left").agg([
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                pl.col("features_mask").any(),
                pl.col("features_reason").max() 
            ]).drop_nulls()
            
            # 賦予聚合後 K 線的雙軌時間
            df_tf = df_tf.with_columns([
                (pl.col("datetime").dt.offset_by(tf) - pl.duration(milliseconds=1)).alias("close_time"),
                pl.col("datetime").dt.timestamp("ms").alias("datetime_ms"),
                (pl.col("datetime").dt.offset_by(tf) - pl.duration(milliseconds=1)).dt.timestamp("ms").alias("close_time_ms")
            ])
            results[tf] = df_tf
            
        return results

    def save_to_datalake(self, symbol: str, exchange: str, datasets: Dict[str, pl.DataFrame]):
        """寫入本地 Parquet，檔名格式: <symbol>_<exchange>_<tf>.parquet"""
        symbol_safe = symbol.replace("/", "_")
        exchange_safe = exchange.replace("/", "_")
        asset_dir = self.data_dir / symbol_safe
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        for tf, df in datasets.items():
            file_path = asset_dir / f"{symbol_safe}_{exchange_safe}_{tf}.parquet"
            df.write_parquet(str(file_path))
            logger.info(f"[v] 保存完成: {file_path.name} ({len(df)} 筆)")

    async def process_single_asset(self, symbol: str, exchange: str):
        """單一資產的雙軌抓取與處理"""
        try:
            # 1. 網路 I/O: 抓取微觀 15m 資料
            raw_15m_df = await self.fetch_historical_async(symbol, exchange, Interval.in_15_minute, n_bars=10000)
            
            # 2. 網路 I/O: 獨立抓取宏觀 1d 資料
            raw_1d_df = await self.fetch_historical_async(symbol, exchange, Interval.in_daily, n_bars=5000)
            
            if len(raw_15m_df) == 0:
                logger.warning(f"[{symbol}] 無 15m 資料可處理。")
                return

            # 3. 執行盤中幽靈缺口填補引擎 (針對 15m)
            filled_15m_df = self.apply_intraday_interpolation(raw_15m_df)

            # 4. 生成多週期 (並套用雙軌時間)
            datasets = self.generate_micro_timeframes(filled_15m_df)
            
            # 5. 處理宏觀時間尺度 (1d)
            if len(raw_1d_df) > 0:
                raw_1d_df = raw_1d_df.with_columns([
                    (pl.col("close").is_not_null() & (pl.col("volume") > 0)).alias("features_mask"),
                    pl.when(pl.col("close").is_null()).then(3).otherwise(0).cast(pl.Int8).alias("features_reason"),
                ]).with_columns([
                    (pl.col("datetime").dt.offset_by("1d") - pl.duration(milliseconds=1)).alias("close_time"),
                    pl.col("datetime").dt.timestamp("ms").alias("datetime_ms"),
                    (pl.col("datetime").dt.offset_by("1d") - pl.duration(milliseconds=1)).dt.timestamp("ms").alias("close_time_ms")
                ])
                datasets["1d"] = raw_1d_df

            # 6. 磁碟 I/O: 儲存
            self.save_to_datalake(symbol, exchange, datasets)
            
        except Exception as e:
            logger.error(f"[{symbol}] 致命錯誤與處理失敗: {e}")

    async def build_multi_asset_lake(self, assets: List[Dict[str, str]]):
        """
        [多核極限] 同時併發下載與處理多檔資產
        """
        logger.info(f"[🚀] 啟動多資產非同步資料湖建構管線: {[a['symbol'] for a in assets]}")
        
        # 使用 asyncio.gather 同時發起所有資產的雙軌下載與處理任務
        tasks = [self.process_single_asset(asset['symbol'], asset['exchange']) for asset in assets]
        await asyncio.gather(*tasks)
        
        logger.info("[🏁] 所有資產資料湖建構完畢！")


if __name__ == "__main__":
    # 在 Windows 避免非同步報錯
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 目標台股資產
    target_equities = [
        {"symbol": "2330", "exchange": "TWSE"},  # TSMC (台積電)
        {"symbol": "2308", "exchange": "TWSE"},  # Delta Electronics (台達電)
        {"symbol": "2454", "exchange": "TWSE"}   # MediaTek (聯發科)
    ]
    
    builder = ZeroTouchDataLake()
    
    # 啟動非同步事件迴圈
    asyncio.run(builder.build_multi_asset_lake(target_equities))
