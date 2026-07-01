import os
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import polars as pl
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

    async def fetch_historical_15m_async(self, symbol: str, exchange: str, n_bars: int = 10000) -> pl.DataFrame:
        """
        [非同步抓取] 具備指數退避重試的歷史拉取
        """
        logger.info(f"[*] 開始抓取 {symbol}:{exchange} 15m 數據...")
        
        max_retries = 5
        df_pd = None
        
        for attempt in range(1, max_retries + 1):
            try:
                # tvDatafeed 是同步的，我們使用 asyncio.to_thread 讓它不阻塞 Event Loop
                df_pd = await asyncio.to_thread(
                    self.tv.get_hist,
                    symbol=symbol,
                    exchange=exchange,
                    interval=Interval.in_15_minute,
                    n_bars=n_bars
                )
                
                if df_pd is not None and not df_pd.empty:
                    logger.info(f"[{symbol}:{exchange}] Successfully fetched data on attempt {attempt}.")
                    break
                else:
                    logger.warning(f"[{symbol}:{exchange}] Attempt {attempt}: API returned empty data.")
                    
            except Exception as e:
                logger.error(f"[{symbol}:{exchange}] 嘗試 {attempt} 失敗! 錯誤: {e}")
                
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.info(f"[{symbol}:{exchange}] 網路或速率限制，等待 {wait_time} 秒後重試...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"[{symbol}:{exchange}] 放棄抓取：達到最大重試次數。")

        if df_pd is None or df_pd.empty:
            raise ValueError(f"No data returned for {symbol} on {exchange}.")
            
        # Reset index and clean up Pandas DataFrame
        df_pd = df_pd.reset_index()
        if 'symbol' in df_pd.columns:
            df_pd = df_pd.drop(columns=['symbol'])
            
        df = pl.from_pandas(df_pd)
        df = df.rename({col: col.lower() for col in df.columns})
        
        # ---------------------------------------------------------------------
        # LOOKAHEAD-BIAS DEFENSE: 剃除最後一根未收盤的 K 線
        # ---------------------------------------------------------------------
        df = df.slice(0, df.height - 1)
        return df.sort("datetime")

    def generate_multi_timeframes(self, df_15m: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """
        [SOTA 升級] 多時間尺度重採樣，並強制寫入物理結束時間 (Close Time)
        """
        timeframes = ["30m", "60m", "4h", "1d"]
        
        # 1. 賦予基準 15m 線 close_time
        df_15m = df_15m.with_columns(
            (pl.col("datetime").dt.offset_by("15m") - pl.duration(milliseconds=1)).alias("close_time")
        )
        results = {"15m": df_15m}
        
        for tf in timeframes:
            # Polars group_by_dynamic 預設以區間的起始點 (Open Time) 作為標籤
            df_tf = df_15m.group_by_dynamic("datetime", every=tf, closed="left", label="left").agg([
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum()
            ]).drop_nulls()
            
            # 2. 賦予宏觀 K 線 close_time
            df_tf = df_tf.with_columns(
                (pl.col("datetime").dt.offset_by(tf) - pl.duration(milliseconds=1)).alias("close_time")
            )
            results[tf] = df_tf
            
        return results

    def save_to_datalake(self, symbol: str, datasets: Dict[str, pl.DataFrame]):
        """寫入本地 Parquet"""
        symbol_safe = symbol.replace("/", "_")
        asset_dir = self.data_dir / symbol_safe
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        for tf, df in datasets.items():
            file_path = asset_dir / f"{symbol_safe}_{tf}.parquet"
            df.write_parquet(str(file_path))
            logger.info(f"[v] 保存完成: {file_path.name} ({len(df)} 筆)")

    async def process_single_asset(self, symbol: str, exchange: str):
        """單一資產的完整生命週期管線"""
        try:
            # 1. 網路 I/O: 非同步抓取
            raw_15m_df = await self.fetch_historical_15m_async(symbol, exchange, n_bars=10000)
            
            if len(raw_15m_df) == 0:
                logger.warning(f"[{symbol}] 無資料可處理。")
                return

            # 2. CPU 密集型: 降維、多週期生成與加入 close_time (使用極速 Polars)
            multi_tf_datasets = self.generate_multi_timeframes(raw_15m_df)
            
            # 3. 磁碟 I/O: 儲存
            self.save_to_datalake(symbol, multi_tf_datasets)
            
        except Exception as e:
            logger.error(f"[{symbol}] 致命錯誤與處理失敗: {e}")

    async def build_multi_asset_lake(self, assets: List[Dict[str, str]]):
        """
        [多核極限] 同時併發下載與處理多檔資產
        """
        logger.info(f"[🚀] 啟動多資產非同步資料湖建構管線: {[a['symbol'] for a in assets]}")
        
        # 使用 asyncio.gather 同時發起所有資產的下載與處理任務
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
