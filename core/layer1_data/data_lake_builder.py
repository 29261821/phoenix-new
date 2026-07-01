import ccxt.async_support as ccxt_async  # SOTA 升級: 導入非同步 CCXT
import polars as pl
import os
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
from dotenv import load_dotenv
from pathlib import Path
import logging

# 載入 .env 檔案中的環境變數
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataLakeBuilder")

class DataLakeBuilder:
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化非同步資料湖建構器
        """
        if data_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.data_dir = project_root / "infra" / "data_lake" / "parquet_store"
        else:
            self.data_dir = Path(data_dir)
            
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        exchange_config = {
            'enableRateLimit': True,
        }
        
        api_key = os.getenv("BINANCE_API_KEY")
        secret_key = os.getenv("BINANCE_SECRET_KEY")
        
        if api_key and secret_key:
            exchange_config['apiKey'] = api_key
            exchange_config['secret'] = secret_key
            logger.info("[i] 已偵測到 API 金鑰，將啟用驗證連線。")
        else:
            logger.info("[i] 未偵測到完整 API 金鑰，將使用公開連線抓取數據。")

        # 初始化非同步交易所實例
        self.exchange = ccxt_async.binance(exchange_config)

    async def close(self):
        """關閉非同步連線"""
        await self.exchange.close()

    async def fetch_historical_1m_async(self, asset: str, start_date: str, end_date: str) -> pl.DataFrame:
        """
        [SOTA 非同步抓取] 具備指數退避重試的極速歷史拉取
        """
        logger.info(f"[*] 開始抓取 {asset} 1m 數據 ({start_date} to {end_date})...")
        
        since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
        until = self.exchange.parse8601(f"{end_date}T00:00:00Z")
        
        all_ohlcv = []
        current_since = since
        max_retries = 5
        
        while current_since < until:
            retries = 0
            while retries < max_retries:
                try:
                    ohlcv = await self.exchange.fetch_ohlcv(asset, '1m', since=current_since, limit=1000)
                    if not ohlcv:
                        break # 資料抓完了
                    
                    all_ohlcv.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 60000 
                    
                    # 遵守交易所的 Rate Limit，讓出 Event Loop 控制權
                    await asyncio.sleep(self.exchange.rateLimit / 1000)
                    break # 成功抓取，跳出重試迴圈
                    
                except ccxt_async.RateLimitExceeded:
                    wait_time = 2 ** retries
                    logger.warning(f"[{asset}] Rate Limit! 等待 {wait_time} 秒...")
                    await asyncio.sleep(wait_time)
                    retries += 1
                except ccxt_async.NetworkError as e:
                    logger.warning(f"[{asset}] 網路異常: {e}，重試中...")
                    await asyncio.sleep(2 ** retries)
                    retries += 1
                except Exception as e:
                    logger.error(f"[{asset}] 致命錯誤: {e}")
                    raise e
                    
            if retries == max_retries:
                logger.error(f"[{asset}] 放棄抓取：達到最大重試次數。")
                break
                
        df = pl.DataFrame(all_ohlcv, schema=["timestamp", "open", "high", "low", "close", "volume"], orient="row")
        return df.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"))

    def apply_missingness_encoding(self, df: pl.DataFrame) -> pl.DataFrame:
        """三維遺失編碼器 (同步 CPU 運算)"""
        start_dt = df["datetime"].min()
        end_dt = df["datetime"].max()
        
        continuous_time = pl.DataFrame({
            "datetime": pl.datetime_range(start_dt, end_dt, "1m", eager=True).cast(pl.Datetime("ms"))
        })
        
        df_aligned = continuous_time.join(df, on="datetime", how="left")
        
        df_encoded = df_aligned.with_columns([
            (pl.col("close").is_not_null() & (pl.col("volume") > 0)).alias("features_mask"),
            pl.when(pl.col("close").is_null()).then(3).otherwise(0).cast(pl.Int8).alias("features_reason"),
        ])
        
        return df_encoded.with_columns([
            pl.col("open").fill_null(strategy="forward").fill_null(0.0).alias("open"),
            pl.col("high").fill_null(strategy="forward").fill_null(0.0).alias("high"),
            pl.col("low").fill_null(strategy="forward").fill_null(0.0).alias("low"),
            pl.col("close").fill_null(strategy="forward").fill_null(0.0).alias("close"),
            pl.col("volume").fill_null(0.0),
            pl.col("timestamp").fill_null(strategy="forward").fill_null(0) 
        ])

    def generate_multi_timeframes(self, df_1m: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """
        [SOTA 升級] 多時間尺度重採樣，並強制寫入物理結束時間 (Close Time)
        """
        timeframes = ["1m", "5m", "15m", "1h", "1d"]
        
        # ==============================================================================
        # 🚨 [防未來視鋼印] 生成 close_time
        # 將「這根 K 線何時才算真正結束」的因果律，直接寫死在 Parquet 檔案的硬碟裡。
        # 未來的 Aligner 只能依賴這個 close_time 進行 Backward Join。
        # ==============================================================================
        
        # 1. 賦予基準 1m 線 close_time (e.g., 10:00 的線，close_time 是 10:00:59.999)
        df_1m = df_1m.with_columns(
            (pl.col("datetime").dt.offset_by("1m") - pl.duration(milliseconds=1)).alias("close_time")
        )
        results = {"1m": df_1m}
        
        for tf in timeframes[1:]:
            # Polars group_by_dynamic 預設以區間的起始點 (Open Time) 作為標籤
            df_tf = df_1m.group_by_dynamic("datetime", every=tf).agg([
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                pl.col("features_mask").any(),
                pl.col("features_reason").max() 
            ])
            
            # 2. 賦予宏觀 K 線 close_time (e.g., 10:00 的 5m 線，close_time 是 10:04:59.999)
            df_tf = df_tf.with_columns(
                (pl.col("datetime").dt.offset_by(tf) - pl.duration(milliseconds=1)).alias("close_time")
            )
            results[tf] = df_tf
            
        return results

    def save_to_datalake(self, asset: str, datasets: Dict[str, pl.DataFrame]):
        """寫入本地 Parquet"""
        symbol_safe = asset.replace("/", "_")
        asset_dir = self.data_dir / symbol_safe
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        for tf, df in datasets.items():
            file_path = asset_dir / f"{symbol_safe}_{tf}.parquet"
            df.write_parquet(str(file_path))
            logger.info(f"[v] 保存完成: {file_path.name} ({len(df)} 筆)")

    async def process_single_asset(self, asset: str, start_date: str, end_date: str):
        """單一資產的完整生命週期管線"""
        # 1. 網路 I/O: 非同步抓取
        raw_1m_df = await self.fetch_historical_1m_async(asset, start_date, end_date)
        
        if len(raw_1m_df) == 0:
            logger.warning(f"[{asset}] 無資料可處理。")
            return

        # 2. CPU 密集型: 標記、降維、縫合 (使用極速 Polars)
        encoded_1m_df = self.apply_missingness_encoding(raw_1m_df)
        multi_tf_datasets = self.generate_multi_timeframes(encoded_1m_df)
        
        # 3. 磁碟 I/O: 儲存
        self.save_to_datalake(asset, multi_tf_datasets)

    async def build_multi_asset_lake(self, assets: List[str], start_date: str, end_date: str):
        """
        [多核極限] 同時併發下載與處理多檔資產
        """
        logger.info(f"[🚀] 啟動多資產非同步資料湖建構管線: {assets}")
        
        # 使用 asyncio.gather 同時發起所有資產的下載與處理任務
        tasks = [self.process_single_asset(asset, start_date, end_date) for asset in assets]
        await asyncio.gather(*tasks)
        
        await self.close()
        logger.info("[🏁] 所有資產資料湖建構完畢！")

if __name__ == "__main__":
    # 在 Windows 避免非同步報錯
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 想要同時抓取的資產清單 (可以無縫擴充到數十檔)
    target_assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    start_str = (datetime.now(timezone.utc) - timedelta(days=2400)).strftime("%Y-%m-%d")
    end_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    builder = DataLakeBuilder()
    
    # 啟動非同步事件迴圈
    asyncio.run(builder.build_multi_asset_lake(target_assets, start_str, end_str))