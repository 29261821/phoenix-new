import os
import logging
import time
from pathlib import Path
from typing import List, Dict
import polars as pl
from tvDatafeed import TvDatafeed, Interval
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# -------------------------------------------------------------------------
# 1. Zero-Touch Environment & Logging Setup
# -------------------------------------------------------------------------
# Set up autonomous directory creation relative to the script's execution path
BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA_LAKE_PATH = BASE_DIR / "market_data_lake"
LOGS_PATH = BASE_DIR / "logs"

# Ensure directories exist without user intervention
DATA_LAKE_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_PATH / "data_lake_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# 2. Pipeline Architecture
# -------------------------------------------------------------------------
class ZeroTouchDataLake:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        
        tv_username = os.getenv("TV_USERNAME")
        tv_password = os.getenv("TV_PASSWORD")
        
        if tv_username and tv_password:
            logger.info("Initializing TvDatafeed (Authenticated Mode)...")
            self.tv = TvDatafeed(username=tv_username, password=tv_password)
        else:
            logger.info("Initializing TvDatafeed (Anonymous Mode)...")
            self.tv = TvDatafeed() 
            
        logger.info(f"Storage path automatically resolved to: {self.storage_path}")

    def fetch_base_data(self, symbol: str, exchange: str, n_bars: int = 5000, max_retries: int = 5, retry_delay: int = 10) -> pl.DataFrame:
        """
        Fetches 15-minute K-line data, applies structural lookahead-bias defense,
        and includes robust retry mechanisms.
        """
        logger.info(f"[{symbol}:{exchange}] Fetching {n_bars} raw 15m bars...")
        
        df_pd = None
        for attempt in range(1, max_retries + 1):
            try:
                df_pd = self.tv.get_hist(
                    symbol=symbol,
                    exchange=exchange,
                    interval=Interval.in_15_minute,
                    n_bars=n_bars
                )
                
                # Check if data is valid
                if df_pd is not None and not df_pd.empty:
                    logger.info(f"[{symbol}:{exchange}] Successfully fetched data on attempt {attempt}.")
                    break
                else:
                    logger.warning(f"[{symbol}:{exchange}] Attempt {attempt}: API returned empty data.")
                    
            except Exception as e:
                # Enhanced error logging
                logger.error(f"[{symbol}:{exchange}] Attempt {attempt} failed! Error reason: {str(e)}")
                
            if attempt < max_retries:
                logger.info(f"[{symbol}:{exchange}] Retrying in {retry_delay} seconds (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                logger.error(f"[{symbol}:{exchange}] All {max_retries} attempts failed. Giving up on this asset.")

        if df_pd is None or df_pd.empty:
            raise ValueError(f"No data returned for {symbol} on {exchange} after {max_retries} attempts. Possible rate limit or symbol delisted.")
        
        # Reset index and clean up Pandas DataFrame
        df_pd = df_pd.reset_index()
        if 'symbol' in df_pd.columns:
            df_pd = df_pd.drop(columns=['symbol'])
            
        # Convert to Polars and standardize columns
        df = pl.from_pandas(df_pd)
        df = df.rename({col: col.lower() for col in df.columns})
        
        # ---------------------------------------------------------------------
        # LOOKAHEAD-BIAS DEFENSE:
        # We strictly exclude the last fetched row. In live quantitative environments,
        # the final candle is still 'forming' and its Close price is an illusion. 
        # Using it causes lookahead bias.
        # ---------------------------------------------------------------------
        df = df.slice(0, df.height - 1)
        
        # Ensure temporal ordering
        return df.sort("datetime")

    def resample_and_store(self, df: pl.DataFrame, symbol: str, timeframes: List[str]):
        """
        Engineers multi-timeframe features via Polars and sinks them to Parquet.
        """
        # Autonomous symbol partition creation
        symbol_dir = self.storage_path / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Sink base 15m data
        base_path = symbol_dir / f"{symbol}_15m.parquet"
        df.write_parquet(base_path)
        logger.info(f"[{symbol}] Sinked base 15m -> {base_path.name} ({df.height} rows)")

        # Generate rolled-up timeframes
        for tf in timeframes:
            logger.info(f"[{symbol}] Polars Resampling -> {tf} timeframe...")
            
            # ---------------------------------------------------------------------
            # MULTI-TIMEFRAME RESAMPLING:
            # We use group_by_dynamic. closed="left" and label="left" are critical
            # to ensure the timestamp of the new bar points to the START of the 
            # period, strictly enforcing temporal boundaries.
            # ---------------------------------------------------------------------
            resampled_df = (
                df.group_by_dynamic("datetime", every=tf, closed="left", label="left")
                .agg([
                    pl.col("open").first().alias("open"),
                    pl.col("high").max().alias("high"),
                    pl.col("low").min().alias("low"),
                    pl.col("close").last().alias("close"),
                    pl.col("volume").sum().alias("volume")
                ])
                .drop_nulls()  # Drop non-trading periods generated by the grouper
            )
            
            tf_path = symbol_dir / f"{symbol}_{tf}.parquet"
            resampled_df.write_parquet(tf_path)
            logger.info(f"[{symbol}] Sinked {tf} -> {tf_path.name} ({resampled_df.height} rows)")

    def run_automation(self, target_assets: List[Dict[str, str]]):
        """
        Master orchestration method for the data pipeline.
        """
        logger.info("Starting Zero-Touch Data Pipeline...")
        success_count = 0

        for asset in target_assets:
            symbol = asset['symbol']
            exchange = asset['exchange']
            try:
                # 1. Extraction & Cleaning
                # max_retries=5 means we'll try up to 5 times (1 initial + 4 retries)
                df_15m = self.fetch_base_data(symbol, exchange, n_bars=10000, max_retries=5, retry_delay=10)
                
                # 2. Multi-Timeframe Transformation & Loading (30m, 60m, 4h, 1d)
                self.resample_and_store(df_15m, symbol, timeframes=["30m", "60m", "4h", "1d"])
                success_count += 1
                
            except Exception as e:
                logger.error(f"Pipeline entirely failed for {symbol}: {str(e)}")
                continue

        logger.info(f"Pipeline Execution Finished. Successfully processed {success_count}/{len(target_assets)} assets.")

# -------------------------------------------------------------------------
# 3. Execution Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Define Target Assets (Taiwanese Equities Example)
    target_equities = [
        {"symbol": "2330", "exchange": "TWSE"},  # TSMC (台積電)
        {"symbol": "2308", "exchange": "TWSE"},  # Delta Electronics (台達電)
        {"symbol": "2454", "exchange": "TWSE"}   # MediaTek (聯發科)
    ]
    
    # Initialize and run
    pipeline = ZeroTouchDataLake(storage_path=DATA_LAKE_PATH)
    pipeline.run_automation(target_equities)
