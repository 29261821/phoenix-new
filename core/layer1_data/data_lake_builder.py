import ccxt.async_support as ccxt_async
import polars as pl
import os
import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, List
from dotenv import load_dotenv
from pathlib import Path
import logging
import yaml
import argparse
import sys

# 確保可以直接在目錄下執行
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# SOTA 升級: 動態在地時區解析
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# 載入 .env 檔案中的環境變數
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataLakeBuilder")

class DataLakeBuilder:
    def __init__(self, data_dir: Optional[str] = None, config_dir: Optional[str] = None):
        """
        初始化非同步資料湖建構器
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        
        if data_dir is None:
            self.data_dir = project_root / "infra" / "data_lake" / "parquet_store"
        else:
            self.data_dir = Path(data_dir)
            
        if config_dir is None:
            self.config_dir = project_root / "configs" / "assets"
        else:
            self.config_dir = Path(config_dir)
            
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

    def load_asset_config(self, asset: str) -> dict:
        symbol_safe = asset.replace("/", "_")
        config_path = self.config_dir / f"{symbol_safe}.yaml"
        if not config_path.exists():
            logger.warning(f"[{asset}] 找不到設定檔 {config_path}，使用系統預設 1m 基準。")
            return {"base_timeframe": "1m", "macro_timeframes": ["5m", "15m", "1h", "1d"]}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_fetch_start_time(self, asset: str, config: dict, force_rebuild: bool) -> str:
        """
        [SOTA 升級] 智慧增量時間錨定
        若存在本地檔案，讀取最後時間點進行增量更新。
        若無，解析 YAML 中指定的起點，並精確根據商品當地時區 (Session) 轉換為 UTC。
        """
        symbol_safe = asset.replace("/", "_")
        base_timeframe = config.get("base_timeframe", "1m")
        file_path = self.data_dir / symbol_safe / f"{symbol_safe}_{base_timeframe}.parquet"
        
        # 1. 取得商品專屬時區 (Fallbacks to UTC)
        tz_str = "UTC"
        if "target_assets" in config and asset in config["target_assets"]:
            tz_str = config["target_assets"][asset].get("timezone", "UTC")
        try:
            tz = ZoneInfo(tz_str)
        except Exception:
            logger.warning(f"[{asset}] 時區 {tz_str} 無效，退回 UTC。")
            tz = timezone.utc
            
        # 2. 增量更新邏輯
        if not force_rebuild and file_path.exists():
            try:
                df_last = pl.read_parquet(file_path, columns=["datetime"])
                last_dt = df_last["datetime"].max()
                
                # 將 Naive DateTime (從 epoch 轉換來) 視為 UTC，並格式化為 CCXT 接受的 ISO8601
                last_dt_utc = last_dt.replace(tzinfo=timezone.utc)
                iso_start = last_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                logger.info(f"[{asset}] 偵測到既有資料，啟動增量更新，起點: {iso_start}")
                return iso_start
            except Exception as e:
                logger.warning(f"[{asset}] 讀取既有 Parquet 失敗 ({e})，轉為全量更新。")
        
        # 3. 首次建構邏輯 (Cold Start)
        train_start_str = "2024-01-01 00:00:00"
        if "dataset_boundaries" in config and "train_start" in config["dataset_boundaries"]:
            train_start_str = config["dataset_boundaries"]["train_start"]
            
        # 套用當地時區 (Local Time -> UTC)
        dt_local = datetime.strptime(train_start_str, "%Y-%m-%d %H:%M:%S")
        dt_local = dt_local.replace(tzinfo=tz)
        dt_utc = dt_local.astimezone(timezone.utc)
        
        iso_start = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info(f"[{asset}] 首次建構 (或強制重建)，起點: {train_start_str} ({tz_str}) -> {iso_start} (UTC)")
        return iso_start

    async def fetch_historical_base_async(self, asset: str, start_utc: str, end_utc: str, base_timeframe: str) -> pl.DataFrame:
        """
        [SOTA 非同步抓取] 具備指數退避重試的極速歷史拉取
        """
        logger.info(f"[*] 開始抓取 {asset} {base_timeframe} 數據 ({start_utc} to {end_utc})...")
        
        since = self.exchange.parse8601(start_utc)
        until = self.exchange.parse8601(end_utc)
        
        all_ohlcv = []
        current_since = since
        max_retries = 5
        
        tf_ms = self.exchange.parse_timeframe(base_timeframe) * 1000
        
        while current_since < until:
            retries = 0
            while retries < max_retries:
                try:
                    # 如果該資產在幣安不存在，這裡會拋出 ccxt.BadSymbol，將由外層 catch 處理
                    ohlcv = await self.exchange.fetch_ohlcv(asset, base_timeframe, since=current_since, limit=1000)
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    current_since = ohlcv[-1][0] + tf_ms 
                    
                    await asyncio.sleep(self.exchange.rateLimit / 1000)
                    break
                    
                except ccxt_async.RateLimitExceeded:
                    wait_time = 2 ** retries
                    logger.warning(f"[{asset}] Rate Limit! 等待 {wait_time} 秒...")
                    await asyncio.sleep(wait_time)
                    retries += 1
                except ccxt_async.NetworkError as e:
                    logger.warning(f"[{asset}] 網路異常: {e}，重試中...")
                    await asyncio.sleep(2 ** retries)
                    retries += 1
                except ccxt_async.BadSymbol as e:
                    logger.error(f"[{asset}] 交易對不存在或不支援: {e}")
                    raise e
                except Exception as e:
                    logger.error(f"[{asset}] 致命錯誤: {e}")
                    raise e
                    
            if retries == max_retries:
                logger.error(f"[{asset}] 放棄抓取：達到最大重試次數。")
                break
                
        df = pl.DataFrame(all_ohlcv, schema=["timestamp", "open", "high", "low", "close", "volume"], orient="row")
        return df.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"))

    def apply_missingness_encoding(self, df: pl.DataFrame, base_timeframe: str) -> pl.DataFrame:
        """三維遺失編碼器 (動態 Base Timeframe)"""
        if len(df) == 0:
            return df
            
        start_dt = df["datetime"].min()
        end_dt = df["datetime"].max()
        
        continuous_time = pl.DataFrame({
            "datetime": pl.datetime_range(start_dt, end_dt, base_timeframe, eager=True).cast(pl.Datetime("ms"))
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

    def generate_multi_timeframes(self, df_base: pl.DataFrame, base_timeframe: str, macro_timeframes: List[str]) -> Dict[str, pl.DataFrame]:
        """
        [SOTA 升級] 多時間尺度重採樣，並強制寫入物理結束時間 (Close Time)
        """
        # 1. 賦予基準 K 線 close_time
        df_base = df_base.with_columns(
            (pl.col("datetime").dt.offset_by(base_timeframe) - pl.duration(milliseconds=1)).alias("close_time")
        )
        results = {base_timeframe: df_base}
        
        for tf in macro_timeframes:
            df_tf = df_base.group_by_dynamic("datetime", every=tf).agg([
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                pl.col("features_mask").any(),
                pl.col("features_reason").max() 
            ])
            
            # 2. 賦予宏觀 K 線 close_time
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

    async def process_single_asset(self, asset: str, force_rebuild: bool = False):
        """單一資產的完整生命週期管線"""
        config = self.load_asset_config(asset)
        base_timeframe = config.get("base_timeframe", "1m")
        macro_timeframes = config.get("macro_timeframes", ["5m", "15m", "1h", "1d"])

        start_utc = self.get_fetch_start_time(asset, config, force_rebuild)
        end_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # 1. 網路 I/O: 抓取最新原始資料
        raw_base_df = await self.fetch_historical_base_async(asset, start_utc, end_utc, base_timeframe)
        
        if len(raw_base_df) == 0:
            logger.info(f"[{asset}] 已是最新，無新資料可更新。")
            return

        # 🚨 [SOTA 防爆機制] 剔除未收盤 K 線
        # 防止未來視與殘缺的特徵破壞神經網路的收斂
        current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        tf_ms = self.exchange.parse_timeframe(base_timeframe) * 1000
        raw_base_df = raw_base_df.filter((pl.col("timestamp") + tf_ms) <= current_time_ms)
        
        if len(raw_base_df) == 0:
            logger.info(f"[{asset}] 抓取到的最新 K 線皆未物理收盤，本次無靜態更新。")
            return

        # 2. CPU 密集型: 缺失編碼
        encoded_new_df = self.apply_missingness_encoding(raw_base_df, base_timeframe)
        
        # 3. 增量縫合邏輯
        symbol_safe = asset.replace("/", "_")
        file_path = self.data_dir / symbol_safe / f"{symbol_safe}_{base_timeframe}.parquet"
        
        if not force_rebuild and file_path.exists():
            old_df = pl.read_parquet(file_path)
            if "close_time" in old_df.columns:
                old_df = old_df.drop("close_time")
            # vstack 並透過 datetime 嚴格去重，保留最新 (last) 抓取的修正資料
            encoded_base_df = old_df.vstack(encoded_new_df).unique(subset=["datetime"], keep="last").sort("datetime")
            logger.info(f"[{asset}] 增量縫合完畢，總資料筆數增長至: {len(encoded_base_df)}")
        else:
            encoded_base_df = encoded_new_df

        # 4. 產生多尺度 (Polars 極速全量重採樣)
        multi_tf_datasets = self.generate_multi_timeframes(encoded_base_df, base_timeframe, macro_timeframes)
        
        # 5. 磁碟 I/O: 儲存
        self.save_to_datalake(asset, multi_tf_datasets)

    async def build_multi_asset_lake(self, assets: List[str], force_rebuild: bool = False):
        """
        [多核極限] 同時併發下載與處理多檔資產，包含 MCP 級解耦與錯誤隔離
        """
        logger.info(f"[🚀] 啟動多資產非同步資料湖建構管線: {assets}")
        
        # 使用 asyncio.gather(return_exceptions=True)
        # 確保單一商品 (如非幣安商品或下架代幣) 引發例外時，不會中斷其他商品的下載
        try:
            tasks = [self.process_single_asset(asset, force_rebuild) for asset in assets]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for asset, res in zip(assets, results):
                if isinstance(res, Exception):
                    logger.error(f"[❌] {asset} 處理失敗，已被隔離。原因: {res}")
        finally:
            await self.close()
            
        logger.info("[🏁] 所有資產資料湖建構/更新流程完畢！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phoenix Data Lake Builder (SOTA CLI)")
    parser.add_argument("--assets", nargs="+", help="指定要抓取的商品清單 (例如 BTC/USDT ETH/USDT)。若未提供，則自動掃描 configs/assets 下所有 YAML。")
    parser.add_argument("--force-rebuild", action="store_true", help="強制重新抓取 (忽略既有的 Parquet)，進行 Cold Start。")
    args = parser.parse_args()

    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    builder = DataLakeBuilder()
    
    if args.assets:
        target_assets = args.assets
    else:
        # 動態掃描 configs/assets/*.yaml
        yaml_files = list(builder.config_dir.glob("*.yaml"))
        target_assets = []
        for yf in yaml_files:
            try:
                with open(yf, "r", encoding="utf-8") as f:
                    conf = yaml.safe_load(f)
                    if conf and "base_asset" in conf:
                        target_assets.append(conf["base_asset"])
            except Exception as e:
                logger.warning(f"[!] 無法解析 {yf.name}: {e}")
                
        if not target_assets:
            logger.error("[!] 未在 configs/assets 找到任何有效的 YAML 設定檔，系統退出。")
            sys.exit(1)
            
    # 啟動非同步事件迴圈
    asyncio.run(builder.build_multi_asset_lake(target_assets, force_rebuild=args.force_rebuild))