Phoenix Core: Layer 1 Data Layer - META 工程文件 (v1.3 SOTA 純粹多流版)

系統定位: 鑽石型架構 (Diamond Architecture) 的單一真理樹幹 (Single Source of Truth)
系統版本: v1.3 (SOTA Standard, Pure Multi-Stream, Flat-Table Deprecated)
底層架構: WSL + Polars (Data Engine) + Asyncio (I/O) + Python (Logic)
核心依賴: polars, numpy, ccxt, importlib, pydantic, pyyaml

🏛️ 架構哲學：鑽石型架構的純淨起點

Layer 1 是一個「無狀態、無腦的巨型觀測站」。
【v1.3 核心升級】：徹底廢除 Layer 1 的 Flat Wide Table 縫合邏輯！
它現在絕對不會為下游進行任何跨時間尺度的強制縫合 (Forward-fill 假平行線會破壞 DL 模型的時空認知)。
它的唯一使命是讀取 YAML 設定，將雜亂的市場數據洗淨，並產出「純粹獨立的 5 張時間尺度 Parquet」(1m, 5m, 15m, 1h, 1d)。
所有跨尺度的特徵對齊與剪枝，全部推遲到 Layer 1.5 (特徵審判庭) 的記憶體中進行。

📂 實體目錄樹狀圖 (Directory Tree)

phoenix/
├── infra/                            
│   ├── data_lake/                    
│   │   └── parquet_store/            # 儲存純淨的 Raw OHLCV Parquet
│   └── feature_store/                # 儲存帶有 G-Interface 寬表特徵的獨立 Parquet
└── core/
    └── layer1_data/                  
        ├── schemas.py                # [介面契約] Pydantic 強驗證合約
        ├── pipeline.py               # [引擎] 核心管線 (廢除縫合，強制回傳 Dict[str, DataFrame])
        ├── data_lake_builder.py      # [基礎設施] Asyncio 歷史抓取器
        ├── feature_builder.py        # [基礎設施] 特徵鍛造工廠 (產出純淨多流 Parquet)
        ├── processors/               
        │   ├── aligner.py            # [防爆] (目前供實盤輔助用)
        │   ├── market_calendar.py    # [防爆] 日曆引擎
        │   └── missing_encoder.py    # [防爆] 遺失編碼
        └── extractors/               
            ├── feature_proxy.py      # [SOTA核心] YAML 驅動的特徵路由
            └── time_clock.py         # [SOTA核心] 時間變形與事件時鐘


Phoenix Core: Layer 1 - Data Lake Builder (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/data_lake_builder.py
系統定位: 異步多資產歷史抓取與資料湖建構 (Bronze/Silver 融合)
防爆核心: 寫入物理 close_time 錨點、極端網路退避、智慧增量更新。

📜 絕對架構與依賴契約

- 網路 I/O: 必須使用 ccxt.async_support 與 asyncio。
- 儲存路徑: 必須指向 infra/data_lake/parquet_store/。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

```python
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
```

--------------------------------------------------------------------------------

Phoenix Core: Layer 1 - Feature Builder (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/feature_builder.py
系統定位: 特徵鍛造工廠 (CLI 進入點)
防爆核心: 廢除 Flat Wide Table 分歧，強制獨立輸出 5 尺度 Parquet。

📜 絕對架構與依賴契約

- 必須讀取 data_lake，透過 pipeline 處理後寫入 feature_store。
- 必須支援動態載入系統目錄 (sys.path.insert)。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

```python
import argparse
import polars as pl
import os
import asyncio
from pathlib import Path
import logging
import yaml
import sys
from typing import Dict, Union

# 確保可以直接在目錄下執行
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.layer1_data.schemas import AssetDataConfig
from core.layer1_data.pipeline import DataLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureBuilder")

class FeatureBuilder:
    """
    [Layer 1 - 終極兵器] Feature Store Builder
    負責從 Data Lake 讀取原始 K 線，透過 DataLayer (FeatureProxy + Aligner)
    產出神經網路/樹模型專用的高維度特徵寬表。
    """
    def __init__(self, data_lake_dir: str = None, feature_store_dir: str = None, config_dir: str = None):
        project_root = Path(__file__).resolve().parent.parent.parent
        
        self.data_lake_dir = Path(data_lake_dir) if data_lake_dir else project_root / "infra" / "data_lake" / "parquet_store"
        self.feature_store_dir = Path(feature_store_dir) if feature_store_dir else project_root / "infra" / "feature_store"
        self.config_dir = Path(config_dir) if config_dir else project_root / "configs" / "assets"
        
        self.feature_store_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self, asset: str, base_timeframe: str, macro_timeframes: list) -> tuple:
        """從 Data Lake 載入已經清理過的原始資料"""
        symbol_safe = asset.replace("/", "_")
        asset_lake_dir = self.data_lake_dir / symbol_safe
        
        base_path = asset_lake_dir / f"{symbol_safe}_{base_timeframe}.parquet"
        if not base_path.exists():
            raise FileNotFoundError(f"找不到 Base 資料檔: {base_path}，請先執行 data_lake_builder.py")
            
        base_df = pl.read_parquet(base_path)
        
        macro_dict = {}
        for tf in macro_timeframes:
            macro_path = asset_lake_dir / f"{symbol_safe}_{tf}.parquet"
            if macro_path.exists():
                macro_dict[tf] = pl.read_parquet(macro_path)
            else:
                logger.warning(f"[{asset}] 找不到 Macro 資料檔: {macro_path}，將略過該尺度。")
                
        return base_df, macro_dict

    def process_single_asset(self, asset: str):
        logger.info(f"[*] 開始為 {asset} 建構 Feature Store...")
        
        # 1. 讀取並解析 YAML
        symbol_safe = asset.replace("/", "_")
        config_path = self.config_dir / f"{symbol_safe}.yaml"
        if not config_path.exists():
            logger.error(f"[{asset}] 找不到設定檔 {config_path}")
            return
            
        with open(config_path, "r", encoding="utf-8") as f:
            raw_yaml = yaml.safe_load(f)
            
        config = AssetDataConfig(
            base_asset=asset,
            target_assets=raw_yaml.get("target_assets", {}),
            base_timeframe=raw_yaml.get("base_timeframe", "1m"),
            macro_timeframes=raw_yaml.get("macro_timeframes", ["5m", "15m", "1h", "1d"]),
            warmup_bars=raw_yaml.get("phase1_window", {}).get("warmup_bars", 300),
            raw_yaml=raw_yaml
        )
        
        # 2. 載入 Data Lake 原始資料
        try:
            base_df, macro_dict = self.load_raw_data(asset, config.base_timeframe, config.macro_timeframes)
        except Exception as e:
            logger.error(f"[{asset}] 載入原始資料失敗: {e}")
            return
            
        # 3. 實例化 DataLayer 並進行 Batch 運算
        layer1 = DataLayer(config)
        features_output = layer1.batch(base_df, macro_dict)
        
        # 4. 儲存至 Feature Store (純粹多資料流)
        asset_feature_dir = self.feature_store_dir / symbol_safe
        asset_feature_dir.mkdir(parents=True, exist_ok=True)
        
        # SOTA 修正：Layer 1 全面廢棄 Flat Wide Table，只輸出獨立的 Parquet
        for tf, df in features_output.items():
            out_path = asset_feature_dir / f"{symbol_safe}_features_{tf}.parquet"
            df.write_parquet(out_path)
            logger.info(f"[v] {asset} {tf} 儲存成功: {out_path.name} (寬度: {len(df.columns)}, 筆數: {len(df)})")

    def build_all(self, target_assets: list = None):
        """全量建構"""
        if not target_assets:
            yaml_files = list(self.config_dir.glob("*.yaml"))
            target_assets = []
            for yf in yaml_files:
                try:
                    with open(yf, "r", encoding="utf-8") as f:
                        conf = yaml.safe_load(f)
                        if conf and "base_asset" in conf:
                            target_assets.append(conf["base_asset"])
                except Exception as e:
                    pass
                    
        for asset in target_assets:
            try:
                self.process_single_asset(asset)
            except Exception as e:
                logger.error(f"[{asset}] Feature Store 建構發生未預期錯誤: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phoenix Feature Store Builder (Layer 1 Final)")
    parser.add_argument("--assets", nargs="+", help="指定要計算特徵的商品 (例如 BTC/USDT)。未提供則全掃描。")
    args = parser.parse_args()

    builder = FeatureBuilder()
    builder.build_all(target_assets=args.assets)

```

--------------------------------------------------------------------------------

Phoenix Core: Layer 1 - Pipeline (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/pipeline.py
系統定位: ⚡ SOTA 協調器 (DataLayer Engine)
防爆核心: 純粹多流字典輸出 (Dict[str, DataFrame])，廢棄 Backward Join 縫合。

📜 絕對架構與依賴契約

- 必須嚴格執行: Encoder -> FeatureProxy -> TimeClock。
- batch() 與 step() 必須回傳多流架構。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

```python
# phoenix/core/layer1_data/pipeline.py

import polars as pl
from typing import Dict, Any, Union
import logging

# 🚨 SOTA 升級：全面改用絕對路徑，根絕 ImportError
from core.layer1_data.schemas import MarketObservation, AssetDataConfig
from core.layer1_data.processors.missing_encoder import MissingnessEncoder
from core.layer1_data.processors.aligner import TimeframeAligner
from core.layer1_data.processors.market_calendar import MarketCalendar
from core.layer1_data.extractors.time_clock import TimeClockExtractor
from core.layer1_data.extractors.feature_proxy import FeatureProxy 

logger = logging.getLogger("DataLayer")

class DataLayer:
    """
    [Layer 1 - ⚡ SOTA 協調器]
    負責將原始數據轉化為 MarketObservation，為下游神經網路提供物理一致的數據視角。
    嚴格區分 streaming (實盤逐筆) 與 batch (回測向量化) 雙引擎架構。
    """
    def __init__(self, config: AssetDataConfig):
        self.config = config
        
        # 100分級實例化約束
        self.encoder = MissingnessEncoder()
        self.aligner = TimeframeAligner()
        self.calendar = MarketCalendar()
        self.clock = TimeClockExtractor()
        
        # [防爆紅線] 傳入解析後的 YAML 字典，由 FeatureProxy 動態決定要掛載哪些指標
        self.feature_proxy = FeatureProxy(config.raw_yaml)

    def step(self, base_window: pl.DataFrame, macro_data: Dict[str, pl.DataFrame]) -> MarketObservation:
        """
        [實盤引擎 / Streaming] 全局心跳觸發
        防爆設計：實盤推論時，傳入的 K 線視窗長度必須大於等於 warmup_bars。
        """
        # 🚨 [防爆紅線] 強契約暖機斷言 (Warm-up Assertion)
        current_len = len(base_window)
        if current_len < self.config.warmup_bars:
            raise RuntimeError(
                f"[Layer 1 暖機未完成] 當前視窗長度 {current_len}，"
                f"低於 YAML 要求的最低暖機根數 {self.config.warmup_bars}！請持續累積資料。"
            )

        # 1. 遺失編碼 (Value, Mask, Reason)
        obs_df = self.encoder.encode(base_window)
        
        # 2. 特徵代理注入 (Base Timeframe)
        obs_df = self.feature_proxy.extract_features(obs_df, self.config.base_timeframe)
        
        # 3. 時間語意萃取 (d_tau 與 事件時鐘)
        obs_df = self.clock.extract(obs_df)
        
        # 4. Macro Timeframes 特徵注入
        processed_macro = {}
        for tf, m_df in macro_data.items():
            processed_macro[tf] = self.feature_proxy.extract_features(m_df, tf)
            
        # [SOTA 架構] 廢棄 Flat Wide Table，全面採用多資料流輸出
        latest_obs = obs_df.tail(1)
        
        features_val_dict = {self.config.base_timeframe: latest_obs.select(pl.exclude(["features_mask", "features_reason", "time_delta_tau", "minutes_since_open_norm", "minutes_until_close_norm"]))}
        features_mask_dict = {self.config.base_timeframe: latest_obs.select(["features_mask"])}
        features_reason_dict = {self.config.base_timeframe: latest_obs.select(["features_reason"])}
        time_features_dict = {self.config.base_timeframe: latest_obs.select(["time_delta_tau", "minutes_since_open_norm", "minutes_until_close_norm"])}
        
        for tf, m_df in processed_macro.items():
            m_latest = m_df.tail(1)
            features_val_dict[tf] = m_latest.select(pl.exclude(["features_mask", "features_reason"]))
            features_mask_dict[tf] = m_latest.select(["features_mask"])
            features_reason_dict[tf] = m_latest.select(["features_reason"])
        
        return MarketObservation(
            timestamp=int(latest_obs["datetime"][0].timestamp() * 1000),
            asset_id=self.config.base_asset,
            features_value=features_val_dict,
            features_mask=features_mask_dict,
            features_reason=features_reason_dict,
            time_features=time_features_dict
        )

    def batch(self, base_history: pl.DataFrame, macro_history: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        """
        [回測引擎 / Batch] 離線向量化處理
        利用 Polars Lazy Evaluation 榨乾 CPU 效能，並物理性切除不穩定的暖機資料。
        """
        logger.info(f"[*] 啟動 DataLayer Batch Engine: 處理大規模歷史回測資料...")
        
        # 1. 大規模遺失編碼
        df = self.encoder.encode(base_history)
        
        # 2. Base Timeframe 特徵代理注入
        df = self.feature_proxy.extract_features(df, self.config.base_timeframe)
        
        # 3. 向量化時間特徵萃取
        df = self.clock.extract(df)
        
        # 4. Macro Timeframes 特徵代理注入
        processed_macro = {}
        for tf, m_df in macro_history.items():
            processed_macro[tf] = self.feature_proxy.extract_features(m_df, tf)
            
        # [SOTA 架構] 廢棄 Flat Wide Table，全面維持多資料流字典
        results = {self.config.base_timeframe: df}
        for tf, m_df in processed_macro.items():
            results[tf] = m_df
            
        # 🚨 [防爆紅線] 各尺度切除暖機期 (Warm-up Slicing)
        # 前面 N 根 K 線的 EMA/RSI 等指標處於未收斂狀態 (包含 FFill 的假資料或 NaN)，
        # 必須強制切除，確保送入神經網路的 Batch 張量處於完美的滿秩狀態。
        for tf in results.keys():
            if len(results[tf]) > self.config.warmup_bars:
                results[tf] = results[tf].slice(self.config.warmup_bars)
            else:
                results[tf] = pl.DataFrame(schema=results[tf].schema)
                
        logger.info(f"[v] Batch Engine DL 雙軌輸出完畢，獨立保留 {len(results)} 個時間尺度特徵表。")
        return results
```

--------------------------------------------------------------------------------

Phoenix Core: Layer 1 - Schemas (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/schemas.py
系統定位: 100分級 Pydantic 強驗證合約
防爆核心: 跨層通訊的絕對真理，支援 Dict[str, pl.DataFrame] 的 Multi-Stream 結構。

📜 絕對架構與依賴契約

- 必須允許 Union[pl.DataFrame, Dict[str, pl.DataFrame]] 傳遞。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

```python
import polars as pl
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Literal, Any, Union

class AssetMetadata(BaseModel):
    """
    [Layer 1] 資產物理與制度描述 (Asset Semantics)
    提供給下游模型學習不同資產的物理特性 (如時區差異、跳動單位)
    """
    asset_id: str = Field(..., description="標的代碼 (e.g., 'BTC/USDT', 'TX00')")
    asset_type: Literal["crypto", "equity_index", "commodity", "forex"] = Field(
        ..., description="資產大類 (決定了模型對待其休市與假日的態度)"
    )
    timezone: str = Field(..., description="交易所時區 (e.g., 'UTC', 'America/New_York')")
    tick_size: float = Field(..., gt=0, description="最小跳動單位 (必須大於 0)")

class AssetDataConfig(BaseModel):
    """
    [Layer 1] 單一商品超參數配置契約 (MCP 級解耦)
    定義該商品專屬的物理特徵、多寬表尺度與出戰特徵。
    """
    base_asset: str = Field(..., description="基準資產 (e.g., 'BTC/USDT')")
    target_assets: Dict[str, AssetMetadata] = Field(..., description="全市場標的註冊表 (包含基準資產與其他需要跨商品關聯的資產)")
    
    base_timeframe: str = Field(
        default="1m",
        description="本商品在資料湖中最底層的物理 K 線尺度 (e.g. '1m', '5m')"
    )
    macro_timeframes: List[str] = Field(
        default=["5m", "15m", "1h", "1d"], 
        description="需進行 Backward Join 縫合的巨觀時間框架"
    )
    
    # 🚨 [SOTA 升級] 物理暖機與動態 YAML 路由合約
    warmup_bars: int = Field(
        default=300, 
        description="系統啟動或回測時的強制暖機 K 線數量，防範 EMA/MACD 等指標初始發散"
    )
    raw_yaml: Dict[str, Any] = Field(
        default_factory=dict, 
        description="解析後的完整 YAML 配置字典，供 FeatureProxy 進行動態特徵註冊與參數覆寫"
    )

    @field_validator("target_assets")
    @classmethod
    def check_base_asset_in_targets(cls, target_assets: Dict[str, AssetMetadata], info: Any) -> Dict[str, AssetMetadata]:
        """防呆驗證：基準資產必須存在於註冊表中"""
        # 兼容 Pydantic v2 的 ValidationInfo
        base_asset = info.data.get("base_asset") if hasattr(info, "data") else None
        if base_asset and base_asset not in target_assets:
            raise ValueError(f"基準資產 {base_asset} 未在 target_assets 註冊表中定義！")
        return target_assets

class MarketObservation(BaseModel):
    """
    [Layer 1] 核心輸出契約 (傳遞給 Layer 2 的標準介面)
    確保進入神經網路的數據具備完美的物理封裝與防爆遮罩。
    """
    timestamp: int = Field(..., description="全局基準 Heartbeat (Unix Timestamp 秒/毫秒)")
    asset_id: str = Field(..., description="資料所屬標的")
    
    # --- 三維遺失編碼 (Missingness Encoding) ---
    features_value: Union[pl.DataFrame, Dict[str, pl.DataFrame]] = Field(..., description="實際數值或 ffill 後的連續數值矩陣 (若為深度學習模式，則為各尺度獨立之字典)")
    features_mask: Union[pl.DataFrame, Dict[str, pl.DataFrame]] = Field(..., description="Boolean: True 表示該 T 刻有真實交易，False 為補值")
    features_reason: Union[pl.DataFrame, Dict[str, pl.DataFrame]] = Field(..., description="IntEnum: 0=正常, 1=休市, 2=熔斷, 3=API斷線")
    
    # --- 時間語意特徵 (Event Clock & Time Deformation) ---
    time_features: Union[pl.DataFrame, Dict[str, pl.DataFrame]] = Field(..., description="包含 time_delta_tau, minutes_since_open 等")

    class Config:
        # 允許 Pydantic 裝載 Polars DataFrame 這種非標準 Python 類型
        arbitrary_types_allowed = True

    @field_validator("features_mask", "features_reason", "time_features")
    @classmethod
    def check_dataframe_length_match(cls, df: pl.DataFrame, info: Any) -> pl.DataFrame:
        """矩陣形狀防呆驗證：所有特徵維度的長度必須嚴格對齊"""
        if not hasattr(info, "data"): return df
        
        # 取得 features_value 作為長度基準
        base_df = info.data.get("features_value")
        if isinstance(base_df, pl.DataFrame) and len(df) != len(base_df):
            field_name = info.field_name if hasattr(info, "field_name") else "unknown"
            raise ValueError(f"維度對齊錯誤！{field_name} 的長度 ({len(df)}) 與 features_value ({len(base_df)}) 不一致。")
        return df
```

--------------------------------------------------------------------------------

Phoenix Core: Layer 1 - Missing Encoder (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/processors/missing_encoder.py
系統定位: ⚡ 防爆核心 (3D Missingness Encoding)
防爆核心: 空值轉化物理狀態碼 (Mask, Reason)，極限 NaN 阻擋。

📜 絕對架構與依賴契約

- 必須使用 IntEnum。
- 必須執行雙層 fill_null 保證零 NaN。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

```python
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
```

--------------------------------------------------------------------------------

Phoenix Core: Layer 1 - Aligner (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/processors/aligner.py
系統定位: ⚡ 歷史對齊器 (Backward Join)
防爆核心: 結束時間錨定 (End-Time Indexing)。

📜 絕對架構與依賴契約

- 雖然 Layer 1 廢棄縫合，但保留此工具供 Layer 1.5 記憶體縫合使用。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

```python
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
```

--------------------------------------------------------------------------------

Phoenix Core: Layer 1 - Market Calendar (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/processors/market_calendar.py
系統定位: ⚡ 市場日曆
防爆核心: 判斷市場開關狀態。

📜 絕對架構與依賴契約

- 提供 is_market_open 介面。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

```python
import polars as pl
from datetime import datetime

class MarketCalendar:
    """
    [Layer 1 - 防爆核心] 市場日曆引擎
    職責：針對不同資產的交易所規則進行開收盤判斷。
    """
    def is_market_open(self, asset_id: str, timestamp: datetime) -> bool:
        # 此處預留介面與 exchange_calendars 整合
        # META 要求：必須明確區分 [休市] 與 [API 斷線]
        return True # 簡化版本
```

--------------------------------------------------------------------------------

Phoenix Core: Layer 1 - Feature Proxy (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/extractors/feature_proxy.py
系統定位: ⚡ YAML 驅動特徵路由
防爆核心: 動態映射，並支援 Timeframe 專屬參數。

📜 絕對架構與依賴契約

- 必須從 YAML 讀取 indicator_default_params[tf]。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

```python
import polars as pl
import importlib
import logging
from typing import Dict, Any, List

logger = logging.getLogger("FeatureProxy")

class FeatureProxy:
    """
    [Layer 1 - ⚡ SOTA 中樞] 動態特徵路由代理
    職責：讀取 YAML 出戰名單，動態映射並載入 features_library 中的指標。
    防爆機制：Fail-fast 預載入、Namespace 碰撞隔離、單次圖優化計算。
    """
    def __init__(self, config_dict: Dict[str, Any]):
        self.plan = config_dict.get("meta_features_plan", [])
        self.defaults = config_dict.get("indicator_default_params", {})
        self.module_cache = {}
        
        # [Fail-Fast 紅線] 系統啟動時立刻驗證並預載入所有模組，防止實盤中 I/O 阻塞或崩潰
        self._preload_modules()

    def _preload_modules(self):
        """預先編譯並快取所需的特徵腳本"""
        logger.info(f"[*] FeatureProxy 開始預載入特徵模組 (共 {len(self.plan)} 項任務)...")
        for item in self.plan:
            indicator_name = item["indicator"]
            if indicator_name not in self.module_cache:
                try:
                    # 動態載入您規劃的特徵軍火庫路徑
                    module_path = f"core.layer1_data.features_library.Indicators.{indicator_name}"
                    mod = importlib.import_module(module_path)
                    
                    # 確保 H 接口與 G 接口存在
                    if not hasattr(mod, "calculate") or not hasattr(mod, f"adapt_{indicator_name}"):
                        raise AttributeError(f"模組 {indicator_name} 缺乏標準 H/G 接口契約。")
                        
                    self.module_cache[indicator_name] = mod
                    logger.debug(f"[+] 成功載入特徵模組: {indicator_name}")
                except ImportError as e:
                    logger.error(f"[!] 找不到特徵模組: {indicator_name}。請檢查 YAML 與檔案路徑。")
                    raise e

    def extract_features(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        """
        [多核極限壓榨] 遍歷 YAML 出戰名單，打包所有 Lazy Expressions 並一次性執行。
        """
        exprs: List[pl.Expr] = []
        close_expr = pl.col("close")
        
        for item in self.plan:
            uid = item["id"]  # 來自 YAML 的 Unique ID (e.g., 'MACD_Fast')
            indicator_name = item["indicator"]
            mod = self.module_cache[indicator_name]
            
            # 1. 參數合併 (YAML 客製化參數 覆蓋 預設參數)
            # [SOTA 升級] 動態依據 timeframe 讀取專屬參數
            tf_defaults = self.defaults.get(timeframe, {})
            if tf_defaults:
                params = tf_defaults.get(indicator_name, {}).copy()
            else:
                # 兼容舊版全域配置
                params = self.defaults.get(indicator_name, {}).copy()
                
            if "params" in item:
                params.update(item["params"])
                
            # 2. 呼叫 H 接口 (人類與策略語意)
            # 傳入 df 是因為部分指標 (如 SMC, ATR) 需要 high/low 進行 Pivot 計算
            h_output = mod.calculate(df=df, params=params)
            
            # 3. 呼叫 G 接口 (深度學習寬表特徵映射)
            adapt_func = getattr(mod, f"adapt_{indicator_name}")
            g_output: Dict[str, pl.Expr] = adapt_func(h_output=h_output, close=close_expr, params=params)
            
            # 4. [防碰撞紅線] 強制重命名
            # 為了允許 YAML 呼叫多次相同的指標 (如 MACD_15m, MACD_1d)，我們必須加上 UID 前綴
            for feat_name, expr in g_output.items():
                safe_alias = f"{uid}_{feat_name}"
                exprs.append(expr.alias(safe_alias))
                
        # 5. [效能紅線] 一次性提交給 Polars 底層 Rust 引擎進行並行圖優化 (DAG)
        if exprs:
            df = df.with_columns(exprs)
            
        return df
```

--------------------------------------------------------------------------------

Phoenix Core: Layer 1 - Time Clock (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/extractors/time_clock.py
系統定位: ⚡ 時間變形萃取
防爆核心: 萃取物理時間非線性流逝率 (d_tau)。

📜 絕對架構與依賴契約

- 計算時必須基於 features_mask 進行防禦。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

```python
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
```

--------------------------------------------------------------------------------

