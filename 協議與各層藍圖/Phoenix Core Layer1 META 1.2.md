Phoenix Core: Layer 1 Data Layer - META 工程文件 (v2.0-Pro 終極版)

系統定位: 鑽石型架構 (Diamond Architecture) 的單一真理樹幹 (Single Source of Truth)
系統版本: v2.0-Pro (100-Point SOTA Standard, Multi-Asset Async Enabled)
底層架構: WSL + Polars (Data Engine) + Asyncio (I/O) + Python (Logic)
核心依賴: polars, numpy, ccxt, importlib, pydantic
上游輸入: 幣安 REST API、本地 Parquet Data Lake
下游輸出: MarketObservation (具備絕對物理一致性、無未來視的張量化前置特徵)

🏛️ 架構哲學：鑽石型架構的起點

Layer 1 是一個「無狀態、無腦的巨型觀測站」。它絕對不為下游的特定模型 (如 LSTM, TCN, Mamba) 進行妥協或維護狀態。它的唯一使命是讀取 YAML 設定，將雜亂的市場數據洗淨、對齊、並產出包含 $N$ 維特徵、布林遮罩、遺失原因與時間變形係數的完整寬表。所有四個世代的模型，都必須以此為唯一真理進行分流訓練。

📂 實體目錄樹狀圖 (Directory Tree)

(備註：已廢除靜態硬編碼的 momentum_proxy.py，全面升級為動態路由的 feature_proxy.py)

phoenix/
├── infra/                            
│   └── data_lake/                    
│       └── parquet_store/            # 儲存帶有 close_time 的多尺度三維 .parquet 檔案
└── core/
    └── layer1_data/                  
        ├── __init__.py               
        ├── schemas.py                # [介面契約] 100分級 Pydantic 強驗證合約 (含暖機參數)
        ├── pipeline.py               # [協調器] 雙引擎 (Streaming/Batch) 數據管線 (含暖機切除)
        ├── data_lake_builder.py      # [基礎設施] Asyncio 多資產併發抓取器 (寫入 close_time)
        ├── fetchers/                 
        │   ├── binance_client.py     # 指數退避 REST 歷史拉取 (實盤 WebSocket 暫緩實作)
        │   └── local_client.py       # Parquet 高速讀取介面
        ├── processors/               
        │   ├── aligner.py            # [防爆] 基於 close_time 的絕對零未來視 Backward Join
        │   ├── market_calendar.py    # [防爆] 交易所日曆引擎
        │   └── missing_encoder.py    # [防爆] 三維遺失編碼 (包含極限 NaN 防護)
        └── extractors/               
            ├── feature_proxy.py      # [SOTA核心] YAML 驅動的動態特徵路由代理 (Namespace 隔離)
            ├── micro_proxy.py        # [預留] Rust FFI 微觀結構分析
            └── time_clock.py         # [SOTA] 事件時鐘與時間變形 ($d\tau$) 萃取

Phoenix Core: Layer 1 - Data Lake Builder (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/data_lake_builder.py
系統定位: 異步多資產歷史抓取與資料湖建構 (Bronze/Silver 融合)
防爆核心: 寫入物理 close_time 錨點、極端網路退避、三維遺失編碼。

📜 絕對架構與依賴契約

網路 I/O: 必須使用 ccxt.async_support 與 asyncio。

資料引擎: 必須使用 polars 進行 DataFrame 操作。

儲存路徑: 必須指向 phoenix/infra/data_lake/parquet_store/{symbol_safe}/。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

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


Phoenix Core: Layer 1 - Pipeline (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/pipeline.py
系統定位: ⚡ SOTA 協調器 (Streaming / Batch 雙引擎)
防爆核心: 絕對路徑引入隔離、warmup_bars 強制斷言 (防污染)、執行順序鎖死。

📜 絕對架構與依賴契約

絕對路徑規範: 必須使用 core.layer1_data... 進行內部模組呼叫。

類別介面: DataLayer(config: DataConfig)

核心方法: step() 供實盤，batch() 供回測。兩者執行順序必須為：Encoder -> FeatureProxy -> Clock -> Aligner。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

# phoenix/core/layer1_data/pipeline.py

import polars as pl
from typing import Dict, Any
import logging

# 🚨 SOTA 升級：全面改用絕對路徑，根絕 ImportError
from core.layer1_data.schemas import MarketObservation, DataConfig
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
    def __init__(self, config: DataConfig):
        self.config = config
        
        # 100分級實例化約束
        self.encoder = MissingnessEncoder()
        self.aligner = TimeframeAligner()
        self.calendar = MarketCalendar()
        self.clock = TimeClockExtractor()
        
        # [防爆紅線] 傳入解析後的 YAML 字典，由 FeatureProxy 動態決定要掛載哪些指標
        self.feature_proxy = FeatureProxy(config.raw_yaml)

    def step(self, raw_1m_window: pl.DataFrame, macro_data: Dict[str, pl.DataFrame]) -> MarketObservation:
        """
        [實盤引擎 / Streaming] 全局心跳觸發
        防爆設計：實盤推論時，傳入的 K 線視窗長度必須大於等於 warmup_bars。
        """
        # 🚨 [防爆紅線] 強契約暖機斷言 (Warm-up Assertion)
        # 如果 WebSocket 剛連線，資料還沒收集滿，絕對禁止呼叫模型進行推論！
        current_len = len(raw_1m_window)
        if current_len < self.config.warmup_bars:
            raise RuntimeError(
                f"[Layer 1 暖機未完成] 當前視窗長度 {current_len}，"
                f"低於 YAML 要求的最低暖機根數 {self.config.warmup_bars}！請持續累積資料。"
            )

        # 1. 遺失編碼 (Value, Mask, Reason)
        obs_df = self.encoder.encode(raw_1m_window)
        
        # 2. 特徵代理注入 (YAML 驅動的動能與微觀特徵)
        obs_df = self.feature_proxy.extract_features(obs_df)
        
        # 3. 時間語意萃取 (d_tau 與 事件時鐘)
        obs_df = self.clock.extract(obs_df)
        
        # 4. 截取最新一筆狀態 (T時刻) 進行多尺度縫合，極大化節省算力
        latest_obs = obs_df.tail(1)
        final_df = self.aligner.align_timeframes(latest_obs, macro_data)
        
        # 5. 封裝為標準契約輸出給 Layer 2
        return MarketObservation(
            timestamp=int(final_df["datetime"][0].timestamp() * 1000), # 確保精確至毫秒
            asset_id=self.config.base_asset,
            features_value=final_df.select(pl.exclude(["features_mask", "features_reason"])),
            features_mask=final_df.select(["features_mask"]),
            features_reason=final_df.select(["features_reason"]),
            time_features=final_df.select(["time_delta_tau", "minutes_since_open_norm", "minutes_until_close_norm"])
        )

    def batch(self, base_1m_history: pl.DataFrame, macro_history: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """
        [回測引擎 / Batch] 離線向量化處理
        利用 Polars Lazy Evaluation 榨乾 CPU 效能，並物理性切除不穩定的暖機資料。
        """
        logger.info(f"[*] 啟動 DataLayer Batch Engine: 處理大規模歷史回測資料...")
        
        # 1. 大規模遺失編碼
        df = self.encoder.encode(base_1m_history)
        
        # 2. 高速向量化特徵代理注入 (無迴圈)
        df = self.feature_proxy.extract_features(df)
        
        # 3. 向量化時間特徵萃取
        df = self.clock.extract(df)
        
        # 4. 向量化歷史多尺度縫合 (Backward Join 保證無前視偏差)
        final_df = self.aligner.align_timeframes(df, macro_history)
        
        # 🚨 [防爆紅線] 暖機期資料切除 (Warm-up Slicing)
        # 前面 N 根 K 線的 EMA/RSI 等指標處於未收斂狀態 (包含 FFill 的假資料或 NaN)，
        # 必須強制切除，確保送入神經網路的 Batch 張量處於完美的滿秩狀態。
        initial_len = len(final_df)
        if initial_len > self.config.warmup_bars:
            final_df = final_df.slice(self.config.warmup_bars)
            logger.info(f"[v] 已切除前 {self.config.warmup_bars} 筆暖機雜訊。")
        else:
            logger.warning(f"[!] 歷史資料總長 {initial_len} 不足暖機長度 {self.config.warmup_bars}，將回傳空矩陣。")
            return pl.DataFrame(schema=final_df.schema)
        
        logger.info(f"[v] Batch Engine 處理完畢，產出矩陣特徵數: {len(final_df.columns)}, 總有效資料筆數: {len(final_df)}")
        return final_df


Phoenix Core: Layer 1 - Schemas (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/schemas.py
系統定位: 100分級 Pydantic 強驗證合約
防爆核心: 跨層通訊的絕對真理，約束矩陣維度對齊 (Length Match Guard)，嚴禁 NaN 破壞模型。

📜 絕對架構與依賴契約

資料驗證: 必須使用 pydantic 的 BaseModel 與 Field。

自訂檢驗: 必須包含 @field_validator 用於確保基準資產存在、確保各 DataFrame 長度嚴格相等。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

import polars as pl
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Literal, Any

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

class DataConfig(BaseModel):
    """
    [Layer 1] 超參數配置契約
    定義全局心跳基準與全市場註冊表。
    """
    base_asset: str = Field("BTC/USDT", description="驅動全局心跳的基準資產")
    target_assets: Dict[str, AssetMetadata] = Field(..., description="全市場標的註冊表 (Key: asset_id)")
    timeframes: List[str] = Field(
        default=["1m", "5m", "15m", "1h", "1d"], 
        description="需進行 Backward Join 縫合的多重時間框架"
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
        base_asset = info.data.get("base_asset") if hasattr(info, "data") else "BTC/USDT"
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
    features_value: pl.DataFrame = Field(..., description="實際數值或 ffill 後的連續數值矩陣")
    features_mask: pl.DataFrame = Field(..., description="Boolean: True 表示該 T 刻有真實交易，False 為補值")
    features_reason: pl.DataFrame = Field(..., description="IntEnum: 0=正常, 1=休市, 2=熔斷, 3=API斷線")
    
    # --- 時間語意特徵 (Event Clock & Time Deformation) ---
    time_features: pl.DataFrame = Field(..., description="包含 time_delta_tau, minutes_since_open 等")

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


Phoenix Core: Layer 1 - Aligner (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/processors/aligner.py
系統定位: ⚡ 防爆核心 (Anti-Lookahead Core)
防爆核心: 結束時間錨定 (End-Time Indexing)、嚴格的 Backward Join 隔離未來視、Staleness 延遲追蹤。

📜 絕對架構與依賴契約

資料引擎: 必須且僅能使用 polars。

無狀態約束: 嚴禁維護跨時間步的狀態緩存，必須為純函數 (Pure Function) 映射。

對齊約束: 必須使用 join_asof 與 strategy="backward"。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

import polars as pl
from typing import Dict
import logging

logger = logging.getLogger("TimeframeAligner")

class TimeframeAligner:
    """
    [Layer 1 - ⚡ SOTA 核心] 多時間框架對齊縫合器
    職責：將 1m 基準 K 線與更高階的尺度 (5m, 15m, 1h...) 進行嚴格的 Backward 對齊。
    
    SOTA 升級亮點 (End-Time Indexing)：
    1. 結束時間錨定: 依賴上游 DataLake 預先算好的 close_time，徹底消滅所有時間戳重疊的前視偏差。
    2. 極致輕量化: 廢除動態 offset_by 計算，大幅降低實盤 CPU 開銷。
    3. 資訊衰減追蹤: 延遲時間改為「距離上級 K 線收盤過了多久」，物理意義更精確。
    """

    @staticmethod
    def align_timeframes(base_df: pl.DataFrame, macro_dfs: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """
        將基準 (1m) 與其他尺度進行 Join 縫合，並計算延遲。
        """
        print("[*] 啟動 Timeframe Aligner: 執行多尺度對齊縫合與衰減計算 (End-Time Backward Join)...")
        
        df_aligned = base_df
        
        for tf, macro_df in macro_dfs.items():
            if tf == "1m":
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
            # 邏輯：用 1m 的 datetime (10:04) 去往回找巨觀的 close_time (09:59:59.999)
            df_aligned = df_aligned.join_asof(
                macro_select,
                left_on="datetime",
                right_on=macro_time_col,
                strategy="backward"
            )
            
            # 5. 計算資料衰減 (Staleness Tracking)
            # 公式: (當前 1m 開盤時間 - 宏觀 K 線物理收盤時間) / 60000 毫秒 = 延遲的分鐘數
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


Phoenix Core: Layer 1 - Market Calendar (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/processors/market_calendar.py
系統定位: ⚡ 防爆核心 (Market Calendar Engine)
防爆核心: 判斷市場物理開關狀態，區分「休市」與「斷線」，防止偽造的 0 波動污染特徵。

📜 絕對架構與依賴契約

介面契約: 必須提供 is_market_open 介面供其他模組呼叫。

(註：當前為簡化版，預留未來擴充 exchange_calendars 的空間)

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

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


Phoenix Core: Layer 1 - Missing Encoder (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/processors/missing_encoder.py
系統定位: ⚡ 防爆核心 (3D Missingness Encoding & Absolute NaN Guard)
防爆核心: 將空值轉化為物理狀態碼，建立三維特徵掩碼，並執行極限 NaN 阻擋。

📜 絕對架構與依賴契約

狀態碼約束: 必須使用 IntEnum 鎖死 MissingReason (0=NORMAL, 1=CLOSED, 2=HALTED, 3=API_DOWN)。

極限防護: 必須執行雙層填充 fill_null(strategy="forward").fill_null(0.0) 保證零 NaN。

單次圖優化: 必須將所有特徵生成打包進單一 with_columns 中執行。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

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



Phoenix Core: Layer 1 - Time Clock (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/extractors/time_clock.py
系統定位: ⚡ SOTA 核心 (Event Clock & Time Deformation)
防爆核心: 萃取物理時間的非線性流逝率 (d_tau)，並在休市時強制將時間流逝歸零。

📜 絕對架構與依賴契約

依賴綁定: 僅限 polars 與 numpy。

狀態凍結約束: 計算 time_delta_tau 時，必須基於 features_mask 進行防禦，休市/斷線期間一律補 0.0。

邊界標準化: 所有事件時鐘特徵必須被 .clip(-1.0, 1.0) 鎖死，防止梯度爆炸。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

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

    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        主介面：一次性套用所有的時間語意特徵
        """
        print("[*] 啟動 Time Clock Extractor: 萃取時間變形與事件時鐘...")
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


Phoenix Core: Layer 1 - Feature Proxy (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/extractors/feature_proxy.py
系統定位: ⚡ SOTA 中樞 (Dynamic Feature Routing Proxy)
防爆核心: 根據 YAML 兵符動態載入外部特徵腳本，確保 Namespace 隔離與單次平行圖優化 (Lazy Eval)。

📜 絕對架構與依賴契約

動態載入防護: 必須實作 _preload_modules()，在初始化階段 Fail-fast 驗證外部 H/G 介面。

Namespace 防碰撞: 所有生成的運算式 (Expr) 必須加上 YAML 提供的 UID 前綴 (f"{uid}_{feat_name}")。

單次圖優化: 必須將所有解析出的 Expressions 打包進一個 List 中，最後一次性交由 with_columns() 處理。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

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

    def extract_features(self, df: pl.DataFrame) -> pl.DataFrame:
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


Phoenix Core: Layer 1 - Binance Client (100% 復現鋼印)

目標檔案: phoenix/core/layer1_data/fetchers/binance_client.py
系統定位: ⚡ SOTA 閘道器 (REST Backoff & WebSocket Streaming)
防爆核心: 確保資料收集穩定，具備自動重連機制與 CCXT 封裝，絕對阻擋網路異常與 API 速率限制。

📜 絕對架構與依賴契約

非同步要求: 必須嚴格使用 ccxt.async_support 進行 REST 請求與 ccxt.pro 進行 WebSocket 訂閱。

退避算法: REST 歷史回補必須實作指數退避 (Exponential Backoff)。

實盤收盤約束: 串流引擎僅回傳最後一根已完成 / 正在跳動的即時狀態。

💻 100% 原始碼藍圖 (Source Code Blueprint)

⚠️ 系統警告：重構或提取此檔案時，必須一字不差地使用以下程式碼，嚴禁擅自增刪邏輯。

import polars as pl
import asyncio
import logging
from typing import Optional, AsyncGenerator
import ccxt.async_support as ccxt_async
import ccxt.pro as ccxt_pro

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BinanceClient")

class BinanceClient:
    """
    [Layer 1 - ⚡ SOTA 核心] 幣安非同步與實盤串流客戶端
    職責：
    1. 提供帶有「指數退避 (Exponential Backoff)」防禦機制的歷史資料拉取 (REST)
    2. 提供高速零延遲的 WebSocket 即時 K 線串流 (Streaming)
    """
    
    def __init__(self, api_key: Optional[str] = None, secret: Optional[str] = None):
        config = {
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True, # CCXT 內建的基礎速率限制
            'options': {
                'defaultType': 'future', # 預設切換為合約市場 (依需求可改為 'spot')
            }
        }
        
        # REST API 非同步客戶端 (用於歷史回補)
        self.rest_client = ccxt_async.binance(config)
        
        # WebSocket 即時串流客戶端 (用於實盤推播)
        self.ws_client = ccxt_pro.binance(config)

    async def fetch_ohlcv_with_backoff(self, symbol: str, timeframe: str, limit: int = 1000, max_retries: int = 5) -> pl.DataFrame:
        """
        [防爆核心] 獲取歷史 K 線，並具備指數退避重試機制
        """
        retries = 0
        while retries < max_retries:
            try:
                # 非同步拉取資料
                ohlcv = await self.rest_client.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not ohlcv:
                    return pl.DataFrame()

                # 高速轉換為 Polars DataFrame
                df = pl.DataFrame(
                    ohlcv, 
                    schema=["timestamp", "open", "high", "low", "close", "volume"],
                    orient="row"
                ).with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"))
                
                return df
                
            except ccxt_async.RateLimitExceeded as e:
                wait_time = 2 ** retries # 1, 2, 4, 8, 16 秒
                logger.warning(f"[Rate Limit] 觸發 API 限制。{wait_time} 秒後重試... ({retries+1}/{max_retries})")
                await asyncio.sleep(wait_time)
                retries += 1
                
            except ccxt_async.NetworkError as e:
                wait_time = 2 ** retries
                logger.warning(f"[Network Error] 網路連線異常: {e}。{wait_time} 秒後重試... ({retries+1}/{max_retries})")
                await asyncio.sleep(wait_time)
                retries += 1
                
            except Exception as e:
                logger.error(f"[Fatal Error] 未知異常: {e}")
                raise e
                
        raise Exception(f"無法獲取 {symbol} 數據：已達最大重試次數 {max_retries}。")

    async def stream_live_kline(self, symbol: str, timeframe: str) -> AsyncGenerator[pl.DataFrame, None]:
        """
        [實盤引擎] 透過 WebSocket 訂閱實時 K 線變化
        每次 K 線更新 (Tick) 或收盤時，Yield 出最新的一根 K 線特徵
        """
        logger.info(f"[*] 啟動 WebSocket 串流：{symbol} ({timeframe})")
        while True:
            try:
                # CCXT Pro 的 watch_ohlcv 會自動處理訂閱與斷線重連
                ohlcv_list = await self.ws_client.watch_ohlcv(symbol, timeframe)
                
                # 取得最新的一根 K 線 (包含尚未收盤的即時跳動)
                latest_candle = ohlcv_list[-1]
                
                df = pl.DataFrame(
                    [latest_candle],
                    schema=["timestamp", "open", "high", "low", "close", "volume"],
                    orient="row"
                ).with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"))
                
                yield df
                
            except Exception as e:
                logger.error(f"[WebSocket Error] 串流中斷: {e}。5 秒後嘗試重新建立連線...")
                await asyncio.sleep(5)

    async def close_connections(self):
        """安全關閉所有連線"""
        logger.info("[*] 關閉 Binance 連線...")
        await self.rest_client.close()
        await self.ws_client.close()

# === 模組測試區 (使用 asyncio 運行) ===
async def _test_runner():
    # 建立客戶端 (公開資料無需 API Key)
    client = BinanceClient()
    symbol = "BTC/USDT"
    
    print("\n--- 測試 1: REST API (帶有防禦機制) ---")
    df = await client.fetch_ohlcv_with_backoff(symbol, "1m", limit=5)
    print(df)
    
    print("\n--- 測試 2: WebSocket 實盤串流 (監聽 3 次 Tick 後退出) ---")
    count = 0
    async for live_df in client.stream_live_kline(symbol, "1m"):
        print(f"[Tick {count+1}] 即時價格: {live_df['close'][0]}, 累積成交量: {live_df['volume'][0]}")
        count += 1
        if count >= 3:
            break
            
    await client.close_connections()

if __name__ == "__main__":
    # 在 Windows 環境下避免 RuntimeError
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(_test_runner())
