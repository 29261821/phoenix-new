Phoenix Core: Layer 1 Data Layer - META 工程文件

系統版本: v1.5-Lite (100-Point Standard)
底層架構: WSL + Rust (Compute) + Python (Logic)
核心依賴: polars, numpy, ccxt, pydantic, pyo3 (Rust FFI)
上游輸入: Binance API (實盤) 或 CSV/Parquet (回測)
下游輸出: MarketObservation (供 Layer 2 使用的對齊特徵)

系統版本: v2.0-Pro (100-Point Standard, Category B Integrated)
底層架構: WSL + Rust (Compute) + Python (Logic)
核心依賴: polars, numpy, exchange_calendars (新增), pydantic, pyo3
上游輸入: 多資料源 (跨市場 API / 本地 Data Lake)
下游輸出: MarketObservation (包含三維遺失編碼與時間變形特徵)

📂 實體目錄樹狀圖 (Directory Tree)

phoenix/
├── infra/                            # 🚀 基礎設施層 (WSL + Rust)
│   ├── rust_core/
│   |   ├── src/
│   |   │   ├── lib.rs                # PyO3 綁定
│   |   │   └── math_ops/
│   |   │       ├── orderflow.rs      # [Rust] 高頻代理特徵運算
│   |   │       └── rolling_vol.rs    # [Rust] 滾動波動率運算
│   |   └── Cargo.toml
│   └── data_lake/                    # [新增] 本地高效能特徵庫
│       └── parquet_store/            # [新增] 儲存多尺度、已標記的三維 .parquet 檔案
└── core/                             # 🧠 金融核心層 (Python)
    └── layer1_data/                  # 本模組路徑
        ├── __init__.py               
        ├── schemas.py                # [介面契約] 嚴格的資料與配置結構 三維狀態與元數據
        ├── pipeline.py               # 數據管線協調器 (區分 streaming 與 batch)
        ├── data_lake_builder.py      # [新增] 歷史數據抓取與三維標籤建構器 (Data Lake Builder)
        ├── fetchers/                 
        │   ├── __init__.py
        │   ├── binance_client.py     
        │   └── local_client.py   
        ├── processors/               
        │   ├── __init__.py
        │   ├── aligner.py            
        │   ├── market_calendar.py    # [新增] 市場日曆引擎 (處理節假日/開收盤切分)
        │   └── missing_encoder.py    # [新增] 三維遺失編碼器 (Value, Mask, Reason)
        └── extractors/               
            ├── __init__.py
            ├── momentum_proxy.py     
            ├── micro_proxy.py        # [Rust FFI]
            └── time_clock.py         # [新增] 事件時鐘與時間變形特徵萃取


📜 介面契約與超參數配置 (Interface Contracts & Config) - ⚠️ 100分級約束

全面捨棄單一的 is_trading，改用多維度張量結構，並強制注入資產物理屬性。

schemas.py 嚴格定義：

from pydantic import BaseModel, Field, validator
import polars as pl
from typing import List, Literal, Dict

class AssetMetadata(BaseModel):
    """資產物理與制度描述 (Asset Semantics)"""
    asset_id: str = Field(..., description="標的代碼 (e.g., 'BTC/USDT', 'TX00')")
    asset_type: Literal["crypto", "equity_index", "commodity", "forex"]
    timezone: str = Field(..., description="交易所時區 (e.g., 'UTC', 'Asia/Taipei')")
    tick_size: float = Field(..., description="最小跳動單位")

class DataConfig(BaseModel):
    """Layer 1 超參數配置契約"""
    base_asset: str = "BTC/USDT"
    target_assets: Dict[str, AssetMetadata] = Field(..., description="全市場標的註冊表")
    timeframes: List[str] = ["1m", "5m", "15m", "1h", "1d"]

class MarketObservation(BaseModel):
    """Layer 1 輸出契約"""
    timestamp: int # 全局基準 Heartbeat (1m)
    asset_id: str
    
    # 核心特徵必須被拆分為三個維度 (Missingness Encoding)
    features_value: pl.DataFrame  # 實際數值或 ffill 後的數值
    features_mask: pl.DataFrame   # Boolean: True 表示該 T 刻有真實交易，False 表示是填補的
    features_reason: pl.DataFrame # IntEnum: 0=Normal, 1=Closed, 2=Halted(熔斷), 3=APIDown
    
    # 時間語意特徵 (Event Clock & Time Deformation)
    time_features: pl.DataFrame   # 包含 minutes_since_open, volatility_time_delta 等

    class Config:
        arbitrary_types_allowed = True



📄 檔案權責與 META 規範 (File Responsibilities & META Specs)

1. pipeline.py (管線協調器與狀態機)

META 約束: 必須實作 class DataLayer。

全局心跳驅動 (Global Heartbeat): 系統時間軸由 base_asset 的 1m K線驅動。每次 base_asset 產生一根 1m K線，即觸發一次 step()。

實盤推論 step(new_kline) -> MarketObservation: 接收新的 1m 心跳，拉取所有關聯商品與多重尺度 (5m~1d) 的最新狀態，進行縫合與特徵萃取。

歷史回測 batch(start_time, end_time) -> MarketObservation: 批次拉取歷史資料，利用 Polars 向量化極速算出所有特徵。

2. processors/async_market.py (異步市場處理器) - ⚠️ 跨市場狀態空間紅線

權責: 解決「某些資產有資料（開盤），某些資產無資料（休市/暫停交易）」的問題。狀態空間模型如果看到價格變成 0 會崩潰，如果只是單純保留舊價格，模型會誤以為該資產出現「完美的零波動」，從而產生錯誤的避險權重。

META 約束:

價格凍結與狀態標籤 (Price Freeze & Status Flag): 當基準心跳 (如 BTC 1m) 觸發時，若某標的 (如美股 SPY) 處於休市：

該標的的價格與指標必須強制 ffill (向後填充上一個有效收盤價)。

強制新增布林特徵 is_trading_SPY = False。這讓後續的神經網路 (Layer 2) 與 HMM (Layer 3) 在空間映射時，能明確學習到「這段 0 波動是因為休市，而非真實的市場死水」，從而遮蔽 (Mask) 該特徵的權重。

3. processors/aligner.py (多時間框架縫合) - ⚠️ 系統絕對防爆點

META 約束:

嚴禁任何 Pandas 操作，全面使用 Polars 提升大數據吞吐量。

必須實作 align_timeframes(base_df: pl.DataFrame, macro_dfs: List[pl.DataFrame]) -> pl.DataFrame。

向後對齊指令鎖死: 必須使用 base_df.join_asof(macro_df, on="timestamp", strategy="backward")。這代表如果現在基準是 10:04 的 1m K線，它對齊 5m 框架時，會嚴格去尋找 10:00 產生的那根 5m K 線，對齊 1h 框架時去尋找 10:00 的 1h K 線，絕對不允許偷看未來資訊。

4. extractors/micro_proxy.py (Rust FFI 接入點)

META 約束:

Python 端不寫迴圈。將 Polars Series 轉為 Numpy Array 後，直接傳遞給 import rust_core。

例如計算 Order Flow Imbalance (OFI)：ofi_array = rust_core.calculate_ofi(high.to_numpy(), low.to_numpy(), close.to_numpy(), volume.to_numpy())，然後立刻轉回 Polars Series pl.Series("ofi_proxy", ofi_array)。

5. processors/market_calendar.py (市場日曆引擎) - ⚠️ 對齊防爆點

權責: 解決「8:45 開盤的 1h 線該如何定義」的時區與聚合問題。

META 約束:

必須依賴 exchange_calendars 或自建的 Session 規則庫。

時間聚合紅線: 聚合 K 線 (Resampling) 時，必須嚴格依據該資產的 AssetMetadata.timezone 與 Session_Start。如果台指期 8:45 開盤，1h 的區間必須嚴格切分為 [08:45-09:45)，而不是盲目地對齊整點 [09:00-10:00)。

必須提供 is_market_open(asset_id, timestamp) 介面，供下游準確判斷。

6. processors/missing_encoder.py (三維遺失編碼器)

權責: 不再讓 ffill 吃掉「沒有資料」這層寶貴的資訊。

META 約束:

發生 NaN 時，features_value 執行 ffill 確保算力底層不會崩潰。

但必須同步寫入 features_mask = False。

必須呼叫 market_calendar.py，若判定當下為假日，則 features_reason = 1；若開盤期間無 Tick，則判斷為流動性枯竭或 API 斷線 features_reason = 3。

7. extractors/time_clock.py (事件時鐘與時間變形) - ⚡ SOTA 核心

權責: 將物理時間轉換為「市場感知時間」。

META 約束:

Event Clock: 必須強制產出 minutes_since_open 與 minutes_until_close 特徵，並標準化至 [-1, 1] 區間。

Time Deformation (時間變形公式): 實作基於波動率與成交量的時間步長。例如，定義市場時間增量 $d\tau_t = \frac{\sigma_t \times V_t}{\overline{\sigma \times V}} dt$。當此值飆高時，代表系統進入「超密集時間（如 CPI 發布瞬間）」，該特徵將直接成為 Layer 2 注意力機制的權重調節器。

🔌 下游依賴 (Layer 2 預告)

產出的 MarketObservation 保證了每一筆資料的 timestamp 對應的特徵，都只有發生在該時間點「之前」的資訊。Layer 2 將安全地使用這些特徵進行神經網路編碼。