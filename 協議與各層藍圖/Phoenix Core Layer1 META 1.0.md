Phoenix Core: Layer 1 Data Layer - META 工程文件

系統版本: v2.0-Pro (100-Point SOTA Standard, Multi-Asset Async Enabled)
底層架構: WSL + Polars (Data Engine) + Asyncio (I/O) + Python (Logic)
核心依賴: polars, numpy, ccxt, exchange_calendars, pydantic
上游輸入: 幣安 REST/WebSocket API、本地 Parquet Data Lake
下游輸出: MarketObservation (具備絕對物理一致性的張量化前置特徵)

📂 實體目錄樹狀圖 (Directory Tree)

phoenix/
├── infra/                            
│   └── data_lake/                    
│       └── parquet_store/            # 儲存多尺度、已標記的三維 .parquet 檔案
└── core/
    └── layer1_data/                  
        ├── __init__.py               
        ├── schemas.py                # [介面契約] 100分級 Pydantic 強驗證合約
        ├── pipeline.py               # [協調器] 雙引擎 (Streaming/Batch) 數據管線
        ├── data_lake_builder.py      # [基礎設施] Asyncio 多資產併發抓取器
        ├── fetchers/                 
        │   ├── binance_client.py     # 指數退避 REST & WebSocket 實盤串流
        │   └── local_client.py       # Parquet 高速讀取介面
        ├── processors/               
        │   ├── aligner.py            # [防爆] Backward Join 與 Staleness (延遲) 追蹤
        │   ├── market_calendar.py    # [防爆] 交易所日曆引擎
        │   └── missing_encoder.py    # [防爆] 三維遺失編碼 (包含極限 NaN 防護)
        └── extractors/               
            ├── momentum_proxy.py     # [極速] Polars Lazy Eval 向量化技術指標
            ├── micro_proxy.py        # [預留] Rust FFI 微觀結構分析
            └── time_clock.py         # [SOTA] 事件時鐘與時間變形 ($d\tau$) 萃取


📜 絕對介面契約 (Absolute Interface Contracts)

Layer 2 模型工廠唯一認可的輸入格式為 MarketObservation。任何不符合此 Pydantic 契約的資料將在進入 GPU 前被直接拋錯攔截。

1. schemas.MarketObservation 定義與形狀約束

class MarketObservation(BaseModel):
    timestamp: int              # 全局基準 Heartbeat (Unix Timestamp 毫秒)
    asset_id: str               # 標的代碼 (e.g., 'BTC/USDT')
    
    # [核心約束] 以下四個 DataFrame 的 Rows 長度必須 100% 相同
    features_value: pl.DataFrame  # 數值特徵矩陣 (開高低收、成交量、RSI、MACD 等)
    features_mask: pl.DataFrame   # Boolean 矩陣 (True: 有效交易, False: 填補值)
    features_reason: pl.DataFrame # IntEnum 矩陣 (0: 正常, 1: 休市, 2: 熔斷, 3: 斷線)
    time_features: pl.DataFrame   # 時間語意矩陣 (d_tau, 事件時鐘)


2. 極端數值防護 (NaN / Inf Guards)

特徵矩陣 (features_value): 絕對不允許存在 NaN。當遇到斷線或開盤無資料時，必須嚴格執行 fill_null(strategy="forward").fill_null(0.0)。

布林遮罩 (features_mask): 對於上述被強制補 0.0 的列，其對應的 Mask 必須為 False，藉此通知 Layer 2 注意力機制 (Attention) 忽略該列。

📄 模組權責與 SOTA 規範 (Module Specifications)

1. pipeline.py (管線協調器) - ⚠️ mCP 級別絕對復現規範

此模組為 L1 中樞，負責呼叫所有處理器。未來重構或跨層呼叫時，必須 100% 遵守以下簽名與內部邏輯，嚴禁任何幻覺發散：

類別命名與依賴綁定: * 類別名稱必須為 DataLayer。

必須 import polars as pl 與 from typing import Dict。

必須匯入自身目錄下的合約與處理器：MarketObservation, DataConfig, MissingnessEncoder, TimeframeAligner, MarketCalendar, TimeClockExtractor, MomentumProxy。

建構子約定 __init__(self, config: DataConfig):

必須綁定超參數：self.config = config。

必須將五大處理器實例化並綁定至實例變數：self.encoder, self.aligner, self.calendar, self.clock, self.momentum。嚴禁在此處傳遞任何額外參數。

實盤引擎約定 step(self, raw_1m_window: pl.DataFrame, macro_data: Dict[str, pl.DataFrame]) -> MarketObservation:

執行順序絕對紅線 (不可顛倒):

遺失編碼: obs_df = self.encoder.encode(raw_1m_window)

動能特徵注入: obs_df = self.momentum.extract_features(obs_df)

時間語意萃取: obs_df = self.clock.extract(obs_df)

O(1) 空間約束: 完成時間序列計算後，必須強制截取最後一筆狀態：latest_obs = obs_df.tail(1)。

多尺度縫合: 僅對截取後的單筆資料執行向後縫合：final_df = self.aligner.align_timeframes(latest_obs, macro_data)。

合約封裝紅線 (必須精確 Mapping): 回傳 MarketObservation 時，欄位選擇必須嚴格對齊：

timestamp: 取自 final_df["datetime"][0].timestamp()

asset_id: 取自 self.config.base_asset

features_value: final_df.select(pl.exclude(["features_mask", "features_reason"]))

features_mask: final_df.select(["features_mask"])

features_reason: final_df.select(["features_reason"])

time_features: 必須且僅包含 final_df.select(["time_delta_tau", "minutes_since_open_norm", "minutes_until_close_norm"])。

回測引擎約定 batch(self, base_1m_history: pl.DataFrame, macro_history: Dict[str, pl.DataFrame]) -> pl.DataFrame:

必須遵循與 step() 相同的執行順序 (Encoder -> Momentum -> Clock)。

禁止截斷: 嚴禁呼叫 .tail(1)，必須對整張表執行 final_df = self.aligner.align_timeframes(df, macro_history)。

必須直接回傳 final_df (Polars DataFrame 型別)，供 Layer 2 直接轉化為巨大張量。

2. data_lake_builder.py / fetchers/binance_client.py (網路與 I/O) - ⚠️ mCP 級別絕對復現規範

此子系統為數據源頭，分為「實盤推播閘道器」與「離線多資產併發引擎」。未來重構必須 100% 遵守以下網路防護與併發約定：

fetchers/binance_client.py (實盤與歷史回補閘道):

類別命名: BinanceClient

依賴綁定: 必須使用非同步版本的 CCXT：import ccxt.async_support as ccxt_async (REST API) 與 import ccxt.pro as ccxt_pro (WebSocket API)。嚴禁在此模組使用同步 ccxt。

歷史回補約定 fetch_ohlcv_with_backoff(self, symbol: str, timeframe: str, limit: int = 1000, max_retries: int = 5) -> pl.DataFrame:

指數退避紅線: 內部必須實作 while retries < max_retries: 的防禦迴圈。

例外攔截: 必須精確 except 捕捉 ccxt_async.RateLimitExceeded 與 ccxt_async.NetworkError。

退避邏輯: 觸發例外時，必須執行 await asyncio.sleep(2  retries) 進行指數退避延遲。

輸出約束: 返回的 pl.DataFrame 必須包含 Schema ["datetime", "open", "high", "low", "close", "volume"]，且 datetime 欄位必須是精確至 ms (毫秒) 級別的 Timestamp。

實盤推播約定 stream_live_kline(self, symbol: str, timeframe: str) -> AsyncGenerator[pl.DataFrame, None]:

必須使用 yield 關鍵字將每筆更新產出為 AsyncGenerator，供 pipeline.py 以 async for 語法流式消耗。

斷線重連紅線: 必須包覆在外層 while True: 與 try...except 區塊內。當 watch_ohlcv 拋出連線異常時，必須觸發 await asyncio.sleep(5) 後自動重建連線。

data_lake_builder.py (離線多資產資料湖建構器):

類別命名: DataLakeBuilder

多核心併發約定 build_multi_asset_lake(self, assets: List[str], start_date: str, end_date: str):

絕對併發紅線: 嚴禁寫出 for asset in assets: await self.process_single_asset(...) 這種會阻塞網路 I/O 的序列化迴圈。

必須建立任務清單 tasks = [self.process_single_asset(...) for ...]，並使用 await asyncio.gather(*tasks) 將所有資產的抓取任務平行發布至事件迴圈。

單一資產管線約定 process_single_asset(self, asset: str, start_date: str, end_date: str):

執行順序 (I/O 與 CPU 分離):

非同步網路 I/O: 呼叫自身綁定的 ccxt_async 抓取資料。

同步 CPU 運算: apply_missingness_encoding(df)。

同步 CPU 運算: generate_multi_timeframes(df_1m)。

同步磁碟 I/O: save_to_datalake(...)。

磁碟路徑鎖死: 在儲存為 Parquet 時，asset_id (如 "BTC/USDT") 必須轉換為安全字串 symbol_safe (如 "BTC_USDT")。最終存放路徑必須嚴格對齊架構圖：{self.data_dir}/{symbol_safe}/{symbol_safe}_{tf}.parquet。

3. phoenix/core/layer1_data/processors/aligner.py
系統定位: ⚡ 防爆核心 (Anti-Lookahead Core)
核心依賴: polars

此文件為 TimeframeAligner 模組的唯一真理 (Single Source of Truth)。未來任何系統重構、模型微調或跨模組呼叫，必須 100% 遵守本文件的簽名與內部演算邏輯，嚴禁任何大語言模型 (LLM) 產生邏輯幻覺或擅自更改矩陣操作方式。

📜 類別與依賴絕對約定 (Class & Dependencies)

類別命名: 必須精確命名為 TimeframeAligner。

依賴綁定: 模組頂部必須且僅需引入 import polars as pl 與 from typing import Dict。若需記錄日誌，可引入 import logging。

無狀態約束 (Stateless): 此類別不維護任何跨時間步的狀態緩存 (State Cache)，所有操作必須基於傳入的 DataFrame 進行純函數 (Pure Function) 映射。因此不需要實作 __init__。

⚙️ 核心方法約定 (Core Method Specifications)

align_timeframes(base_df: pl.DataFrame, macro_dfs: Dict[str, pl.DataFrame]) -> pl.DataFrame

此方法為多尺度對齊的唯一入口，其內部執行順序與運算子具有不可妥協的絕對紅線。

執行前置處理 (Preprocessing):

迴圈約定: 必須遍歷 macro_dfs 字典，並跳過基準時間框 (e.g., if tf == "1m": continue)。

時間軸強制排序 (Sorting Red Line): 進入任何 Join 操作前，必須對 macro_df 執行 .sort("datetime")。未排序的時間序列會導致後續的 Asof Join 產生致命的錯位或引擎崩潰。

巨觀時間戳保留: 必須將 macro_df 的 datetime 欄位重新命名為獨一無二的暫存欄位，例如 datetime_macro_{tf}，這是後續計算 Staleness (老化時間) 的唯一憑據。

命名空間隔離 (Namespace Isolation):

為了防止與 1m 基準特徵碰撞，必須動態過濾 macro_df 中的特徵。

僅保留 ["open", "high", "low", "close", "volume"] 與巨觀時間戳。

必須對所有的價量欄位強制加上後綴：pl.col(c).alias(f"{c}_{tf}")。

核心對齊防爆紅線 (Backward Join Constraint):

嚴禁使用一般的 join 或 merge。

必須精確呼叫 Polars 的高頻交易對齊函數：join_asof。

策略鎖死: 參數必須鎖死為 strategy="backward"。

物理意義解釋: 確保當前時間 $T$ (例如 10:04) 只能對齊到 $\le T$ 的最後一筆巨觀 K 線 (即 10:00 的 5m K 線)。若誤用 nearest 或 forward，系統將提前獲取 10:05 的收盤資訊，導致回測績效造假，實盤必定破產。

資訊衰減追蹤器 (Staleness Tracking) - SOTA 級別特徵:

對齊完成後，必須立刻計算資料的老化程度，供 Layer 2 的 Attention 機制或 SSM (Mamba) 評估權重。

運算公式: (當前 1m 基準時間 - 巨觀 K 線原始時間) 轉換為分鐘數。

Polars 實作約定:

staleness_col = f"staleness_{tf}_mins"
df_aligned = df_aligned.with_columns(
    ((pl.col("datetime") - pl.col(f"datetime_macro_{tf}")).dt.total_milliseconds() / 60000.0)
    .alias(staleness_col)
)


算完後，必須立刻 drop 掉暫存的 datetime_macro_{tf}，保持輸出矩陣的純淨度。

極限 NaN 防護 (Matrix Purity Guard):

在回測的最早期（例如開盤的前幾分鐘），高階巨觀 K 線（如 1h, 1d）根本還不存在，join_asof 會無可避免地產生 null (NaN)。

防禦紅線: 在回傳最終的 df_aligned 之前，必須執行全局的兩段式填充防禦：
df_aligned = df_aligned.fill_null(strategy="forward").fill_null(0.0)

第一段 forward: 正常對齊時的空隙填補。

第二段 0.0: 開局無資料時的絕對歸零防護。

🛡️ 對 Layer 2 (市場表徵層) 的絕對輸出保證

當任何模組呼叫 TimeframeAligner.align_timeframes() 後，保證獲得以下承諾：

無未來污染 (Zero Look-ahead Leakage): 所有 _{tf} 結尾的特徵，絕對不包含任何來自未來的物理資訊。

老化感知 (Staleness Awareness): 必定存在對應的 staleness_{tf}_mins 欄位。Layer 2 的神經網路可以安全地依賴這個數值，決定是否要捨棄太過久遠的巨觀特徵。

滿秩無損 (Full Rank / No NaN): 輸出的 DataFrame 中，每一個 Cell 都包含合法的浮點數，可直接無縫轉換為 PyTorch 的 FloatTensor 進入 GPU 運算。

4. phoenix/core/layer1_data/processors/missing_encoder.py
系統定位: ⚡ 防爆核心 (3D Missingness Encoding & Absolute NaN Guard)
核心依賴: polars, enum

此文件為 MissingnessEncoder 模組的唯一真理 (Single Source of Truth)。未來任何系統重構、模型微調或跨模組呼叫，必須 100% 遵守本文件的簽名與內部演算邏輯，嚴禁任何大語言模型 (LLM) 產生邏輯幻覺或使用不安全的 Pandas 填充策略。

📜 類別與依賴絕對約定 (Class & Dependencies)

依賴綁定: 模組頂部必須引入 import polars as pl、from typing import Optional 以及 from enum import IntEnum。

強型別狀態碼約束 (IntEnum Red Line):

嚴禁在程式碼中直接使用 Magic Numbers (0, 1, 2, 3) 來代表缺失原因。

必須定義 MissingReason(IntEnum) 類別，並嚴格鎖死以下物理意義：

NORMAL = 0 (正常交易)

CLOSED = 1 (市場休市，如週末的美股)

HALTED = 2 (熔斷或暫停交易)

API_DOWN = 3 (應該要有資料，但 API 斷線或流動性徹底枯竭)

⚙️ 核心方法約定 (Core Method Specifications)

encode(self, df: pl.DataFrame, is_market_open_map: Optional[pl.DataFrame] = None) -> pl.DataFrame

此方法負責將原始 1m K 線轉化為三維狀態矩陣。其內部運算必須遵循單次執行圖 (Single Execution Graph) 的效能最佳化原則。

特徵遮罩判定 (Mask Condition):

必須定義 Lazy Expression (惰性運算式) 來判斷該 K 線是否為有效交易。

判定條件: (pl.col("close").is_not_null()) & (pl.col("volume") > 0)。

這個運算式最終必須被命名為 features_mask (Boolean 型別)。

缺失原因判定 (Reason Encoding):

必須動態判斷 is_market_open_map 是否存在。

日曆存在時 (Left Join): 必須先將 df 與 is_market_open_map 進行 join(on="datetime", how="left")。接著透過 pl.when().then().otherwise() 進行階層式判定：

當 features_mask 為 True -> MissingReason.NORMAL.value

當 is_open 為 False -> MissingReason.CLOSED.value

否則 -> MissingReason.API_DOWN.value

無日曆時 (Fallback):

當 features_mask 為 True -> MissingReason.NORMAL.value

否則 -> MissingReason.API_DOWN.value

型別鎖死: 判定結果必須強制轉型為 pl.Int8，並命名為 features_reason，以節省記憶體。

絕對零 NaN 裝甲 (Absolute NaN Guard):

這是神經網路防爆的最後一道防線。原始 K 線可能在第一筆資料就出現 NaN (例如 API 一開機就沒抓到資料)，導致傳統的 Forward Fill 徹底失效。

數值填充紅線: 所有價量欄位 (open, high, low, close) 的填充運算式，必須嚴格採用雙層防禦：
pl.col("xxx").fill_null(strategy="forward").fill_null(0.0)

volume 欄位因為沒有向前參考的意義，缺失時直接補零：pl.col("volume").fill_null(0.0)。

效能榨取紅線 (Single Pass Execution):

嚴禁呼叫多次 df.with_columns()。

Mask 運算式、Reason 運算式與所有的數值填充運算式，必須全部打包在同一個 df.with_columns([...]) 陣列中，交由 Polars 底層 Rust 引擎進行多執行緒平行圖優化。

執行完畢後，若有暫存的日曆欄位 (如 is_open)，必須透過 df.drop() 將其移除，保持矩陣純淨。

🛡️ 對 Layer 2 (市場表徵層) 的絕對輸出保證

當原始資料經過 MissingnessEncoder.encode() 處理後，Layer 2 可以依賴以下絕對承諾：

零 NaN 保證: 整個 DataFrame 絕對不存在任何 null 或 NaN。可以直接轉換為神經網路所需的 Dense Tensor。

三維狀態對齊: 每一列資料必定同時具備「填充後的數值」、「是否為真實資料的 Mask」以及「為何缺失的 Reason」。Layer 2 的 Attention 機制可以透過 Mask 直接將假資料的權重歸零，不干擾模型學習。

休市語意保留: 休市期間的資料雖然被補為前一個收盤價，但 Reason 代碼會明確標示 1 (CLOSED)，讓下游的 Regime Layer (L3) 與 Transition Layer (L4) 能夠據此觸發「狀態凍結 (State Freeze)」機制。

5. phoenix/core/layer1_data/extractors/momentum_proxy.py
系統定位: ⚡ SOTA 核心 (Lazy Evaluation Vectorized Feature Engineering)
核心依賴: polars

此文件為 MomentumProxy 模組的唯一真理 (Single Source of Truth)。未來任何系統重構、模型微調或跨模組呼叫，必須 100% 遵守本文件的簽名與內部演算邏輯，嚴禁任何大語言模型 (LLM) 引入 pandas, numpy 迴圈或第三方技術指標庫 (如 TA-Lib)。

📜 類別與依賴絕對約定 (Class & Dependencies)

類別命名: 必須精確命名為 MomentumProxy。

依賴綁定: 模組頂部必須且僅需引入 import polars as pl 與 from typing import List。

零 Pandas 紅線 (Zero Pandas Policy): 嚴禁在此檔案中出現任何 import pandas as pd。所有運算必須且只能在 Polars 的 Rust 記憶體模型中進行。

⚙️ 核心特徵運算式約定 (Core Expression Specifications)

此模組的核心設計模式是「運算式生成 (Expression Generation)」。所有的技術指標計算函式，絕對不能直接接收或回傳 DataFrame，而是必須回傳 pl.Expr 或 List[pl.Expr]，讓運算邏輯保持惰性 (Lazy)。

RSI 運算式約定 (expr_rsi)

簽名: expr_rsi(price_col: str = "close", window: int = 14) -> pl.Expr

邏輯約束: 必須使用 pl.col(price_col).diff() 計算差值。對於上漲 (gain) 與下跌 (loss)，必須使用 .clip() 進行分離，並使用 .rolling_mean() 進行平滑。

防除零紅線: 計算 RS 時必須加上 1e-8  epsilon：rs = gain / (loss + 1e-8)。

MACD 運算式約定 (expr_macd)

簽名: expr_macd(price_col: str = "close", fast: int = 12, slow: int = 26, signal: int = 9) -> List[pl.Expr]

邏輯約束: 必須使用 Polars 原生的指數移動平均 .ewm_mean(span=..., adjust=False)。

輸出約束: 必須同時回傳三個運算式：macd_line, macd_signal, 與 macd_hist。

布林通道運算式約定 (expr_bollinger_bands)

簽名: expr_bollinger_bands(price_col: str = "close", window: int = 20, num_std: float = 2.0) -> List[pl.Expr]

輸出約束: 必須回傳四個運算式：

bb_upper (上軌) 與 bb_lower (下軌)

bb_width: 帶寬壓縮比 (upper - lower) / (basis + 1e-8)，用於捕捉突破。

bb_pct: 價格相對位置 %B，(price - lower) / (upper - lower + 1e-8)。

真實波動幅度約定 (expr_atr)

簽名: expr_atr(high_col: str = "high", low_col: str = "low", close_col: str = "close", window: int = 14) -> pl.Expr

邏輯約束: 必須利用 pl.col(close_col).shift(1) 取得前收盤價。計算 TR (True Range) 的三個組件後，必須使用 pl.max_horizontal([tr1, tr2, tr3]) 進行平行最大值擷取，最後套用 .rolling_mean() 平滑。

🚀 主執行方法約定 (Main Execution Method)

extract_features(self, df: pl.DataFrame) -> pl.DataFrame

此方法為動能特徵注入的唯一入口，其任務是將所有惰性運算式 (Lazy Expressions) 打包並交給 Rust 引擎一次性執行。

多核極限壓榨紅線 (Single-Pass Vectorization):

嚴禁使用 for 迴圈依序呼叫指標函式。

嚴禁對 DataFrame 進行多次的 .with_columns() 呼叫。

必須寫法: 將所有指標函式產生的 pl.Expr 收集成一個 List，然後僅呼叫一次 df.with_columns(feature_exprs)。

物理意義: 這允許 Polars 底層構建一張完整的有向無環圖 (DAG)，在 C++ / Rust 層級自動分配 CPU 執行緒同時計算 MACD、RSI 與 ATR，將百萬級 K 線的回測計算時間壓縮至毫秒級。

🛡️ 對 Layer 2 (市場表徵層) 的絕對輸出保證

當原始資料經過 MomentumProxy.extract_features() 處理後，保證獲得以下承諾：

時間複雜度恆定: 無論是 Batch 模式下塞入 10 年的歷史資料，還是 Streaming 模式下塞入 100 根 K 線的滾動視窗，特徵萃取的耗時將極大程度逼近 $O(1)$ 或被高度優化，絕不成為實盤交易的延遲瓶頸。

物理含義完整: 萃取出的特徵 (如 bb_width 與 bb_pct) 已被轉化為「無單位」或「標準化」的相對比例，直接對神經網路的權重更新具有友善度，減輕 Layer 2 進行 Normalization 的負擔。

6. phoenix/core/layer1_data/extractors/time_clock.py
系統定位: ⚡ SOTA 核心 (Event Clock & Time Deformation $\Delta\tau$)
核心依賴: polars

此文件為 TimeClockExtractor 模組的唯一真理 (Single Source of Truth)。未來任何系統重構、模型微調或跨模組呼叫，必須 100% 遵守本文件的簽名與內部演算邏輯，嚴禁任何大語言模型 (LLM) 產生數學幻覺或遺漏休市期間的時間凍結約束。

📜 類別與依賴絕對約定 (Class & Dependencies)

類別命名: 必須精確命名為 TimeClockExtractor。

依賴綁定: 模組頂部必須引入 import polars as pl。

建構子約束 (__init__):

必須定義並綁定兩個關鍵超參數：

vol_window: int = 60 (短期波動率的滾動視窗，預設 1 小時)

baseline_window: int = 1440 (歷史基準波動率的滾動視窗，預設 24 小時)

必須定義微小值常數 self.epsilon = 1e-8，用於防止除以零的數學崩潰。

⚙️ 核心數學特徵約定 (Core Mathematical Specifications)

此模組的任務是打破「物理時間絕對均勻」的假設，提取出非線性的「市場感知時間」。

1. apply_time_deformation(self, df: pl.DataFrame) -> pl.DataFrame

此函式計算時間變形係數 (Time Deformation, $d\tau_t$)，衡量當下市場「時間流逝的速度」。

運算順序絕對紅線:

對數報酬率 (Log Return): 必須計算 close 的對數差分。為防止 NaN，必須使用 .fill_null(0.0) 進行安全處理。

短期波動率 ($\sigma_t$): 計算對數報酬率在 vol_window 內的 .rolling_std()。

市場活躍度 ($Activity_t$): 定義為「短期波動率」乘上「成交量」：$\sigma_t \times Volume_t$。

基準活躍度 ($Baseline$): 計算市場活躍度在 baseline_window 內的 .rolling_mean()。

時間變形係數 ($d\tau_t$): 公式為 $Activity_t / (Baseline_t + \epsilon)$。

狀態凍結絕對防護 (Freeze Mask Red Line):

$d\tau_t$ 的計算結果必須通過 features_mask 的嚴格審查。

必須使用 pl.when(pl.col("features_mask") == False).then(0.0).otherwise(...)。

物理意義: 當市場休市或斷線時，沒有任何真實交易發生，因此市場的「內部時間」必須強制停止流逝 ($d\tau_t = 0.0$)。

特徵清理: 輸出前必須透過 .drop() 丟棄運算過程中產生的中介特徵 (如 log_ret, sigma_t)，只保留最終的 time_delta_tau。

2. apply_event_clock(self, df: pl.DataFrame, session_start_mins: int = 0, session_duration_mins: int = 1440) -> pl.DataFrame

此函式將線性的 Timestamp 轉換為神經網路易於理解的「週期性相角 (Phase Angle)」。

週期壓縮紅線 (Normalization):

神經網路對巨大的整數極其敏感，直接輸入絕對分鐘數會導致梯度爆炸。

必須將計算出的「距離開盤分鐘數 (mins_since_open)」轉換至 [-1.0, 1.0] 區間。

公式約定: (mins_since_open / (session_duration_mins / 2.0)) - 1.0。並強制使用 .clip(-1.0, 1.0) 鎖死邊界。

輸出雙特徵約束: 必須同時產出 minutes_since_open_norm 與 minutes_until_close_norm，為模型提供兩種不同相位的週期資訊。

🚀 主執行方法約定 (Main Execution Method)

extract(self, df: pl.DataFrame) -> pl.DataFrame

此方法為外部呼叫的唯一介面。

執行順序: 必須依序傳遞 DataFrame：先呼叫 apply_time_deformation，再呼叫 apply_event_clock。

🛡️ 對 Layer 2 (市場表徵層) 的絕對輸出保證

當原始資料經過 TimeClockExtractor.extract() 處理後，保證獲得以下承諾：

完美的 Mamba 步長映射 (Perfect SSM Step Size): 產出的 time_delta_tau 被設計為永遠是大於等於 0.0 的實數。Layer 2 的 Mamba 模型可以直接將其 Mapping 為內部狀態空間矩陣的 $\Delta$ 參數。

休市狀態凍結: 因為在休市期間 time_delta_tau 被鎖死為 0.0，當 Layer 2 的 Mamba 模型進行 $e^{A\Delta}$ 離散化轉換時，會完美退化為單位矩陣 $I$，從而保證 Hidden State 發生 $h_t = h_{t-1}$ 的絕對物理凍結，徹底根絕模型在週末產生幻覺。

🛡️ 對 Layer 2 (市場表徵層) 的絕對輸出保證

當這份合約生效後，接下來開發的 Layer 2 可以 100% 放心依賴以下假設：

形狀完美: 傳入的 value, mask, time 張量長度絕對對齊，不需要在 PyTorch 裡再做防呆 assert。

沒有 NaN: 張量可以直接丟進神經網路層計算，無需擔心出現 Loss = NaN 的災難。

無未來污染: 在 Batch 回測模式下，所有的資料列都已經在時間軸上嚴格 Backward 隔離，Layer 2 算出的回測績效即為實盤的真實期望值。


# Phoenix Core: Layer 1 FeatureProxy - META 絕對約束文件

**目標模組:** `phoenix/core/layer1_data/extractors/feature_proxy.py` (取代舊有的 momentum_proxy.py)
**系統定位:** ⚡ SOTA 中樞 (YAML-Driven Dynamic Feature Routing)
**核心依賴:** `polars`, `importlib`

此文件為 `FeatureProxy` 模組的唯一真理。此模組負責從 YAML 設定檔動態載入 `features_library` 中的指標模組，絕對禁止在模組內出現任何寫死的指標名稱 (如 RSI, MACD)。

## 📜 類別與依賴絕對約定 (Class & Dependencies)

*   **類別命名:** 必須精確命名為 `FeatureProxy`。
*   **依賴綁定:** 允許引入 `polars as pl`, `importlib`, `typing (Dict, Any, List)`。嚴禁依賴 `pandas` 或任何具體的特徵檔案。
*   **介面對齊:** 必須接收來自 `configs` 的 YAML 字典解析結果 (包含 `meta_features_plan` 與 `indicator_default_params`)。

## ⚙️ 核心方法與防爆紅線 (Core Method Specifications)

### `__init__(self, config_dict: Dict[str, Any])`

*   **Fail-Fast 模組預載入紅線:** 
    *   嚴禁在實盤 `step()` 中動態 `import`，這會造成嚴重的 I/O 延遲。
    *   必須在 `__init__` 階段遍歷 `meta_features_plan`，透過 `importlib.import_module()` 將所有需要的指標腳本 (如 `src.features.indicators.SMC_Context`) 載入記憶體字典中。
    *   若 YAML 寫錯或檔案不存在，必須在此刻直接引爆 `ImportError`，嚴禁帶病進入實盤。

### `extract_features(self, df: pl.DataFrame) -> pl.DataFrame`

此方法將遍歷出戰名單，呼叫 H 接口與 G 接口，最後一次性注入特徵。

*   **參數合併約定 (Parameter Resolution):**
    *   必須先抓取 `indicator_default_params` 作為基底。
    *   利用 `meta_features_plan` 中的 `params` (若有) 覆蓋基底參數。
*   **命名空間防碰撞紅線 (Namespace Collision Guard):**
    *   *致命情境:* 若 YAML 定義了兩個 MACD (一個快, 一個慢)，G 接口吐出的欄位名稱都會是 `feat_macd_position_stable`，導致欄位覆寫崩潰。
    *   *絕對防禦:* 在收集 G 接口吐出的字典 (Dict[str, pl.Expr]) 時，必須利用 `pl.Expr.alias()` 強制重新命名，格式鎖死為：`{YAML_ID}_{Original_Feature_Name}`。
*   **單次向量化紅線 (Single-Pass Execution):**
    *   必須建立一個 `exprs: List[pl.Expr]`。
    *   遍歷完所有 YAML 出戰名單後，僅能呼叫一次 `df.with_columns(exprs)`。

## 🛡️ 對 Layer 1.5 & Layer 2 的絕對輸出保證
*   **完全解耦:** 無論增減多少特徵、替換多少 YAML 方案，Layer 1 主程式碼 0 修改。
*   **特徵溯源:** 每一個輸出的 Column Name 都帶有 YAML 賦予的 Unique ID，後續 Layer 1.5 (LightGBM) 進行特徵重要性 (Feature Importance) 審判時，能精準反查是哪一組參數的貢獻。