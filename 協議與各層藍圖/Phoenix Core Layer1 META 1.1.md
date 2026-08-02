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


📜 絕對介面契約 (schemas.py)

Layer 2 模型工廠與 Layer 1.5 特徵審判庭唯一認可的輸入格式為 MarketObservation。任何不符合此 Pydantic 契約的資料將在進入模型前被直接拋錯攔截。

1. DataConfig (超參數與特徵合約)

class DataConfig(BaseModel):
    base_asset: str = Field("BTC/USDT", description="驅動全局心跳的基準資產")
    target_assets: Dict[str, AssetMetadata] = Field(...)
    timeframes: List[str] = Field(["1m", "5m", "15m", "1h", "1d"])
    
    # 🚨 [防爆紅線] 暖機與動態特徵合約
    warmup_bars: int = Field(300, description="強制暖機 K 線數量，防範 EMA/MACD 等指標初始發散")
    raw_yaml: Dict[str, Any] = Field(..., description="解析後的 YAML 字典，供 FeatureProxy 進行動態特徵註冊")


2. MarketObservation (輸出形狀約束)

class MarketObservation(BaseModel):
    timestamp: int              # 全局基準 Heartbeat (Unix Timestamp 毫秒)
    asset_id: str               # 標的代碼 (e.g., 'BTC/USDT')
    
    # [核心約束] 以下四個 DataFrame 的 Rows 長度必須 100% 相同，且絕對不允許存在 NaN
    features_value: pl.DataFrame  # 數值特徵矩陣 (開高低收、成交量、動態載入之指標)
    features_mask: pl.DataFrame   # Boolean 矩陣 (True: 有效交易, False: 填補值)
    features_reason: pl.DataFrame # IntEnum 矩陣 (0: 正常, 1: 休市, 2: 熔斷, 3: 斷線)
    time_features: pl.DataFrame   # 時間語意矩陣 (d_tau, 事件時鐘)


📄 模組權責與 SOTA 規範：管線協調器 (pipeline.py)

此模組為 Layer 1 運算中樞，負責調度所有 Processor 與 Extractor。未來重構或跨層呼叫時，必須 100% 遵守以下簽名與內部物理邊界，嚴禁任何妥協。

1. 建構子約定 __init__(self, config: DataConfig)

必須綁定超參數：self.config = config。

必須將四大處理器實例化：self.encoder, self.aligner, self.calendar, self.clock。

動態特徵掛載紅線: 必須實例化 self.feature_proxy = FeatureProxy(config.raw_yaml)，將特徵解析權完全移交給 YAML 字典，嚴禁在此處硬編碼任何指標。

2. 實盤引擎約定 step(self, raw_1m_window: pl.DataFrame, macro_data: Dict) -> MarketObservation

🚨 強契約暖機斷言 (Warm-up Assertion Guard):

在執行任何運算前，第一行程式碼必須檢查 len(raw_1m_window) < self.config.warmup_bars。

若條件成立，必須強制引爆 RuntimeError，拒絕推論。這保證了神經網路絕對不會吃到未收斂的 EMA 或 RSI 雜訊。

執行順序絕對紅線 (不可顛倒):

遺失編碼: encoder.encode()

特徵代理注入: feature_proxy.extract_features()

時間語意萃取: clock.extract()

O(1) 空間約束與縫合: 截取最新一筆狀態 (.tail(1)) 後，才呼叫 aligner.align_timeframes 進行多尺度 Backward Join，極大化節省實盤算力。

合約封裝: 必須將最終的 DataFrame 精確拆解並封裝入 MarketObservation 中返回。

3. 回測引擎約定 batch(self, base_1m_history: pl.DataFrame, macro_history: Dict) -> pl.DataFrame

極致向量化: 必須對整張歷史大表依序執行 Encoder -> FeatureProxy -> Clock -> Aligner。嚴禁在 Batch 內使用 Python for 迴圈。

🚨 暖機期資料切除 (Warm-up Slicing Guard):

完成所有向量化運算後，由於前 $N$ 根 K 線的滾動特徵必定包含 NaN 或未收斂的假資料，必須在輸出前強制執行 final_df.slice(self.config.warmup_bars)。

這確保了交給 DatasetLoader 和 Layer 2 的巨大張量，在時間軸的第一個步長就是滿秩 (Full Rank) 且物理意義精確的完美特徵。


📄 模組權責與 SOTA 規範：基礎設施與 I/O 閘道

此子系統為數據的源頭，負責歷史回補與特徵的物理時間錨定。未來的任何擴充必須嚴格遵守「I/O 與 CPU 運算分離」的非同步併發架構。

1. 離線多資產資料湖建構器 (data_lake_builder.py)

系統定位: 非同步抓取、三維標記、多尺度重採樣、以及物理收盤時間 (Close Time) 的終極定義者。

多核心併發約定 (build_multi_asset_lake):

絕對併發紅線: 嚴禁寫出 for asset in assets: await self.process_single_asset(...) 這種會阻塞網路 I/O 的序列化迴圈。

必須建立任務清單 tasks = [self.process_single_asset(...) for ...]，並使用 await asyncio.gather(*tasks) 將所有資產的抓取任務平行發布至事件迴圈。

🚨 [SOTA] 物理收盤時間的 Schema 轉移 (Schema Shift):

歷史痛點: 若在推論階段 (實盤) 才去推算多尺度 K 線的結束時間，會遭遇時區錯亂、不同市場 Session 差異等災難。

實作紅線: 必須在 generate_multi_timeframes 降維重採樣時，強制寫入 close_time 欄位。

公式約定: (pl.col("datetime").dt.offset_by(tf) - pl.duration(milliseconds=1))。

物理意義: 將「這根 K 線何時才算真正結束、才能被看見」的因果律，直接刻印在 Parquet 檔案的硬碟裡。

Graceful Shutdown 約定:

在關閉非同步交易所連線 (await self.exchange.close()) 後，必須強制執行 await asyncio.sleep(0.25)，給予底層 aiohttp 釋放 TCP 連線池的時間，防止 Unclosed connector 報錯。

2. 幣安非同步閘道器 (fetchers/binance_client.py)

依賴綁定: 必須使用非同步版本的 CCXT：import ccxt.async_support (REST) 與 ccxt.pro (WebSocket)。

歷史回補指數退避 (fetch_ohlcv_with_backoff):

必須實作 while retries < max_retries: 的防禦迴圈。

觸發 RateLimitExceeded 或網路異常時，必須執行 await asyncio.sleep(2  retries) 進行指數退避延遲。

📄 模組權責與 SOTA 規範：多尺度時間縫合器 (aligner.py)

系統定位: ⚡ 防爆核心 (Anti-Lookahead Core)。負責將高階巨觀尺度 (5m, 1h...) 縫合回基準 1m 心跳，並保證絕對的因果隔離。

無狀態約束 (Stateless): 此類別不維護任何跨時間步的狀態緩存，所有操作必須基於傳入的 DataFrame 進行純函數 (Pure Function) 映射。因此不需要實作 __init__。

核心方法：align_timeframes(base_df: pl.DataFrame, macro_dfs: Dict[str, pl.DataFrame]) -> pl.DataFrame

此內部執行順序與運算子具有不可妥協的絕對紅線。

1. 前置處理與命名空間隔離

時間軸強制排序: 進入任何 Join 操作前，必須對 macro_df 執行 .sort("close_time")。

命名空間隔離 (Namespace Isolation):

僅保留 ["open", "high", "low", "close", "volume"] 與巨觀時間戳 close_time。

必須對所有的價量欄位強制加上後綴：pl.col(c).alias(f"{c}_{tf}")，避免與基準表的特徵產生覆寫碰撞。

2. 🚨 核心對齊防爆紅線 (Close-Time Backward Join)

嚴禁使用一般的 join 或 merge。

運算子鎖死: 必須精確呼叫 Polars 的高頻交易對齊函數：join_asof。

對齊邏輯鎖死:

必須使用 left_on="datetime" (基準的開盤時間)。

必須使用 right_on="close_time" (巨觀 K 線的物理結束時間，由 Data Lake 提供)。

策略參數必須鎖死為 strategy="backward"。

物理意義展示:
確保當前時間 10:04:00 只能對齊到 $\le$ 10:04:00 的最後一筆巨觀 close_time。因此 10:00 的 5m K 線 (其 close_time 為 10:04:59.999) 將被拒絕，系統會精準匹配到 09:55 的 5m K 線。這在數學上 100% 杜絕了未來視。

3. 資訊衰減追蹤器 (Staleness Tracking)

對齊完成後，必須立刻計算資料的老化程度，供 Layer 2 的 Attention 機制或 SSM 評估權重。

Polars 實作約定:

staleness_col = f"staleness_{tf}_mins"
df_aligned = df_aligned.with_columns(
    ((pl.col("datetime") - pl.col(macro_time_col)).dt.total_milliseconds() / 60000.0)
    .alias(staleness_col)
)


計算完畢後，必須 .drop(macro_time_col) 拋棄巨觀時間戳，保持矩陣乾淨。

4. 極限 NaN 防護 (Matrix Purity Guard)

在回測的最早期（例如開盤的前幾分鐘），高階巨觀 K 線根本還不存在，join_asof 會無可避免地產生 null。

防禦紅線: 在回傳前，必須執行全局的兩段式填充防禦：df_aligned.fill_null(strategy="forward").fill_null(0.0)。保證最終輸出的特徵滿秩無損。

📄 模組權責與 SOTA 規範：動態特徵路由代理 (feature_proxy.py)

此模組正式取代舊有的硬編碼特徵庫，負責解析 raw_yaml 並動態掛載您設計的高階特徵引擎 (H/G 雙接口架構)。這是實現「一次編寫，多 YAML 分流訓練」的核心中樞。

系統定位: ⚡ SOTA 中樞 (YAML-Driven Dynamic Feature Routing)。負責從 YAML 讀取 meta_features_plan，並動態映射載入 features_library 中的指標模組。

依賴與零 Pandas 紅線: 允許引入 polars, importlib。嚴禁在此模組寫死任何指標名稱 (如 RSI, MACD)。嚴禁使用 pandas。

核心方法與防爆紅線

1. 建構子預載入 (__init__)

Fail-Fast 模組預載入紅線:

嚴禁在實盤 step() 中動態執行 import，這會導致致命的 I/O 阻塞延遲。

必須在 __init__ 階段讀取 YAML 出戰名單，透過 importlib.import_module() 將所需的腳本載入 self.module_cache 字典中。

若模組不存在或缺乏 calculate (H接口) / adapt_xxx (G接口) 函數，必須直接引爆 AttributeError 或 ImportError，拒絕系統啟動。

2. 特徵提取與注入 (extract_features)

參數合併約定: 必須以 indicator_default_params 為基底，再用 meta_features_plan 中各別定義的 params 進行字典更新 (Override)。

🚨 命名空間防碰撞紅線 (Namespace Collision Guard):

當 YAML 中多次呼叫同一個指標 (例如 MACD_Fast, MACD_Slow) 時，G 接口輸出的欄位名稱會產生覆寫碰撞。

絕對防禦: 必須讀取 YAML 中定義的 uid (即 item["id"])。在收集 G 接口吐出的 pl.Expr 時，必須強制呼叫 .alias(f"{uid}_{feat_name}") 進行安全重新命名。

單次向量化紅線 (Single-Pass Execution):

必須將所有重命名後的 pl.Expr 收集在同一個 List 內。

遍歷完成後，僅允許執行一次 df.with_columns(exprs)，交由底層 Rust 引擎建構單一 DAG (有向無環圖) 進行多核極限平行運算。

📄 模組權責與 SOTA 規範：三維遺失編碼器 (missing_encoder.py)

系統定位: ⚡ 防爆核心 (3D Missingness Encoding & Absolute NaN Guard)。將「無資料」轉化為神經網路可理解的物理狀態。

核心方法與強型別約束

強型別狀態碼約束 (IntEnum Red Line):

嚴禁使用 Magic Numbers。必須使用 MissingReason(IntEnum)。

物理意義鎖死：0=NORMAL (正常), 1=CLOSED (休市), 2=HALTED (熔斷), 3=API_DOWN (斷線)。

遮罩判定 (Mask Condition): (pl.col("close").is_not_null()) & (pl.col("volume") > 0)。

🚨 絕對零 NaN 裝甲 (Absolute NaN Guard):

為防止第一筆資料就是空值導致 forward_fill 失敗，所有價量特徵 (open, high, low, close) 必須實施雙層防禦：fill_null(strategy="forward").fill_null(0.0)。

volume 缺失則直接 fill_null(0.0)。

單次圖優化 (Single Pass): Mask 運算、Reason 判定與 FFill 必須包裝在同一個 with_columns 陣列中執行。

📄 模組權責與 SOTA 規範：事件時鐘與時間變形 (time_clock.py)

系統定位: ⚡ SOTA 核心 (Event Clock & Time Deformation $\Delta\tau$)。打破均勻時間假設，提供市場物理時間的流逝速率。

核心特徵運算約束

1. 市場時間流逝率 (apply_time_deformation)

運算公式: $d\tau_t = (\sigma_t \times V_t) / \mu(\sigma \times V) + \epsilon$。

🚨 狀態凍結絕對防護 (Freeze Mask Red Line):

$d\tau_t$ 必須通過 features_mask 的審查。

必須使用 pl.when(pl.col("features_mask") == False).then(0.0) 進行覆寫。

物理意義: 當市場休市，時間凍結 ($d\tau_t = 0.0$)。確保未來 Layer 2 Mamba 模型進行 $e^{A\Delta}$ 離散化時退化為單位矩陣 $I$，強制記憶體狀態 $h_t = h_{t-1}$，徹底根除週末幻覺。

2. 事件時鐘標準化 (apply_event_clock)

週期壓縮紅線:

絕對禁止輸入未經縮放的「距離開盤分鐘數」(會導致梯度爆炸)。

必須將數值標準化並 .clip(-1.0, 1.0) 鎖死在神經網路最佳的激勵區間。

🛡️ 對 Layer 1.5 & Layer 2 的終極輸出保證

當 pipeline.py 完成並輸出 MarketObservation 時，我們對下游的所有決策與運算層下達以下絕對承諾：

無未來視 (Zero Look-ahead Leakage): 所有多尺度對齊已在資料庫層級錨定 close_time，實盤與回測時空絕對平齊。

形狀完美無損 (No NaN, Full Rank): 傳入的 value, mask, time 張量長度 100% 相同，絕不包含任何 null，無需在 PyTorch 中寫任何防呆 assert。

純淨的特徵溯源: 每一個 Columns 都帶有 YAML 賦予的 Unique ID，Layer 1.5 (LightGBM) 的 Feature Importance 可以精準追蹤特徵貢獻。

完美的狀態機映射 (Perfect State-Machine Mapping): 休市時的 $d\tau_t = 0$ 與 features_reason = 1，為後續所有的 Attention Mask、SSM 轉移矩陣與 HMM 凍結，提供了最精準的觸發信號。