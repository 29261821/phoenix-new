Phoenix Core: 另類數據 (News) 攝取與 Bronze 層架構 META - Part 1

系統定位: NLP Worker 的前置武裝部隊，負責在混亂的網際網路中，安全、不漏接、具備容錯能力地將異質新聞抓取並封存。
目標資產: 聚焦 2330 (TSMC)、2308 (Delta)、2408 (Nanya) 及全球宏觀，具備無縫擴充至全台股/美股的彈性。
架構標準: 16000x16000 SOTA (機構級 Data Lakehouse 規範)

🏛️ 模組 1：異質來源轉接器矩陣 (Heterogeneous Adapter Pattern)

網路世界的資料格式極度混亂。我們絕對不能把爬蟲邏輯和儲存邏輯寫在一起。必須強制導入 Adapter (轉接器) 設計模式。

SOTA 目錄結構預覽

phoenix/
└── core/
    └── layer0_ingestion/
        ├── news_adapters/
        │   ├── base_adapter.py      # [合約] 所有來源必須實作的 Abstract Base Class
        │   ├── finmind_api.py       # [結構化] 呼叫 FinMind TaiwanStockNews
        │   ├── moneydj_scraper.py   # [非結構] 解析 MoneyDJ HTML
        │   └── rss_streamer.py      # [實盤] 訂閱即時 RSS Feed
        └── ingestion_engine.py      # [中樞] 管理異步併發、Proxy 輪替與指數退避


轉接器絕對契約 (base_adapter.py)

任何新增的資訊源，無論是付費 API 還是野雞網站，都必須回傳一個嚴格的 RawArticle Pydantic 物件。這保證了 ingestion_engine 不需要去理解 HTML 標籤，它只負責把物件存進硬碟。

🧱 模組 2：Bronze Vault (青銅物理金庫) 儲存契約

抓下來的資料，在進行任何 NLP 解析或清洗前，必須先進入 Bronze Layer (青銅層)。
青銅層的最高哲學：絕對不丟棄任何原始資訊 (No Data Left Behind)。

分區策略 (Partitioning) 的致命糾正

過去的錯誤: 按照股票代號分區 (ticker=2330/)。

16000x16000 SOTA 修正: 新聞是「多對多 (Many-to-Many)」的實體！一篇講述「台積電與南亞科合作」的新聞，如果按 ticker 存會產生兩份重複的硬碟佔用，後續 NLP 算 Embedding 也會算兩次，浪費巨量算力。

絕對規範: 必須按照 「爬取發生的日期 (Ingestion Date)」 進行 Parquet 分區。關聯性（標籤）留在後面的 Silver 層處理。

infra/data_lake/bronze_news/
├── ingest_date=2026-06-29/
│   ├── finmind_001.parquet
│   └── moneydj_001.parquet


青銅層 Schema 絕對紅線 (Polars / Parquet)

Bronze 層不負責解析股價影響，只負責「防腐」。

bronze_news_schema = {
    # --- 1. 物理溯源 ---
    "surrogate_key": pl.Utf8,    # UUID4 (系統內部唯一流水號，防撞擊)
    "source_id": pl.Utf8,        # 來源標識 (e.g., "finmind", "moneydj")
    "source_url": pl.Utf8,       # 原始連結
    
    # --- 2. 雙時態戳記 (Bitemporal) ---
    "publish_time_raw": pl.Utf8, # 來源聲稱的發布時間 (保留字串原始格式，如 "2026/06/29 14:00" 或 "14 hours ago"，留給 Silver 層解析)
    "ingestion_time": pl.Int64,  # [絕對真理] 我們的伺服器收到 Response 的 UTC 毫秒 (絕不竄改)
    
    # --- 3. 原始數據保留 (Disaster Recovery Red Line) ---
    # 這是防止爬蟲解析錯誤的終極防線。如果半年後發現當初漏抓了「作者」欄位，
    # 只要 raw_payload 還在，我們就能重新跑一次 Silver 轉換，而不用重新發出幾百萬次 HTTP 請求。
    "raw_payload": pl.Binary,    # 壓縮後的原始 JSON Response 或 HTML 網頁源碼 (Zstd 壓縮)
    
    # --- 4. 粗略擷取 (可能包含 HTML 殘渣) ---
    "title_raw": pl.Utf8,        
    "content_raw": pl.Utf8,      
    "status_code": pl.Int16      # HTTP 狀態碼 (用於監控，若為 403 則 content 為空，排入 DLQ)
}


⚙️ 模組 3：非同步攝取引擎 (Ingestion Engine)

負責回補 5 年新聞的 ingestion_engine.py 必須具備與 Layer 1 binance_client.py 相同等級的容錯能力，甚至更強，因為新聞網站的反爬蟲機制遠比交易所嚴苛。

100分級防禦規範：

全域令牌桶 (Global Token Bucket): 控制每秒發送的 HTTP 請求總數（例如 FinMind 限制 300次/5分鐘），超過則強制 asyncio.sleep。

死信佇列 (Dead Letter Queue, DLQ): 如果某篇 HTML 解析失敗或遭遇 403 封鎖，該筆任務不得被直接丟棄。必須將其 URL 與錯誤原因寫入 bronze_dlq.parquet。系統維護人員可藉此修復爬蟲，並針對 DLQ 重新發起抓取。

冪等性 (Idempotency): 引擎在抓取前，必須先透過 Bloom Filter 或查詢 Bronze Lake，確認該 source_url 是否已存在。即使引擎崩潰重啟，重新執行 5 年回補腳本，也絕對不會產生重複的 Parquet Row。


Phoenix Core: 另類數據 (News) 實體解析與 Silver 層架構 META - Part 2

系統定位: NLP Worker 的無塵室。將 Bronze 層的髒數據清洗、去重、解析實體，並鑄造絕對無未來視的 Point-in-Time (PIT) 雙時態鋼印。
架構標準: 16000x16000 SOTA (機構級 Data Lakehouse 規範)

🏛️ 模組 1：實體解析矩陣 (Entity Resolution Matrix)

新聞文本極少直接寫出股票代號 (Ticker)。它可能寫「台積電」、「TSMC」、「神山」，或是寫「南亞科」、「南科 (需與南部科學園區做歧義排除)」。

知識圖譜映射 (Knowledge Graph Mapping)

在進入 Silver 層時，必須經過 entity_resolver.py 模組。這不是用簡單的 if "台積電" in text 就能解決的，必須建立 SOTA 級別的 Regex/Token 映射字典：

# entity_resolver.py 概念核心
ENTITY_GRAPH = {
    "2330": {
        "primary": ["台積電", "TSMC", "2330.TW"],
        "aliases": ["護國神山", "台積"],
        "anti_patterns": ["台積電設備廠", "台積電供應鏈"] # 排除誤判
    },
    "2308": {
        "primary": ["台達電", "Delta Electronics", "2308.TW"],
        "aliases": ["台達"],
        "anti_patterns": ["台達化"] 
    },
    "2408": {
        "primary": ["南亞科", "Nanya Technology", "2408.TW"],
        "aliases": ["南科 (需搭配上下文判斷)"],
        "anti_patterns": ["南部科學園區", "南亞塑膠"]
    }
}


輸出紅線: 每一篇新聞經過解析後，必須產出一個 tagged_entities: List[str] 欄位 (例如: ["2330", "2308"])，這是未來 Layer 1 路由特徵的唯一憑證。

⚔️ 模組 2：Point-in-Time (PIT) 雙時態中樞

這是華爾街對付另類數據的核心武器。新聞具有三種極度危險的時空陷阱：

隱形修改 (Stealth Edits): 媒體在 09:30 發布新聞，10:30 偷偷加上「營收不如預期」，但發布時間依然標示 09:30。

延遲收錄 (Late Arriving Data): 13:00 的新聞，我們的 API/爬蟲因為網路異常，遲至 13:05 才抓到。

未來回補 (Look-ahead Backfill): 2026 年回補 2021 年漏抓的新聞。

雙時態防禦演算法 (bitemporal_engine.py)

為了徹底解決上述問題，Silver 層必須強制執行以下運算：

正規化發布時間 (publish_time): 將 Bronze 層的字串解析為標準 UTC Unix 毫秒。

繼承攝取時間 (ingestion_time): 直接沿用 Bronze 層不可篡改的系統時間。

計算內容指紋 (content_hash): SHA-256(Title + Content)。

🚨 絕對覆寫防禦 (Upsert/Append-Only Logic):
當資料從 Bronze 轉入 Silver 時：

如果 news_id 不存在，直接寫入。

如果 news_id 存在，但 content_hash 不同（抓到了媒體的隱形修改！），絕對不能覆蓋舊資料。必須產生一筆新的紀錄 (New Row)，其 publish_time 不變，但 ingestion_time 為最新抓到的時間。

未來回測時，系統只認 ingestion_time。這樣，10:00 的 K 線絕對看不到 10:30 才被修改的文字內容。

🧱 模組 3：Silver Vault (精煉白銀層) 儲存契約

經過清洗、解析與雙時態錨定後，資料存入 Silver Parquet。這份資料已經完全結構化，去除了 HTML，並準備好隨時餵給 NLP Worker。

Silver 層 Schema 絕對紅線 (Polars / Parquet)

import polars as pl

silver_news_schema = {
    # --- 1. 唯一識別與防偽 ---
    "surrogate_key": pl.Utf8,    # UUID (Silver 層專屬流水號)
    "bronze_ref_id": pl.Utf8,    # 溯源回 Bronze 層的 ID (用於除錯與重新清洗)
    "source_id": pl.Utf8,        # 來源 (e.g., "finmind")
    "content_hash": pl.Utf8,     # SHA-256 指紋 (防隱形修改)
    
    # --- 2. PIT 雙時態核心 (神聖不可侵犯) ---
    "publish_time": pl.Int64,    # [事件時間] 媒體宣稱發布的 UTC 毫秒
    "ingestion_time": pl.Int64,  # [系統時間] 我們實際掌握該資訊的 UTC 毫秒
    
    # --- 3. 實體與內容 ---
    "tagged_entities": pl.List(pl.Utf8), # 陣列: ["2330", "2408", "Macro"]
    "title_clean": pl.Utf8,      # 去除 HTML、特殊符號後的純淨標題
    "content_clean": pl.Utf8,    # 去除 HTML、廣告尾綴後的純淨正文
    
    # --- 4. 基礎中繼資料 ---
    "word_count": pl.Int32,      # 字數統計 (用於過濾太短的無效快訊)
    "is_duplicate": pl.Boolean   # 是否為通稿 (透過跨來源比對 content_hash 判定)
}


🔄 分區策略 (Partitioning)

與 Bronze 相同，Silver 層依然保持以 ingestion_date (我們知道這件事的日子) 進行 Parquet 分區，而不是 publish_date。這確保了時光機 (Time-Travel) 回測的絕對正確性。

Phoenix Core: 另類數據 (News) NLP 解析與 Gold 層架構 META - Part 3

系統定位: 將 Silver 層純淨的文本，轉化為「連續、去重、具備物理衰減特性」的時間序列信號，並與 Layer 1 寬表進行無縫對齊的終極橋樑。
架構標準: 16000x16000 SOTA (Gold Layer 契約與非同步對齊矩陣)

🏛️ 模組 1：NLP Worker 的煉金契約 (The Gold Vault)

NLP Worker (無論底層是調用 GPT-4o 還是本地微調的 FinBERT) 絕對不允許直接把資料塞給 Layer 1。它必須將計算結果寫入 Gold Layer (黃金特徵層)。

🚨 離線與非同步隔離紅線

絕對隔離: NLP Worker 是一個獨立的 Process/Container。它透過監聽 Silver 層 Parquet 的更新來觸發運算。

計算成本控制: 一篇新聞只會被 LLM 計算一次。算出的向量與情緒分數會被永久固化在 Gold 層。這讓 Layer 1 在進行 5 年回測時，能在幾毫秒內讀取完畢，而不用等待幾個月的 LLM API 呼叫。

Gold 層 Schema 絕對紅線 (Polars / Parquet)

這份合約定義了 Layer 1 將會接收到什麼樣的信號。

import polars as pl

gold_news_schema = {
    # --- 1. 溯源與時態 (繼承自 Silver 層的不可篡改鋼印) ---
    "surrogate_key": pl.Utf8,    # 唯一識別碼
    "ingestion_time": pl.Int64,  # [絕對真理] 我們的伺服器實際掌握該資訊的 UTC 毫秒
    "tagged_entity": pl.Utf8,    # 實體 (如 "2330"。注意：若一篇新聞包含兩個實體，在這裡必須展開為兩列 Row)
    
    # --- 2. 語意去重與新穎度 (Deduplication & Novelty) ---
    "embedding_vector": pl.List(pl.Float32), # e.g., 384 維的 MiniLM 向量
    "novelty_score": pl.Float32, # 新穎度 [0.0 ~ 1.0]。若與過去 6 小時的新聞高度相似，分數趨近 0。
    
    # --- 3. 核心量化信號 (The Alpha) ---
    "sentiment": pl.Float32,     # 情緒極性 [-1.0 (極端利空) ~ 1.0 (極端利多)]
    "impact_magnitude": pl.Float32, # 影響力/震撼度 [0.0 ~ 1.0]。由 LLM 判定這件事的嚴重級別。
    "decay_halflife": pl.Int32   # 預估半衰期 (分鐘)。例如「財報超預期」半衰期 720 分鐘，「工廠火災」半衰期 120 分鐘。
}


⚙️ 模組 2：實盤與回測的編排器 (Orchestrator)

新聞是「非均勻分佈」的事件。可能半夜 3 小時沒新聞，也可能早上 09:00 一分鐘內湧入 50 篇。

1. 歷史回測 (Batch Mode)

pipeline.py 在建構 5 年的寬表時，會直接讀取 gold_news_signals.parquet，將 5 年的信號一次性載入記憶體備用。

2. 實盤推播 (Streaming Mode)

在實盤中，news_streamer.py 透過 WebSocket 或高頻 RSS 輪詢抓到新聞後，會「光速貫穿」三層：
Bronze 寫入 -> Silver 清洗與雙時態標記 -> NLP Worker 即時推理 -> 寫入 Redis/Gold Parquet。
此過程必須在 500 毫秒 內完成，確保 Layer 1 的 1m 心跳在下一次脈動時，能及時捕捉到這筆新聞信號。

🧱 模組 3：Layer 1 的終極握手協議 (The Handshake)

當資料安穩躺在 Gold 層後，Layer 1 的 aligner.py (或是專屬的 news_sentiment.py extractor) 該如何把它縫合進 1m K 線寬表中？

🚨 Backward Join 與雙時態對齊防爆紅線

當 Layer 1 準備縫合 10:04 的 1m K 線時：

對齊基準: 絕對不是使用新聞宣稱的 publish_time，而是 強制使用 ingestion_time。

因果律保障: 如果媒體在 09:30 發布新聞，但偷偷在 10:30 修改內文 (被 Silver 層抓到並 Upsert 產生一筆 ingestion_time=10:30 的新紀錄)，那麼 10:04 的 K 線在執行 join_asof(strategy="backward") 時，只會匹配到 09:30 的舊版新聞特徵。10:30 的新特徵必須等到 10:31 的 K 線才能看見。

特徵連續化 (The G-Interface Pulse)

新聞縫合進來後，在 1m 寬表上會呈現大量 Null (沒新聞的分鐘) 和偶爾的 Spike (有新聞的分鐘)。
在 features_library/news_sentiment.py 中，必須透過 Polars 原生的指數衰減函數，將離散的脈衝轉換為平滑的波動：

# 概念映射：
# pulse_t = sentiment * impact_magnitude * novelty_score
# feat_news_bias = pulse_t.ewm_mean(half_life=decay_halflife)


這樣一來，神經網路 (TCN 或 Mamba) 就不會看到突兀的孤立點，而是看到如同丟入石頭般，一圈圈平滑擴散並逐漸衰減的「情緒漣漪」。