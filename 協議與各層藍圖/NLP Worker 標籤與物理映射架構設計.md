Phoenix Core: NLP Worker 標籤與物理映射架構設計

系統定位: 獨立運行於 Layer 1 之外的旁支微服務 (Sidecar Worker)。
目標: 將雜亂無章的文本，轉換為結構化、去重、且具備物理意義的時間序列特徵。

🧠 核心命題解答與架構對策

1. 稀疏性與突發性 (Sparse & Burst)

痛點: 幾天沒新聞，突然 20 分鐘出三條。

對策: H/G 雙接口的 EMA 半衰期映射。
我們在 news_sentiment.py 中利用 EMA (指數移動平均) 處理。沒有新聞的日子，特徵值會隨時間平滑衰減至基準線 (0.0)。一旦突發新聞出現，會產生一個瞬間的 Spike (脈衝)，接著再度進入半衰期。這讓「離散的文字」變成了「連續的波動率」，神經網路就能完美消化。

2. 新聞叢集與重複 (Redundancy)

痛點: 20 分鐘內三家媒體報導同一件事 (例如：台積電法說會超預期)。如果單純相加，情緒分數會爆炸，導致模型誤判。

對策: 語意向量去重 (Embedding Deduplication)。
NLP Worker 在處理新文本時，必須先呼叫輕量級 Embedding 模型 (如 all-MiniLM-L6-v2)，並與過去 6 小時內的「新聞記憶庫」比對餘弦相似度 (Cosine Similarity)。

若相似度 $> 0.85$ (同一件事)：不增加新的情緒脈衝，僅微幅上調該事件的 Impact (代表事件發酵、媒體跟進)。

若相似度 $< 0.85$ (獨立事件)：視為全新脈衝寫入。

3. 異質性影響力與宏觀傳導 (Diverse Impact & Macro)

痛點: 油價跌影響台積電？急性利空 vs 慢性利空？

對策: 多維實體標籤 (Entity & Factor Tagging)。
NLP 不應該去「猜」油價對台積電是好是壞，NLP 只負責客觀標記。我們要求 LLM 解析新聞時，必須輸出以下標籤：

Entity: (如 TSMC, Oil, Fed)

Event_Type: (如 Earnings, Geopolitics, Supply_Chain)

Decay_Class: Acute (急性，半衰期 2 小時) 或 Chronic (慢性，半衰期 7 天)。

Layer 1 會將 Oil 的新聞特徵平行餵給台積電的模型，讓 Layer 1.5 的樹模型自己去發現「油價特徵」與「台積電收益」的隱性非線性關係。

4. 價格與新聞的時間錯位 (Price-News Lag)

痛點: 利多出盡 (Buy the rumor, sell the news)；或者新聞出了盤整很久才噴。

對策: 價格動能與新聞情緒的「特徵交叉 (Feature Crossing)」。
這是最致命的一點。單看新聞特徵必定破產。在寬表中，我們同時擁有 新聞情緒 (News_Bias) 與 價格乖離率 (BB_Pct_20)。

場景 A (內線早知道): 新聞發布前，價格已暴漲 (BB_Pct > 1.0)。當下 News_Bias 飆升，樹模型看到 (高價格乖離 + 極度利多)，會判定這不是追高信號，而是「利多出盡的倒貨信號」。

場景 B (突發利多): 價格在底部盤整 (BB_Pct = 0.5)，突然 News_Bias 飆升。樹模型判定這是「真實的突破驅動力」。

5. 隱性關聯與特徵聚合 (Hidden Links & Aggregation)

痛點: 同質性該疊加嗎？不同質性互不影響嗎？

對策: 獨立通道 + 非線性聚合 (Log-Sum)。

同質性聚合: 遇到多條台積電利多，不能無限線性疊加 (1+1+1=3)，必須使用 tanh 或對數聚合 (Log-sum)，使其收斂在 [-1.0, 1.0] 之間，模擬市場對同一利多的「邊際效應遞減」。

異質性保留: Global_Macro_News (如聯準會降息) 與 Specific_Asset_News (如台達電財報) 必須是寬表中的「兩個獨立欄位」。絕對不能把拔河的兩端加在一起變 0。保留獨立維度，神經網路才能學習到：「大盤極度恐慌時，個股的利多新聞無效 (被宏觀覆蓋)」。

🛠️ NLP Worker 輸出契約 (JSON Schema)

NLP Worker 解析完一篇新聞後，必須寫入 Database 或 Parquet 的標準格式：

{
  "timestamp": 1718000000000,
  "news_id": "news_uuid_12345",
  "entities_mentioned": ["TSMC", "Semiconductor", "Macro_Tech"],
  "event_type": "Earnings_Surprise",
  "decay_class": "Acute",
  
  "evaluation": {
    "sentiment_polarity": 0.85,  // -1.0 到 1.0
    "impact_magnitude": 0.9,     // 0.0 到 1.0 (事件震撼度)
    "novelty_score": 0.95        // 1.0 代表全新事件，0.1 代表媒體炒冷飯 (透過 Embedding 比對得出)
  }
}


這份資料進入 Layer 1 後，就會被 news_sentiment.py (特徵引擎) 轉換為 pulse = sentiment * impact * novelty，接著套用 EMA 變成一波波的連續漣漪！