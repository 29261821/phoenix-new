# Phoenix 系統：Layer 1.7 新聞採集與實體對齊架構規範

## 1. 核心使命與 MCP 架構定位
Layer 1.7 是非結構化資料的「漏斗」。它的職責是 24 小時不間斷地從多個來源抓取文字訊息，並將其標準化、實體化（打標籤）、時間對齊，最後存入中心化資料庫，供 Layer 1.8 調用。
為了現實落地，本層採用 **微服務 / MCP (Model Context Protocol)** 架構，與交易主引擎完全物理隔離。

## 2. 系統解耦架構 (The MCP Decoupling)
在實盤中，新聞的發生是異步的（Asynchronous），而 K 線的生成是同步的（Synchronous）。
- **MCP Server 1 (Scraper Node)**: 輕量級的事件驅動伺服器，專職訂閱 WebSocket (如 X/Twitter API, 幣安新聞, RSS 訂閱)。一收到新聞，立即壓上**伺服器接收時間戳 (Received Timestamp)**，推入 Redis 或 Kafka 佇列。
- **MCP Server 2 (Entity Tagger)**: Python 服務，消費佇列。使用輕量級 NLP / 正則表達式進行「實體對齊 (Entity Alignment)」。判斷這則新聞屬於 `BTC`, `ETH`, 還是 `Macro` (總體經濟/美聯儲)。
- **持久化層**: 寫入 PostgreSQL 或 SQLite (`news_raw_events` 表)。

## 3. 致命缺陷防禦 (Timestamp Shift Defense)
### 缺陷：時間戳穿越 (Timestamp Leakage)
- **錯誤作法**：新聞發布時間是 10:04:59，系統將其合併到 `10:00:00 ~ 10:05:00` 的 K 線中進行訓練。
- **後果**：模型在預測 `10:05:00` 之後的走勢時，偷看到了 10:04:59 的新聞，但在實盤中，10:05:00 根本來不及運算這則新聞，導致實盤崩潰。
- **修正 (Strict Shift)**：
  1. 所有新聞採用 `received_timestamp`。
  2. 新聞只能合併到 **下一個** 完整的 K 線。10:04:59 的新聞，其特徵必須附加在 `10:05:00` (Open) 的 K 線上，供模型預測 `10:05~10:10` 的走勢。

## 4. 輸出介面 (The Interface)
`[event_id, received_time, tradable_time, entity_tags, raw_text, source_weight]`
這將成為 Layer 1.8 訓練層的唯一合法輸入來源。
