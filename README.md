# Phoenix Core v2.0-Pro 專案架構

本專案採用「核心共用 + 個人工作區隔離」的多人協作架構。
為避免 5 人開發時發生程式碼衝突與污染，請嚴格遵守以下工作區規範：

## 🗂 目錄架構與職責分配

### 1. 核心共用區 (Shared Core) - 所有人共用
- `core/`：系統最核心的引擎、基礎建設（Layer 1 ~ Layer 6）。請勿在這裡放個人測試腳本。
  - `core/layer1_data/`：數據處理層（如 `data_lake_builder.py`）。
- `configs/`：全域設定檔（如 `assets/TW_STOCKS.yaml`）。所有的設定都必須抽離至此，嚴禁在 Code 中硬編碼。
- `infra/`：基礎設施層與最終產出的資料湖 (`data_lake/parquet_store/`)。

### 2. 個人工作區 (Personal Workspaces) - 隔離開發
每個開發者擁有自己的專屬資料夾，用來放置原始數據、測試腳本與實驗特徵。
- `zhou_data_engineering/`：Zhou 專屬的工作區。
  - 負責：台股原始數據收集 (`market_data_lake/`)、個人指標實驗 (`ocean_example/`)。
- *(其他成員的工作區請依照此命名慣例建立)*

---

## 🚀 常用共用指令 (Commands)

### Layer 1: Data Lake Builder (台股數據管線)
當你在自己的工作區準備好原始數據後，請使用以下指令呼叫**共用的核心管線**，系統會自動套用 11 大 SOTA 防線並產出最終特徵至 `infra/data_lake/`。

**執行方式 1：自動掃描所有 YAML (推薦)**
```bash
python3 core/layer1_data/data_lake_builder.py
```

**執行方式 2：指定特定台股標的**
```bash
python3 core/layer1_data/data_lake_builder.py --assets 2330 2308 2454
```

> **提示**：Zhou 已經在他的資料夾內準備了一鍵執行的腳本，請參考 `zhou_data_engineering/run_pipeline.sh`。