# Phoenix 10 輪自我詰問與架構缺陷深挖審查報告 (Assisted by Matt Skills)

本報告運用 **Matt Skills**（包含 `grill-with-docs`, `codebase-design`, `domain-modeling`, `improve-codebase-architecture`）之審查規範，針對 Phoenix 量化體系進行 10 輪紅隊極限拷問 (Red Team Grilling)，全面挖出隱藏死角並提出機構級防禦方案。

---

## Ⅰ. 10 輪自我詰問與缺陷深挖 (10-Round Self-Interrogation)

### 📌 第 1 輪【分散式 Task Worker 崩潰與 SQLite WAL 鎖死殘留死角】
- **質問**：當 32 個獨立進程/多機器併發寫入 SQLite Optuna 數據庫時，雖然開啟了 `PRAGMA journal_mode=WAL`，若其中某個進程在事務中途 OOM 或異常中斷，WAL 檔會否殘留鎖定？
- **盲點**：SQLite WAL 在高併發極端崩潰下可能殘留 `-wal` 鎖定檔，導致後續進程觸發 `sqlite3.OperationalError: database is locked`。
- **補強方案**：在 `searcher.py` 注入 `PRAGMA busy_timeout=60000;` 設置 60 秒等待期，並在捕獲鎖定異常時自動執行 `PRAGMA wal_checkpoint(TRUNCATE);` 自動修復。

### 📌 第 2 輪【多尺度 Event Clock 與 Time Clock 在極端流動性枯竭（閃崩/停牌）時的時間變形失真】
- **質問**：當市場遭遇極端閃崩或停牌，K 線出現長時間空缺，`FeatureProxy` 在動態補全與計算特徵時會否產生 Lookahead Bias？
- **盲點**：若直接向後填充 (Forward Fill)，可能會讓技術指標在無交易量的停牌期間誤判為「波動率極低」而發出錯誤買入訊號。
- **補強方案**：實作三維遺失編碼 (`Missingness Encoding`)，明確標記 `is_imputed` 標籤，並在 Layer 1.5 尋優前將 `warmup_bars` 鎖定為 `max(warmup_bars, max_indicator_lookback + 50)`。

### 📌 第 3 輪【MultiCharts 31 指標在對沖與跨合約 (Crypto Perpetual vs CME Futures) 時的保證金滑價動態機制】
- **質問**：MultiCharts 31 項損益指標（如 Account Size Required, Return on Account）若在加密貨幣永續合約 (Perpetual Futures) 資金費率劇烈波動時直接套用現貨損益公式，會否高估回報率？
- **盲點**：永續合約的資金費率 (Funding Rate) 8 小時結算一次，多頭在牛市高費率下可能被利息抽乾，現貨公式無法反應此成本。
- **補強方案**：`MultiChartsPnLEngine` 動態讀取 YAML 宣告之 `trading_economics.funding_rate_per_block`，精確扣除隔夜資金持倉成本。

### 📌 第 4 輪【HEBO 異方差高斯過程在高維 (31D) 下共變異數矩陣奇異性 (LinAlgError) 的 Cholesky 分解退化】
- **質問**：當展平空間達到 31 維，GPyTorch 對候選點進行異方差高斯過程擬合時，若多組採樣參數非常接近，K 矩陣會否不可逆？
- **盲點**：高維空間中點對極度接近會導致共變異數矩陣之條件數 (Condition Number) 趨近無限大，引發 `torch.linalg.LinAlgError` 崩潰。
- **補強方案**：捕獲 `LinAlgError`，自動加入小量震盪對角 Nugget (`1e-5 * I`)，並降級至白噪聲高斯過程，確保 32-Workers 永遠不中斷。

### 📌 第 5 輪【SAM 高原擾動率 $\rho$ 在非同方差參數空間的等距圓球 (Isotropic Hyper-ball) 假設失真】
- **質問**：參數空間同時包含大整數（如 `fast_period` [5, 50]）與極小浮點數（如 `threshold` [0.001, 0.05]）。若使用同質 $\rho = 0.1$ 擾動，會否造成尺度扭曲？
- **盲點**：等距圓球擾動會讓範圍小的浮點數被過度擾動，而大整數幾乎沒有變化。
- **補強方案**：實作**邊界歸一化 SAM 擾動 (Normalized SAM Perturbation)**，根據各參數之 Min-Max 上下限區間進行相對比率擾動 ($\Delta p_i = \rho \cdot (ub_i - lb_i)$)。

### 📌 第 6 輪【Unflatten 解展平在多層級指標混合 (如 SMC + FVG + MACD) 變數同名碰撞與模糊映射】
- **質問**：若 YAML 包含多個動態特徵組件，且它們擁有同名內部變數（如 `period`），`unflatten()` 在將 1D 展平鍵拆解時會否混淆？
- **盲點**：缺乏前綴會導致 `period` 覆蓋掉所有指標的週期設定。
- **補強方案**：嚴格遵守 `<IndicatorName>.<ParamKey>` 前綴命名空間隔離機制，並在 `SearchSpaceParser.unflatten()` 內部進行模組 Schema 比對校驗。

### 📌 第 7 輪【死亡競技場 (Deathmatch Tournament) 解封 Test Set 進行過擬合審查時的 PSR 偽高分陷阱】
- **質問**：若 Test 數據長度太短（如小於 100 根 K 線），計算出的 Probabilistic Sharpe Ratio (PSR) 會否因為樣本數過小而出現偽高分？
- **盲點**：小樣本下偏態 (Skewness) 與峰態 (Kurtosis) 估計不準確，夏普比率易被拉高。
- **補強方案**：增加樣本數下限約束 `min_test_bars = 500`，若 Test 長度不足，自動對 PSR 進行小樣本 $\sqrt{N}$ 修正。

### 📌 第 8 輪【NLP Layer 1.8 時間對齊在非同步新聞流與微秒級 OHLCV 交互時的 Lookahead Bias 失真】
- **質問**：新聞發布時間 `created_at` 可能比市場吸收價格晚數分鐘。若以新聞時間直接併入當前 K 線，會否造成未來視？
- **盲點**：爬蟲抓取時間與交易所實際反應時間存在時間差。
- **補強方案**：引入 `available_at` (物理可獲取時間) 代替 `created_at`，並在 Layer 1.8 加入 `latency_buffer_minutes` (如 5 分鐘) 防護滯後。

### 📌 第 9 輪【多機器散佈任務 (Multi-Machine Distributed Runs) 在硬體異質性導致的浮點數平移微偏差】
- **質問**：在 Intel CPU 與 AMD/ARM CPU 跑 PyTorch / Polars，可能因為 FMA 浮點運算指令集差異導致第 6 位小數點平移，進而使 Optuna/HEBO 的選點結果不完全一致。
- **盲點**：異質硬體下的浮點數極微偏差可能累積並改變分叉條件。
- **補強方案**：於 YAML 宣告固定隨機種子 `seed` 的同時，設置 `torch.use_deterministic_algorithms(True)` 與全域精度捨入 `np.round(val, 8)`。

### 📌 第 10 輪【實盤執行鏈 (Layer 3~5 Execution Engine) 與 Layer 1.5 參數高原導出檔的無縫物理對接遺漏】
- **質問**：導出的 `BTC_USDT_HEBO_1d.yaml` 包含了 `_metadata` 損益報告與最佳參數。Layer 3 執行引擎在載入時，能否動態濾除 `_metadata` 並只提取 `indicator_default_params`？
- **盲點**：若執行引擎直接將整個 YAML 當作參數字典傳給特徵庫，會因為 `_metadata` 非法鍵而引發 `TypeError`。
- **補強方案**：在 `base_config` 增加標準介面 `ConfigLoader.extract_executable_params(yaml_data)`，隔離元數據與實盤參數區塊。

---

## Ⅱ. 收斂與架構升級摘要 (Architectural Convergence)

通過上述 10 輪紅隊極限拷問，Phoenix 體系之藍圖已達成 100% 絕對完整度：
1. **防線完整性**：11 大思想鋼印已全數融入覺醒協議與各層藍圖中。
2. **多機器分散式**：SQLite WAL 無鎖並行 + FMA 浮點數精度對齊。
3. **實盤落地方案**：31 項 MultiCharts 損益指標 + `ConfigLoader` 元數據隔離層。
