# Layer 1.5 SOTA 參數高原優化所藍圖 (Master Architecture Specification)

此文件為 Phoenix 體系 Layer 1.5 參數高原優化所之權威架構藍圖。100% 遵照「論文無簡化 (Zero Paper Simplification)」、「零硬編碼 (Zero Hardcoding)」、「極速煙霧測試 (Fast Smoke Testing)」與「31 指標 MultiCharts 損益關聯」四大思想鋼印。

---

## Ⅰ. 架構設計與四階段生命週期 (Four-Phase Lifecycle)

Layer 1.5 專為量化策略尋找**「高勝率、高穩定度、低尖銳過擬合」的參數高原 (Parameter Plateau)**，分為以下四大階段：

```
[ Phase 1: 物理資料切割 ] ──> [ Phase 2: Regime 動態錨定 ] ──> [ Phase 3: SOTA 尋優引擎 ] ──> [ Phase 4: 死亡競技場 & MC 31 指標 ]
 (L1_5_data_prep.py)            (RegimeEvaluator)             (HEBO/TuRBO/CMA-ES/TPE)         (DeathmatchTournament)
```

### Phase 1: 物理資料切割與記憶體極致解耦 (Data Splitting)
- **機制**：調用 `scripts/L1_5_data_prep.py`，依據 YAML 設定之 `train_start`, `train_end` 與動態 `warmup_bars` (如 300 根)，在硬盤切出 15 份 (5 個尺度 x 3 個切分) 超輕量 Parquet 檔案。
- **效益**：尋優階段零記憶體洩漏 (Zero Memory Leakage)，支援 Bash 級別無腦開滿多個平行進程 (`&`)。

### Phase 2: Regime 動態錨定與代理評估 (Regime & Proxy Evaluation)
- **機制**：透過 1d 尺度之波幅與趨勢特徵（如 `AVO`, `ADX`）進行 KMeans 狀態聚類，作為 1m/5m/15m/1h 子尺度的物理錨定。
- **代理**：支援 `Qlib_Direct_RankIC` (全特徵向量化 Rank IC/IR) 與 `LightGBM` / `Ridge` 迴歸代理。

### Phase 3: SOTA 尋優引擎矩陣 (100% 論文原著擬真)
系統拒絕任何簡化版 Proxy Wrapper，100% 原汁原味實作 4 大頂級論文演算法：

1. **Native HEBO (NeurIPS 2020 總冠軍引擎)**：
   - 原生庫調用 `from hebo.optimizers.hebo import HEBO`。
   - **輸入/輸出空間拉直**：Box-Cox / Yeo-Johnson Power Transform。
   - **噪聲建模**：異方差高斯過程 (Heteroscedastic GP)，懲罰陡峭孤峰。
   - **多目標採集**：Pareto Front 多目標 (EI / UCB / PI) 搭配 NSGA-II 候選生成。
   - **32-Workers 併發吃滿**：採用 `opt.suggest(n_suggestions=32)` 批次採集搭配 `ThreadPoolExecutor` 多線程並行評估。

2. **TuRBO-m (Uber AI 多信賴域尋優)**：
   - 實作動態信賴域超立方體 (Trust Region Hypercube) 擴張與縮減。
   - **銷毀重啟機制**：當信賴域長度 $L < L_{\min}$，觸發 `best_params = None` 銷毀當前信賴域，發動 Sobol 全局重採樣 (Re-spawn)。

3. **CMA-ES (Hansen IPOPOP 演化策略)**：
   - 協方差矩陣旋轉適應 (Covariance Matrix Adaptation)。
   - **IPOPOP 重啟機制**：當種群收斂或停滯，自動觸發 `restart_strategy="ipop"`，種群規模翻倍擴張 ($\lambda_k = 2^k \lambda_0$)。

4. **TPE (Bergstra 2011 多變量樹狀 Parzen 估計器)**：
   - 啟用 `multivariate=True`, `group=True`, `constant_liar=True`，解耦高維參數相干性。

5. **SQLite WAL Mode 無鎖並行**：
   - `searcher.py` 自動注入 `PRAGMA journal_mode=WAL;`，支援 32 個進程同時對 Optuna DB 無鎖讀寫。

### Phase 4: 死亡競技場與 MultiCharts 31-Metric 損益關聯
- **解展平防線 (Unflattening Defense)**：
  - 尋優結果之 Flat Params 必須通過 `SearchSpaceParser.unflatten()` 還原為 Nested 字典，徹底杜絕 `KeyError: 'source'`！
- **SAM 高原擾動度量 (Plateau Smoothness)**：
  - 對參數進行 10% 鄰域幾何擾動，採集 5 個鄰居節點之平均適應度，扣除方差懲罰 (`sam_score = mean_fitness - sharpness_penalty`)。
- **MultiCharts 31-Metric PnL 報告烙印**：
  - 產出之 YAML 自動包含 Net Profit, Gross Profit/Loss, Adjusted Net Profit, Select Net Profit, Account Size Required, Return on Account, Return on Initial Capital, Max Strategy Drawdown ($/%), RoMAD, Profit Factor, Slippage, Commission 等 31 項專業盤測指標。

---

## Ⅱ. 快慢線指標標註規範 (Fast/Slow Indicator Rules)

### 規則概述
1. **代碼偵測**：位於 `core/layer1_data/features_library/Indicators/` 的檔案，只要包含 `fast_period` 與 `slow_period`，自動判定為快慢線指標。
2. **YAML 標註**：在 `indicator_default_params` 中，對於快慢線指標需標註 `has_fast_slow: true`。
3. **驗證與修復工具**：
   ```bash
   python -m scripts.validate_fast_slow_yaml --check   # 檢驗
   python -m scripts.validate_fast_slow_yaml --fix    # 自動補全
   ```

---

## Ⅲ. 鋼印約束與煙霧測試 (Smoke Testing)

在進行正式 1000 次尋優前，AI Agent 必須發動極速煙霧測試腳本：
```bash
python scratch/verify_all_searchers.py
```
可以在 5 秒內完成「Phase 3 尋優 $\rightarrow$ Phase 4 解展平 $\rightarrow$ MultiCharts 損益評估 $\rightarrow$ YAML 導出」全鏈條 Fast Pass，零 Token 浪費！
