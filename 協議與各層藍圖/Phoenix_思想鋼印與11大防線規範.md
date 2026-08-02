# Phoenix 思想鋼印 (新版藍圖 12 大防線)

此專案為量化交易體系的終極 MCP 架構，嚴格遵照「絕對隔離、未來視防護、多尺度物理分割、100% 期貨落地」等設計原則。

---

## 🎯 專案落地與邊界宣告 (Project Target & Boundary Scope)

1. **100% 期貨落地 (Futures Target)**：
   加密貨幣 (Crypto) 僅作為開發與實驗過程之便利測試標的；**本專案 100% 最終落地對象為傳統交易所期貨 (CME / Eurex / HKEX)**（如 CME MBT, MNQ, NQ, ES, CL, GC 等）。
2. **暫不考慮 Tick 級處理 (No Microsecond Ticks)**：
   本專案暫不加入微秒級 Tick 訂單簿引擎，全身心專注於 1m, 5m, 15m, 1h, 1d 等 K 線層級數據流之正確性、參數高原尋優與防過擬合。

---

## 🛡️ Phoenix 核心 12 大思想鋼印 (Mental Blueprint Steel Stamp)

### 1. 禁用二次標準化防線 (Zero Double Z-Score Defense)
- **原則**：Layer 1 特徵庫 (如 `AVO.py`, `AO.py`, `Kalman_Filter.py`) 內部已實作動態滾動 Z-Score，輸出具備 Scale-Free 物理特性。
- **紅線**：在 Layer 1.5 Proxy 或 Layer 2 評估階段，**嚴禁套用全域 StandardScaler 或 MinMax 縮放**，防止極端值 (Fat Tails) 資訊失真。

### 2. 參數高原度量與 SAM 擾動 (Plateau Smoothness & SAM Fitness)
- **原則**：搜尋目標不是單點過擬合，而是**寬闊的參數高原 (Parameter Plateau)**。
- **紅線**：Proxy 適應度函數必須結合 **SAM (Sharpness-Aware Minimization)** 鄰域擾動懲罰 (`sam_score = mean_fitness - sharpness_penalty`) 與 **Rank IR 冗餘懲罰**。

### 3. 多時間尺度與組件 100% 純動態對應 (100% Dynamic Component Mapping)
- **原則**：不論新增指標、新增尺度 (`4h`, `12h`, `30m`), 新增世界線、或新增交易標的，**核心代碼嚴禁硬編碼**。
- **紅線**：`DataLayer.batch()` 必須動態綁定 `target_scale`；`SearchSpaceParser` 純動態展平。

### 4. 交易經濟學契約 (Trading Economics Specification)
- **原則**：交易成本與滑點必須納入評估。
- **規範**：YAML 必須動態宣告 `trading_economics` (`margin`, `tick_size`, `point_value`, `slippage_ticks`, `spread_ticks`, `commission_per_order`)。

### 5. MultiCharts 擬真盤測損益關聯鋼印 (MultiCharts PnL Performance Report Linkage)
- **原則**：參數高原尋優不能僅停留於特徵層面的 Rank IC/IR 抽象數據，必須與真實盤測損益系統連動落地。
- **規範**：Phase 4 死亡競技場 (`DeathmatchTournament`) 必須透過 `MultiChartsPnLEngine` 模擬真實下單（包含 CME MBT 合約乘數、最小跳動點、滑價與手續費），並在產出的 YAML 配置中 100% 烙印 31 項 MultiCharts 專業損益報告指標。

### 6. 反簡化鋼印：論文 100% 擬真機轉防線 (Anti-Simplification Paper Defense)
- **原則**：凡使用者提及學術論文或 SOTA 演算法 (如 HEBO, TuRBO, CMA-ES)，AI 必須做到 100% 完整理解，並在程式碼中完全還原論文原著之核心數學機轉。
- **紅線**：**嚴禁使用任何簡化版代理包 (Proxy Wrappers) 或單點貪婪採樣器來敷衍**。必須原汁原味實作原著演算法。

### 7. 反硬編碼鋼印：100% YAML 解耦單一真理源防線 (Anti-Hardcoding Single Source of Truth)
- **原則**：所有領域配置必須 100% 嚴格以 YAML 配置檔為單一真理源 (Single Source of Truth)。
- **紅線**：**嚴禁在任何 Python 代碼中設置硬編碼預設值 (Fallback Defaults) 或二次截斷數值**。缺少 Key 必須拋出明確的 KeyError！

### 8. 反不思考搪塞鋼印：零裁切與 100% 異常堆疊追蹤防線 (Anti-Evasion Full Traceability Defense)
- **原則**：當發生計算失敗時，AI 必須進行深入根因分析，嚴禁使用「截斷數值/靜化隱藏 (Silent Clamping)」。
- **紅線**：**嚴禁人工裁切適應度**（例如 `max(-50.0, fitness)`），失敗必須印出觸發異常的 `flat_params` 與 Traceback。

### 9. 反結構破壞與層疊解展平防線 (Anti-Structural Corruption & Unflattening Defense)
- **原則**：在將參數寫回 YAML 配置或傳遞給特徵工廠時，必須 100% 通過 `SearchSpaceParser.unflatten()` 轉化為標準巢狀字典。
- **紅線**：**嚴禁將展平字典 (`{'MACD.fast_period': 12}`) 直接賦值給 `indicator_default_params`**，杜絕 `KeyError: 'source'`！

### 10. 反盲目宣告與端對端真實運轉驗證防線 (Anti-Complacency E2E Test Defense)
- **原則**：AI 絕不允許在未經過實地完整執行前宣告「已 100% 檢查完畢」。

### 11. 極速煙霧測試與 Token 節流防線 (Fast Smoke Testing & Token Context Efficiency Defense)
- **原則**：在對話交互中，AI 驗證代碼正確性時，**嚴禁發動長達數小時的千次正式生產尋優 (1000 Trials Heavy Run)**。
- **規範**：必須透過極速微型煙霧測試 (`search_budget = 1` 或 `2` 的獨立驗證腳本），在 5 秒內完成「Phase 3 尋優 $\rightarrow$ Phase 4 死亡競技場解展平 $\rightarrow$ 損益評估 $\rightarrow$ YAML 導出」的全鏈條 Fast Pass。

### 12. 期貨交易日曆與休市斷點防線 (Futures Calendar & Session Break Defense)
- **原則**：當處理 100% 落地之期貨數據時，必須嚴格加載 `market_calendar`，濾除每日 17:00-18:00 結算休市、週末休市與國定假日。
- **紅線**：**嚴禁在休市與假日區間進行 Forward Fill 補線**，防止指標產生假性低波動度 (Zero Volatility) 訊號扭曲！
