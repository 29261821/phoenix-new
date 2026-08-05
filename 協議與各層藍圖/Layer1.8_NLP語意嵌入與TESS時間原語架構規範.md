# Phoenix 系統：Layer 1.8 NLP 語意嵌入與 TESS 時間原語架構規範

## 1. 核心使命
Layer 1.8 接收來自 Layer 1.7 的標準非結構化新聞流，將其轉化為可用於時間序列機器學習的「高維度純淨特徵向量」，並完全解耦實盤即時推理。

## 2. 三代 NLP 演算法評比與 SOTA 選型

| 梯次 | 演算法名稱 | 核心機制 | 實盤致命傷 | 結論評級 |
| :--- | :--- | :--- | :--- | :--- |
| **方法 A** | **FinBERT Baseline** | 提取 `[-1, 1]` 情緒與 768 維密集向量。 | 768 維向量帶來巨大雜訊與 Modality Gap，易被 L1.6 淘汰。 | ❌ 弱勢梯隊 (淘汰) |
| **方法 B** | **Graph RAG** | Neo4j 知識圖譜 + GNN 拓撲融合。 | 實盤檢索延遲過高，記憶體爆炸，不適合個人實盤。 | ⚠️ 強勢但工程代價過高 |
| **方法 C** | **TESS 時間原語** (ICML SOTA) | 本地 LLM 壓縮成 **4 大時間原語** + **信心門控**。 | 0 延遲、零過擬合、閉環嵌入 Layer 1.5。 | 🏆 第一梯次 (SOTA 唯一首選) |

## 3. TESS (Temporal Evolution Semantic Space) 實體機制

### 步驟 1：離線 LLM 時間原語萃取 (MCP 適配)
LLM (如 Ollama Llama-3) 讀取新聞後，被強制要求輸出 4 個數學原語與信心分：
1. **分佈偏移 (Distribution Shift)**：`[-1, 0, 1]`（方向）
2. **波動率衝擊 (Volatility Shift)**：`[0, 1, 2, 3]`（預期震幅）
3. **時序形狀 (Shape)**：如 *Rise-then-Fall*（短多長空）
4. **衰減半衰期 (Half-life Lag)**：事件發酵預期時間

### 步驟 2：信心門控 (Confidence Gating)
$$\text{Final Feature} = \text{Primitive Value} \times \text{Sigmoid}(\text{Log-Prob Confidence} - \text{Threshold})$$
若 LLM 信心不足，特徵強制歸零 (0)，徹底消滅 LLM 幻覺干擾。

### 步驟 3：閉環融入 Layer 1.5 參數高原優化所
將新聞半衰期與信心閥值導出為 YAML 超參數（如 `News_Decay: { half_life: 12, threshold: 0.8 }`），由 Layer 1.5 (HEBO/CMA-ES) 進行 3,000 次尋優，自動算出一款資產對新聞的最佳記憶力半衰期！
