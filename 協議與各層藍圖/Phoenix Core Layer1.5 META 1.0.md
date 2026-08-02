Phoenix Core: Layer 1.5 Feature Adjudicator - META 工程文件

系統定位: 鑽石架構的分流中樞 (Feature Pruning & Routing)
目標任務: 解決高維特徵共線性 (Multicollinearity) 與樹模型/小神經網路的維度災難。
上游輸入: Layer 1: 320維寬表 (MarketObservation / DataFrame)
下游輸出: 降維且無共線性的精簡寬表，並根據 Suite (版本) 動態分流至 L2 或 L5。

--------------------------------------------------------------------------------

                      ↙           |           |           ↘

             【Lite 版】     【Core 版】  【Pro 版】     【Ultimate 版】

            (樹模型專線)      (LSTM)     (PatchTST)     (Mamba+Attention)



[L1.5 降維]   LightGBM 砍特徵   可選          Bypass         Bypass (直吃全頻譜)

[L2 表徵]     無 (直接預測)    64維 LSTM    256維 Trans    1024維 SSM

[L3 狀態機]   無             專屬 HMM-A   專屬 HMM-B     專屬高維 HMM-C

[L4 轉移]     無             專屬 TVTP-A  專屬 TVTP-B    專屬 TVTP-C

[L5 預測]     直接輸出預期     專屬預測頭A   專屬預測頭B    專屬預測頭C

                      ↘           |           |           ↙

--------------------------------------------------------------------------------


💎 鑽石架構 (Diamond Architecture) 分流協議

Layer 1.5 必須根據 YAML 兵符中的 suite (作戰版本)，執行絕對的物理分流：

【Lite 版】(樹模型專線):

啟動審判: 強制啟動 LightGBM 進行離線特徵重要性 (Feature Importance) 排序，並透過相關性過濾 (Correlation Filter) 剃除高度共線性的指標。

維度壓縮: 320D -> 30~50D 黃金特徵。

路由: Bypass (繞過) L2, L3, L4，直接將降維後的資料送入 Lite 版專屬的 L5 預測頭 (直接輸出預期收益)。

【Core 版】(LSTM):

啟動審判: 可選啟動。將維度壓縮至 64D~128D，減輕 RNN 長期記憶負擔。

路由: 送入 L2 (LSTM 表徵層)。

【Pro / Ultimate 版】(PatchTST / Mamba):

強制 Bypass: 嚴禁啟動審判庭。

路由: 完整保留 320D 全頻譜，直接送入 L2。依賴 Transformer/Mamba 天生的 Attention 與選擇性狀態機制 (Selective State) 自行探索特徵權重。

📂 實體目錄樹狀圖 (Directory Tree)

phoenix/
└── core/
    └── layer1_5_adjudicator/         
        ├── __init__.py               
        ├── schemas.py                # [介面契約] 審判庭配置與特徵註冊表
        ├── pipeline.py               # [分流協調器] 決定 Bypass 或套用 Mask
        ├── evaluators/               
        │   ├── collinearity.py       # [防爆] 移除 Pearson/Spearman 相關係數 > 0.85 的冗餘特徵
        │   └── tree_adjudicator.py   # [審判核心] 基於 LightGBM 的特徵重要性評分器
        └── registry/                 
            └── approved_features.json # [物理鋼印] 離線審判後固化的「出戰名單」


📜 絕對介面契約 (Interface Contracts)

1. schemas.py 嚴格定義

from pydantic import BaseModel, Field
from typing import List, Dict

class AdjudicatorConfig(BaseModel):
    """Layer 1.5 超參數配置"""
    enabled: bool = Field(True, description="是否啟用特徵審判 (Ultimate 版強制 False)")
    max_features: int = Field(50, description="允許通過的最高特徵數量")
    correlation_threshold: float = Field(0.85, description="共線性剃除紅線")

class FeatureRegistry(BaseModel):
    """
    [物理鋼印] 審判結果註冊表
    訓練期產生，實盤期純讀取，絕對禁止實盤動態修改。
    """
    suite_name: str
    approved_features: List[str]      # 最終存活的特徵名稱 (e.g., ["MACD_feat_macd_momentum_stable", ...])
    rejected_features: Dict[str, str] # 剃除名單與原因 (e.g., {"RSI_15m": "High Collinearity with RSI_1h"})


📄 模組權責與 SOTA 規範 (Module Specifications)

1. pipeline.py (分流與過濾協調器)

實盤引擎 (Streaming - $O(1)$ 過濾):

在實盤 step() 中，絕對禁止呼叫任何運算 (如計算相關性或跑樹模型)。

必須直接讀取 approved_features.json，對 Layer 1 傳來的 MarketObservation 執行 $O(1)$ 複雜度的 Column 切片 (Slicing/Masking)。

速度要求: 特徵過濾耗時必須 $< 1ms$。

回測/訓練引擎 (Batch - The Adjudication):

在離線訓練階段，必須依序執行：

呼叫 tree_adjudicator.py 訓練一棵全局 LightGBM 樹，取得全體 320 維的初步重要性排名。

從最高分的特徵開始往下挑選，將每個新特徵送入 collinearity.py 進行檢查。若該特徵與「已入選名單」中的任何特徵相關性 $> 0.85$，直接斬首 (Reject)。

直到選滿 max_features 數量為止，將結果固化寫入 JSON。

2. evaluators/collinearity.py (共線性斬首器) - ⚠️ 防爆紅線

痛點: MACD_5m 與 MACD_15m 高度連動，如果同時餵給樹模型，樹的分裂增益 (Split Gain) 會被稀釋，導致模型抓不到真正的邊界。

SOTA 約束: 必須計算特徵之間的 Spearman Rank Correlation (斯皮爾曼等級相關係數)，因為金融特徵往往是非線性的，傳統的 Pearson 相關係數會失效。

3. evaluators/tree_adjudicator.py (樹模型審判官)

防前視偏差紅線 (No Look-ahead Leakage):

執行特徵重要性評分時，傳入的 Dataset 絕對不能包含 Validation (驗證集) 與 Test (測試集) 的資料。

特徵的篩選必須「只基於訓練集 (Training Set)」的資訊。若用全域數據選特徵，這等同於嚴重的資料外洩 (Data Leakage)，實盤將遭遇滑鐵盧。

🛡️ 對下游 (Layer 2 / Layer 5) 的終極承諾

當 Layer 1.5 執行完畢，下游模型將獲得以下保障：

Lite 版的極致純淨: LightGBM (L5) 將收到一個維度極低 ($<50$)、每一根 Feature 都是獨立且強勢 Alpha、完全沒有資訊重疊的完美表格。訓練速度提升 10 倍，過擬合機率降至最低。

Ultimate 版的無損傳遞: Mamba 模組會收到原始的、未經任何人工武斷破壞的 320 維張量，確保其高維狀態空間能捕捉到最微小的非線性互動。