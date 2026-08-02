Phoenix Core: Layer 3 Regime Layer - META 工程文件

系統版本: v1.5-Lite (100-Point Standard)
底層架構: Python (Logic) + hmmlearn / torch (Compute)
核心依賴: pydantic, numpy, hmmlearn, scipy
上游輸入: Layer 2: RepresentationOutput (低維度密集向量)
下游輸出: MarketState (市場政權機率分佈)

系統版本: v2.0-Pro (100-Point Standard, Multi-Asset & Freeze Aware)
底層架構: Python (Logic) + hmmlearn / torch (Compute)
核心依賴: pydantic, numpy, hmmlearn
上游輸入: Layer 2: RepresentationOutput (128維市場潛在空間, 包含多資產)
下游輸出: MarketState (市場政權機率分佈)


📂 實體目錄樹狀圖 (Directory Tree)

此層級將連續的市場表徵空間，離散化為 N 種具備物理意義的市場狀態。在 v2.0-Pro 中，新增了狀態凍結器以應對多資產的休市情境。

phoenix/
└── core/
    └── layer3_regime/                # 本模組路徑
        ├── __init__.py
        ├── schemas.py                # [介面契約] 嚴格的資料與配置結構
        ├── pipeline.py               # 政權管線協調器 (區分 streaming 與 batch)(管理多資產 HMM 實例)
        ├── models/                   # 核心數學模型
        │   ├── __init__.py
        │   ├── hmm_core.py           # 高斯隱馬爾可夫模型 (Gaussian HMM) 封裝
        │   ├── alignment.py          # [防爆核心] 狀態語意對齊 (Label Switching)
        │   └── state_freezer.py      # [新增/SOTA] 休市狀態凍結器
        └── agents_hook/              # 預留 Agent 介入點
            ├── __init__.py
            └── prior_injector.py     # 外部情緒先驗機率融合器 (Bayesian Update)


📜 介面契約與超參數配置 (Interface Contracts & Config) - ⚠️ 100分級約束

為了防止 AI 生成程式碼時發生幻覺與變數命名不一致，所有實作必須嚴格遵守以下 Pydantic Schema 定義，不得擅自修改欄位名稱或型別。

schemas.py 嚴格定義：

from pydantic import BaseModel, Field
import numpy as np
from typing import List, Literal, Dict

class RegimeConfig(BaseModel):
    """Layer 3 超參數配置契約"""
    n_components: int = Field(4, description="強制定義狀態數量 (e.g., 4 種狀態)")
    covariance_type: Literal["spherical", "diag", "full", "tied"] = "full"
    alignment_metric: Literal["volatility", "momentum"] = "volatility"
    window_size: int = Field(252, description="線上推論時的滾動觀測視窗長度")
    input_dim: int = Field(128, description="嚴格對齊 Layer 2 的 embedding_dim")

class MarketState(BaseModel):
    """Layer 3 輸出契約 (傳遞給 Layer 4 & 6)"""
    timestamp: int 
    asset_id: str                   # [新增] 多資產追蹤
    state_probs: np.ndarray         # Shape: (n_components,), 加總為 1.0
    dominant_state: int             # argmax(state_probs)
    aligned_labels: List[str]       # e.g., ["Low Vol", "Mid Vol", "High Vol Trend", "Crash"]
    is_frozen: bool                 # [新增] 標記該狀態是否因為休市而凍結

    class Config:
        arbitrary_types_allowed = True


📄 檔案權責與 META 規範 (File Responsibilities & META Specs)

1. pipeline.py (多資產管線協調器)

權責: 管理資料流，並為每一檔資產維護獨立的 HMM 狀態機。

META 約束: * 必須實作 class RegimeLayer，內部維護一個 Dict[str, HMMCore] 的路由表，根據傳入的 asset_id 呼叫對應的 HMM 模型。

必須實作 step(repr_output: RepresentationOutput, is_trading: bool) -> MarketState: * 如果 is_trading == False (來自 Layer 1 的 Mask 資訊)，則直接呼叫 state_freezer.py，嚴禁執行 HMM 的 Forward 演算法。

2. models/state_freezer.py (休市狀態凍結器) - ⚠️ 跨市場防呆紅線

權責: 解決「非同步市場時間」導致的狀態矩陣飄移問題。

META 約束:

當接收到休市/斷線訊號時，HMM 的觀測機率 (Emission Probability) 將失效。

必須強制回傳 $T-1$ 時刻的 state_probs，並將 is_frozen = True 寫入 MarketState。

這個動作確保了在美股休市的台灣半夜，美股的 Regime 不會因為假性的 0 波動而被錯誤切換到「State 0: 低波死水」。

3. models/alignment.py (狀態語意對齊) - ⚠️ 系統絕對紅線

權責: 解決無監督學習的「標籤切換問題 (Label Switching)」，為 Layer 6 提供絕對可信的規則邊界。

META 約束:

必須實作 align_states(hmm_model, config: RegimeConfig)。

物理語意對齊: 在 128 維的 Latent Space 中，Covariance 的物理意義對人類是不可讀的。因此，必須在離線訓練階段，計算 HMM 每個 Hidden State 發生時，Layer 1 對應的真實波動率 (Realized Volatility) 均值。

依照真實波動率由小到大，強制重新映射 (Remap) 內部狀態 ID。確保產出的 State 0 永遠是「最低波動」，State N-1 永遠是「極端高波動」。

4. models/hmm_core.py (隱馬爾可夫核心)

權責: 封裝 hmmlearn.GaussianHMM 或自定義的 PyTorch HMM。

META 約束: * 輸入維度必須嚴格斷言為 input_dim == 128。

在 step() 實盤推論時，嚴禁呼叫 .fit()。

🛡️ 實盤與回測行為一致性保證 (Streaming vs. Batch Consistency)

凍結機制的向量化 (Vectorized Freezing):

在 Batch 模式 (大範圍回測) 下，必須利用 Layer 1 傳遞過來的 features_mask 建立一個 boolean mask。在執行 predict_proba 時，遇到 mask == False 的時間步，必須使用 numpy.where 或 ffill 邏輯將前一個時間步的機率分佈填入，確保回測與實盤的凍結行為 100% 一致。