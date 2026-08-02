Phoenix Core: Layer 4 State Transition Layer - META 工程文件

系統版本: v1.5-Lite (100-Point Standard)
底層架構: Python (Logic) + torch / numpy (Compute)
核心依賴: pydantic, numpy, torch
上游輸入: 1. Layer 3: MarketState (當前市場政權機率)
2. Layer 2: RepresentationOutput (當前市場表徵向量)
下游輸出: TransitionState (下一時刻的狀態轉移預測矩陣與機率)

系統版本: v2.0-Pro (100-Point Standard, Multi-Asset & Identity Freeze)
底層架構: Python (Logic) + torch / numpy (Compute)
核心依賴: pydantic, numpy, torch
上游輸入: 1. Layer 3: MarketState (包含多資產與凍結標記)
2. Layer 2: RepresentationOutput (128維市場潛在空間)
下游輸出: TransitionState (下一時刻的狀態轉移預測矩陣與機率)

📂 實體目錄樹狀圖 (Directory Tree)

此層級打破了靜態馬可夫鏈的限制，引入動態感知的時變矩陣。在 v2.0-Pro 中，新增了多資產路由與休市單位矩陣映射器。

phoenix/
└── core/
    └── layer4_transition/            
        ├── __init__.py
        ├── schemas.py                # [介面契約] 嚴格的資料與配置結構
        ├── pipeline.py               # 轉移管線協調器 (多資產路由)
        ├── models/                   
        │   ├── __init__.py
        │   ├── static_transition.py  # 基線模型：靜態馬可夫轉移矩陣
        │   ├── dynamic_tvtp.py       # [SOTA核心] 時變轉移機率網路 (TVTP)
        │   └── identity_freezer.py   # [新增/防爆] 休市單位矩陣強制映射器
        └── metrics/                  
            ├── __init__.py
            └── stability_tracker.py  # 狀態震盪器監控 (計算 Entropy)


📜 介面契約與超參數配置 (Interface Contracts & Config) - ⚠️ 100分級約束

嚴格規範矩陣的維度與多資產追蹤。在轉移矩陣的計算中，任何維度錯置 (Dimension Mismatch) 都會導致系統在實盤中崩潰。

schemas.py 嚴格定義：

from pydantic import BaseModel, Field
import numpy as np
from typing import Literal

class TransitionConfig(BaseModel):
    """Layer 4 超參數配置契約"""
    n_components: int = Field(4, description="狀態數量，必須與 Layer 3 嚴格一致")
    model_type: Literal["static", "dynamic_tvtp"] = Field("dynamic_tvtp", description="使用的轉移模型類型")
    lookahead_steps: int = Field(1, description="預測未來第 N 步的轉移 (預設為 1)")
    smoothing_factor: float = Field(0.1, description="轉移矩陣的平滑係數，防止極端跳動")

class TransitionState(BaseModel):
    """Layer 4 輸出契約 (傳遞給 Layer 5 & 6)"""
    timestamp: int 
    asset_id: str                   # [新增] 多資產追蹤
    current_state_probs: np.ndarray # Shape: (n_components,) - 來自 Layer 3
    transition_matrix_t: np.ndarray # Shape: (n_components, n_components) - 當下的轉移矩陣
    next_state_probs: np.ndarray    # Shape: (n_components,) - 預測 T+1 的狀態機率
    transition_entropy: float       # 轉移不確定性 (用來衡量市場是否處於混沌期)
    is_frozen: bool                 # 繼承自 Layer 3 的休市/斷線標記

    class Config:
        arbitrary_types_allowed = True


📄 檔案權責與 META 規範 (File Responsibilities & META Specs)

1. pipeline.py (多資產管線協調器)

權責: 接收 Layer 2 與 Layer 3 的輸出，為每檔資產協調模型計算下一步的機率分佈。

META 約束: * 必須實作 class StateTransitionLayer。

核心公式強制約束: next_state_probs 的計算必須嚴格遵循矩陣乘法：$P(S_{t+1}) = P(S_t) \times T_t$。其中 $P(S_t)$ 是來自 L3 的 current_state_probs 行向量 (1xN)，$T_t$ 是本層產出的 transition_matrix_t (NxN)。

防爆路由: 在呼叫神經網路預測轉移矩陣前，必須先檢查 MarketState.is_frozen。如果為 True，立刻攔截並轉交給 identity_freezer.py。

2. models/identity_freezer.py (休市單位矩陣映射器) - ⚠️ 跨市場防呆紅線

權責: 在資產休市或數據斷線時，徹底凍結狀態的演化。

META 約束:

接收到 is_frozen == True 時，嚴禁呼叫 TVTP 網路。

必須強制回傳一個 單位矩陣 (Identity Matrix, $I$) 作為 $T_t$。

數學意義: 根據馬可夫鏈乘法，$P(S_{t+1}) = P(S_t) \times I = P(S_t)$。這確保了在長達 14 小時的台指期休市期間，系統預測的「下一刻狀態」永遠精準等於「當前狀態」，完全杜絕了神經網路在休市期間因為 0 波動特徵而產生的「無中生有 (Hallucination)」狀態轉移。

3. models/dynamic_tvtp.py (動態時變轉移網路) - ⚡ SOTA 核心

權責: 打破 HMM 的「靜態轉移假設」。捕捉市場微觀結構即將變盤的瞬間。

META 約束:

必須是一個輕量級 PyTorch 網路（如 2 層 MLP）。

輸入: RepresentationVector (Shape: [Batch, 128])

輸出: 動態轉移矩陣 $T_t$ (Shape: [Batch, N, N])。

數學紅線 (Softmax 約束): 網路輸出的 $T_t$ 矩陣，其每一列 (Row) 的總和必須嚴格套用 torch.softmax(dim=-1) 以確保等於 1.0。必須確保這是一個合法的機率轉移矩陣。

4. metrics/stability_tracker.py (狀態震盪防爆器)

權責: 解決量化系統常見的 "Ping-Pong Effect" (頻繁來回切換狀態)。

META 約束:

必須實作 calculate_transition_entropy(transition_matrix_t)。

當轉移矩陣的特徵根變得接近均勻分佈時，Entropy 會升高，代表「系統對下一步極度迷惘」。

算出的 Entropy 必須封裝進 TransitionState，供 Layer 6 在決策時作為「降低整體槓桿 (De-leveraging)」的強力參考。

🛡️ 實盤與回測行為一致性保證 (Streaming vs. Batch Consistency)

為了確保決策邊界不被未來函數污染，並極大化多資產回測速度：

動態矩陣的因果限制: dynamic_tvtp.py 輸出 $T_t$ 時，絕對只能依賴 $X_t$ (時間 T 的表徵)。

Batch 模式的向量化條件遮罩 (Vectorized Conditional Masking): * 在處理包含幾十萬筆 K 線的 Batch 回測時，不能寫 Python for 迴圈。

必須使用 PyTorch 的 torch.where(is_frozen_mask.unsqueeze(-1).unsqueeze(-1), identity_matrix, tvtp_output)，利用 GPU 高速一次性將所有休市時間點的轉移矩陣替換為單位矩陣 ($I$)，以榨乾 CUDA 效能。