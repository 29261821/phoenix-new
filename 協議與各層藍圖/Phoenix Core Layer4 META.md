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

1. phoenix/core/layer4_transition/schemas.py
系統定位: Layer 4 的 I/O 絕對物理邊界與超參數合約。

🚨 類別命名與依賴絕對約定
允許依賴: 必須且僅限引入 pydantic (BaseModel, Field)、numpy (別名 np)、typing (Literal)。

禁止依賴: 嚴禁引入 torch、pandas 或任何神經網路與計算庫，必須維持純粹的資料合約性質。

類別名稱: 必須精確命名為 TransitionConfig 與 TransitionState。

📐 核心屬性與張量防爆邊界
TransitionConfig 絕對約束 (超參數紅線):

n_components: int，必須與 Layer 3 嚴格一致，決定所有矩陣的維度。

model_type: Literal["static", "dynamic_tvtp"]，預設為 "dynamic_tvtp"。嚴禁在此出現其他字串幻覺。

lookahead_steps: int，預設為 1。

smoothing_factor: float，預設為 0.1。

TransitionState 絕對約束 (輸出合約紅線):

timestamp: int (Unix Timestamp)。

asset_id: str (多資產追蹤)。

current_state_probs: np.ndarray。Shape 防爆約定: 長度必須嚴格等於 (n_components,) 的一維向量。

transition_matrix_t: np.ndarray。Shape 防爆約定: 必須嚴格等於 (n_components, n_components) 的二維方陣，且每一列 (Row) 總和必須為 1.0。

next_state_probs: np.ndarray。Shape 防爆約定: 長度必須嚴格等於 (n_components,)。

transition_entropy: float (系統混沌度指標)。

is_frozen: bool (繼承自 Layer 3 的休市標記)。

Pydantic 防呆: 內部必須宣告 class Config: arbitrary_types_allowed = True 以允許 NumPy 陣列。

📄 模組權責與 SOTA 規範 (Module Specifications)
2. phoenix/core/layer4_transition/models/identity_freezer.py
系統定位: ⚡ 跨市場防呆紅線 (休市單位矩陣映射器)
權責: 在資產休市或數據斷線時，徹底凍結狀態的演化，從數學根源阻絕神經網路的無中生有 (Hallucination)。

🚨 類別命名與依賴絕對約定
允許依賴: 必須引入 torch、numpy (別名 np)、typing (Union)。

類別名稱: 必須精確命名為 IdentityFreezer。

無狀態約束: 此類別內部嚴禁維護任何跨步狀態，所有方法必須實作為 @staticmethod。

⚙️ 核心方法簽名與防爆紅線
方法 1: 基礎單位矩陣生成

簽名: def get_identity_matrix(n_components: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:

邏輯約束: 必須呼叫 torch.eye(n_components, dtype=torch.float32, device=device)。

方法 2: 向量化條件遮罩 (Vectorized Batch Freeze)

簽名: def apply_vectorized_freeze(is_frozen_mask: torch.Tensor, tvtp_output: torch.Tensor) -> torch.Tensor:

張量對齊要求: is_frozen_mask 必須為 Shape (Batch,)；tvtp_output 必須為 Shape (Batch, N, N)。

執行順序絕對紅線 (GPU 榨乾法則):

必須先生成與 Batch 對齊的單位矩陣 identity_batch，Shape 為 (Batch, N, N) (利用 unsqueeze(0).expand(...))。

必須將 is_frozen_mask 擴張為 (Batch, 1, 1) 以進行 Broadcasting。

條件替換紅線: 必須且只能呼叫 torch.where(mask_expanded, identity_batch, tvtp_output) 進行矩陣級別替換。嚴禁寫出任何 Python for 迴圈處理 Batch 數據。

方法 3: 實盤單步凍結 (Streaming Freeze)

簽名: def apply_streaming_freeze(is_frozen: bool, transition_matrix: np.ndarray) -> np.ndarray:

型別鎖死: 必須使用 NumPy (np.eye(..., dtype=np.float64)) 處理，確保實盤管線中 CPU 運算的極低延遲與高精度。如果 is_frozen == True，無條件回傳單位方陣。


3. phoenix/core/layer4_transition/metrics/stability_tracker.py
系統定位: ⚡ 狀態震盪防爆器 (Ping-Pong Effect 監控)
權責: 計算轉移矩陣的 Shannon Entropy。當矩陣特徵根接近均勻分佈時，發出高混沌訊號，供下游 Layer 6 執行投資組合級別的降槓桿。

🚨 類別命名與依賴絕對約定
允許依賴: 必須且僅限引入 numpy (別名 np)、torch、typing (Union)。

無狀態函數約束: 必須實作為純函數 (Pure Function)，嚴禁依賴任何全域變數或類別狀態。

⚙️ 核心方法簽名與防爆紅線
簽名: def calculate_transition_entropy(transition_matrix: Union[np.ndarray, torch.Tensor]) -> float:

對齊要求: 傳入的矩陣必須是合法的轉移矩陣，即 Shape 為 (N, N) 且每一列總和為 1.0。

執行順序與數學絕對紅線:

極限 NaN 防護 (除零防爆): 第一行必須定義微小值 epsilon = 1e-9。在執行 log 運算前，無論是 NumPy 或 PyTorch 路徑，必須強制套用 clip/clamp (範圍 [epsilon, 1.0])，絕對禁止系統因為 log(0) 拋出 NaN 導致梯度損壞。

多型路由 (Polymorphism): 必須透過 isinstance 動態判斷傳入的是 Tensor 還是 ndarray，並分別走向對應的底層 API 進行計算。

維度計算紅線: 必須先對最後一個維度 dim=-1 / axis=-1 計算列級別 (Row-wise) 的熵：-sum(P * log(P))，最後再對所有列取 mean() 平均值，回傳單一浮點數 float。

4. phoenix/core/layer4_transition/models/static_transition.py
系統定位: 🛡️ 基準模型與降級防禦機制 (Baseline & Fallback)
權責: 提供非時變的固定狀態轉移機率矩陣。在 TVTP 網路失效或需要消融實驗 (Ablation Study) 時作為對照組。

🚨 類別命名與依賴絕對約定
允許依賴: 必須且僅限引入 torch 與 torch.nn (別名 nn)。

類別名稱: 必須精確命名為 StaticTransitionModel，並繼承自 nn.Module。

⚙️ 核心方法簽名與防爆紅線
建構子約定 (__init__):

簽名: def __init__(self, n_components: int = 4):

參數優化紅線: 嚴禁將機率直接寫死為常數矩陣。必須透過 nn.Parameter 將矩陣註冊為可學習參數 (Learnable Parameter)，以允許 MLE (最大概似估計) 離線優化。

慣性初始化紅線: 初始矩陣必須是 Logits 而非機率。必須對 Logits 矩陣執行 fill_diagonal_(2.0)，確保網路在未經訓練前具有強烈的「狀態慣性」，防止開局產生無意義的頻繁切換。

前向傳播約定 (forward):

簽名: def forward(self, x: torch.Tensor) -> torch.Tensor:

介面對齊約束: 為了與 TVTP 保持多型對齊，即使不依賴 x 運算，也必須接收 x 作為輸入。x 的 Shape 預期為 [Batch, 128]。

數學機率紅線 (Softmax Constraint): 必須對內部的 transition_logits 執行 torch.softmax(..., dim=-1)，確保產出合法機率矩陣。

向量化廣播 (O(1) Broadcasting): 嚴禁使用 Python for 迴圈。必須捕捉 batch_size = x.size(0)，並透過 unsqueeze(0).expand(batch_size, -1, -1) 將 [N, N] 矩陣高速展開為 [Batch, N, N] 供下游管線使用。

5. phoenix/core/layer4_transition/models/dynamic_tvtp.py
系統定位: 🚀 SOTA 核心大腦 (時變轉移機率網路)
權責: 接收 128 維市場潛在特徵，打破馬可夫鏈的靜態假設，動態生成下一刻的狀態轉移矩陣。

🚨 類別命名與依賴絕對約定
允許依賴: 必須且僅限引入 torch 與 torch.nn (別名 nn)。

類別名稱: 必須精確命名為 DynamicTVTP，並繼承自 nn.Module。

⚙️ 核心方法簽名與防爆紅線
建構子約定 (__init__):

簽名: def __init__(self, input_dim: int = 128, n_components: int = 4, hidden_dim: int = 64):

網路拓撲約束: 必須實作為輕量級 2 層 MLP (nn.Linear -> nn.GELU -> nn.Dropout(p=0.1) -> nn.Linear)，以確保實盤推論耗時小於 1ms。

維度對齊: 輸入維度嚴格鎖死對齊 Layer 2 的 embedding_dim (128)。輸出維度必須映射為 n_components * n_components。

權重防爆初始化 (_initialize_weights):

簽名: def _initialize_weights(self):

Xavier 初始化: 最終層權重必須使用 nn.init.xavier_uniform_(..., gain=0.1) 控制初始變異數。

對角線偏差約束 (Diagonal Bias Red Line): 這是消滅 Ping-Pong 效應的核心。必須構建一個 (N, N) 的零矩陣，將對角線 fill_diagonal_(2.0) 後，透過 view(-1) 攤平，並強制賦值給最終層的 bias.data。

前向傳播約定 (forward):

簽名: def forward(self, x: torch.Tensor) -> torch.Tensor:

因果隔離: 網路運算只能依賴當前輸入 x，嚴禁引入未來函數或跨步記憶 (此為 L2 的責任)。

張量重塑 (Reshape Guard): 網路輸出 Shape 為 [Batch, N*N]，必須嚴格使用 view(batch_size, self.n_components, self.n_components) 進行記憶體連續性重塑。

終極數學紅線 (Softmax Constraint): 必須對重塑後的矩陣執行 torch.softmax(..., dim=-1)，絕對保證輸出的 T_t 矩陣每一列機率和為 1.0。任何遺漏此步驟的行為將導致下游聯合分佈全面崩潰。

6. phoenix/core/layer4_transition/pipeline.py系統定位: 🧠 多資產轉移管線協調器 (State Transition Pipeline Coordinator)權責: 統合 L2 表徵與 L3 狀態，執行休市防爆路由與 TVTP 網路推論，嚴格實踐 $P(S_{t+1}) = P(S_t) \times T_t$ 的數學物理意義。🚨 類別命名與依賴絕對約定允許依賴: 必須且僅限引入 torch、numpy (別名 np)、typing (Dict, List, Optional)。跨層合約依賴: 必須引入 MarketState (L3), RepresentationOutput (L2), TransitionConfig, TransitionState, DynamicTVTP, IdentityFreezer, calculate_transition_entropy。嚴禁在此引入任何 L1 或 L5 的合約，維持層級隔離。類別名稱: 必須精確命名為 StateTransitionLayer。⚙️ 核心方法簽名與防爆紅線建構子約定 (__init__):簽名: def __init__(self, config: TransitionConfig, device: torch.device = torch.device('cpu')):模型掛載紅線: 必須根據 config.model_type 動態實例化模型 (DynamicTVTP 或 StaticTransitionModel)，並將模型推至指定的 device。推論模式鎖死: 實例化模型後，必須立即強制呼叫 self.model.eval()，嚴防實盤時觸發 Dropout 或 BatchNorm 的動態更新，確保推論絕對穩定。單一資產實盤推論 (step):簽名: def step(self, current_regime: MarketState, repr_output: RepresentationOutput) -> TransitionState:執行順序絕對紅線 (Data Plumbing Guard):實體對齊斷言 (ID Match Guard): 第一行必須強制執行 assert current_regime.asset_id == repr_output.asset_id。如果 L2 與 L3 的資料屬不同資產卻被送入，必須讓系統立即崩潰，絕不允許張量錯位。休市攔截路由 (Frozen Intercept): 檢查 current_regime.is_frozen。若為 True: 嚴禁呼叫神經網路。必須呼叫 IdentityFreezer.apply_streaming_freeze，直接產出全零狀態下的單位矩陣 $I$。若為 False: 將 repr_output.embedding 轉換為 Shape (1, 128) 的 PyTorch Tensor 並移至設備。無梯度推論隔離 (No Grad Red Line): 呼叫神經網路前向傳播時，必須包覆在 with torch.no_grad(): 區塊內，防止實盤推論時的 VRAM 記憶體洩漏 (Memory Leak)。馬可夫鏈推進公式 (Matrix Multiplication): 必須精確呼叫 NumPy 矩陣乘法：p_s_t_next = np.matmul(p_s_t, t_matrix_np)。其中 p_s_t 形狀為 (N,)，t_matrix_np 形狀為 (N, N)，輸出為 (N,)。混沌度封裝: 呼叫 calculate_transition_entropy 計算轉移矩陣熵值。合約輸出: 將計算結果嚴格打包為 TransitionState 輸出。多資產協調路由 (step_multi_asset):簽名: def step_multi_asset(self, regimes: Dict[str, MarketState], reprs: Dict[str, RepresentationOutput]) -> Dict[str, TransitionState]:交集比對紅線 (Intersection Guard): 內部嚴禁直接遍歷所有資產，必須確認該 asset_id 同時存在於 regimes 與 reprs 的 Keys 中（因為異步市場中某些資產可能因為斷線而在當前 Step 遺失），才能將兩者提取並呼叫單步 step 方法。實盤與回測行為一致性保證 (Streaming vs Batch Vectorization):針對歷史回測，必須實作對應的 batch(...) 方法。向量化約束: 在 batch 模式下，嚴禁對歷史時間步 (Time steps) 使用 Python for 迴圈呼叫 step()。統一遮罩紅線: 必須將整個時間序列的 Representation 合併為 Tensor (Batch, 128) 送入網路，並將 L3 的 is_frozen 狀態合併為 (Batch,) 的 Boolean Mask。隨後，強制交由 IdentityFreezer.apply_vectorized_freeze 進行 GPU 級別的一次性單位矩陣替換，確保回測與實盤的數學產出達到 $10^{-6}$ 小數點級別的絕對一致，同時極大化吞吐量。

