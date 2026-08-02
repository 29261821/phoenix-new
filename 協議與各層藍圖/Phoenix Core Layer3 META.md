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

1. 全局設計原則與時間隔離紅線Layer 3 的核心任務是將 Layer 2 產出的連續高維潛在空間表徵 (128-dimensional Latent Space) 離散化為具備明確物理語意且互相排斥的市場狀態。無未來污染紅線 (Causality Constraint)： 在任何時間步 $T$，Layer 3 的任何組件（包括隱馬爾可夫核心、狀態凍結器與先驗融合器）絕對只能依賴 $\le T$ 的歷史資訊。嚴禁在 step() 或 batch() 中以任何間接形式引入 $>T$ 的未來數據。異步時間對齊 (Async Time Synchronization)： 面對跨市場、跨交易所的非同步交易時段（例如週末加密貨幣開盤但台指期/美股休市），本層必須透過剛性遮罩 (Hard Masking) 阻斷馬可夫鏈的步進，嚴禁將假性的「零波動特徵」誤判為低波狀態。不簡化與零幻覺原則： 任何涉及矩陣乘法、貝氏更新與機率歸一化的運算，必須包含顯式的數值下溢 (Underflow) 防護與邊界斷言 (Assertions)，嚴禁省略極端數值裝甲。

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


3. schemas.py 絕對介面契約規範本檔案為 Layer 3 的 I/O 絕對物理邊界。嚴禁在此檔案中引入 torch、pandas、scipy 或任何神經網路與底層運算庫，維持純粹的資料型別合約性質。3.1 允許引入庫清單 (Whitelist)Pythonfrom pydantic import BaseModel, Field, validator
import numpy as np
from typing import List, Literal, Dict, Any

3.2 RegimeConfig 超參數配置契約類別名稱： RegimeConfig (必須繼承 pydantic.BaseModel)核心屬性與形狀約束：n_components: int。強制定義 HMM 的隱藏狀態數量。必須附加約束：大於 1 且小於等於 10 (gt=1, le=10)。預設值為 4。covariance_type: Literal["spherical", "diag", "full", "tied"]。定義高斯發射機率的共變異數矩陣型別。
預設值為 "full"。alignment_metric: Literal["volatility", "momentum"]。標籤對齊基準。預設值為 "volatility"。window_size: int。
線上滾動推論或暖機所需的觀測時序長度。必須大於 0 (gt=0)。預設值為 252。input_dim: int。輸入特徵維度。必須強制預設為 128。
【META 絕對紅線】屬性驗證器 (Validators)：必須實作 @validator('input_dim')。當傳入值不等於 128 時，必須拋出 ValueError，其錯誤訊息必須精確包含 "META 絕對紅線: input_dim 必須嚴格等於 128"。此欄位嚴禁妥協，用以防禦 Layer 2 輸出與 Layer 3/4 矩陣乘法的維度錯置。必須實作 @validator('n_components')。當 v < 2 時，必須拋出 ValueError，說明狀態數不足以執行政權切換。
3.3 MarketState 輸出契約 (下游傳遞載體)類別名稱： MarketState (必須繼承 pydantic.BaseModel)核心屬性與型別約束：timestamp: int。全局基準 Heartbeat (Unix Timestamp 毫秒)。asset_id: str。多資產追蹤的標的代碼 (e.g., 'BTC/USDT', 'TXO/FITX')。state_probs: np.ndarray。
當前時刻各個政權的機率分佈。Shape 約束：必須為一維向量，長度嚴格等於 (n_components,)。dominant_state: int。
主導狀態 ID，數值必須等於 argmax(state_probs)。aligned_labels: List[str]。
經物理語意對齊後的標籤描述清單。is_frozen: bool。防爆標記。當標的處於休市、熔斷或數據斷線時，此值必須為 True。
Pydantic 宣告配置： 必須包含內部類別 class Config: arbitrary_types_allowed = True，以允許 np.ndarray 通過校驗。
【數學紅線】屬性驗證器 (Validators)：必須實作 @validator('state_probs')。必須使用 np.isclose(np.sum(v), 1.0, atol=1e-5) 審查傳入的機率向量。
若總和不為 1.0（允許 $10^{-5}$ 浮點數微差），必須立即拋出 ValueError，其錯誤訊息必須包含 "數學約束錯誤：狀態機率總和必須為 1.0"。此項檢查為硬性紅線，用以防止下游 Layer 4 在計算轉移矩陣或 Layer 6 計算期望值時發生分佈發散或 NaN 崩潰。
4. __init__.py 命名空間隔離合約為了保證模組化黑盒子原則，Layer 3 內部的子模組與具體模型實現細節（如 StateAligner、HMMCore 等）嚴禁向外層（如系統主循環或其他無關層級）直接暴露。
上游與下游層級僅能引入 RegimeConfig、MarketState 與 RegimeLayer。
4.1 phoenix/core/layer3_regime/__init__.py 規範頂層初始化檔案必須且僅能包含以下匯入與 __all__ 宣告，嚴禁擴充：Pythonfrom .schemas import RegimeConfig, MarketState
from .pipeline import RegimeLayer

__all__ = [
    "RegimeConfig",
    "MarketState",
    "RegimeLayer"
]
4.2 phoenix/core/layer3_regime/models/__init__.py 規範模型層初始化檔案負責封裝內部具體數學運算，必須且僅能包含以下內容：Pythonfrom .hmm_core import HMMCore
from .alignment import StateAligner
from .state_freezer import StateFreezer

__all__ = [
    "HMMCore",
    "StateAligner",
    "StateFreezer"
]
4.3 phoenix/core/layer3_regime/agents_hook/__init__.py 規範工具介入點初始化檔案負責封裝外部 Agent 融合邏輯，必須且僅能包含以下內容：Pythonfrom .prior_injector import PriorInjector

__all__ = [
    "PriorInjector"
]

5. pipeline.py 管線協調器核心規範pipeline.py 是 Layer 3 的大腦中樞，負責調配多資產的 HMM 實例與狀態凍結器。它必須完美隔離各市場的非同步時間軸，並提供實盤單步推論與離線歷史回測的雙引擎。5.1 類別命名與依賴絕對約定類別名稱： 必須精確命名為 RegimeLayer。允許引入庫清單 (Whitelist)：Pythonimport numpy as np
from typing import Dict, List, Any
import logging
from .schemas import RegimeConfig, MarketState
from .models.hmm_core import HMMCore
from .models.state_freezer import StateFreezer
【嚴格禁用紅線】 嚴禁引入 pandas 或任何未經架構委員會核可的外部第三方時序管理庫。所有多資產隔離路由必須基於 Python 原生 Dict 與 numpy 底層矩陣進行高速映射。
5.2 核心方法簽名 (Signatures)本類別必須且只能實作以下三個核心方法，嚴禁修改參數型別或變更回傳合約：建構子 (Constructor)：Pythondef __init__(self, config: RegimeConfig, assets: List[str]) -> None
實盤單步推論引擎 (Streaming Inference Engine)：Pythondef step(self, repr_output: Any, is_trading: bool) -> MarketState
離線批次回測引擎 (Batch Inference Engine)：Pythondef batch(self, repr_outputs: List[Any], masks: np.ndarray) -> List[MarketState]
5.3 執行順序絕對紅線
5.3.1 __init__ 初始化執行順序：
第一步： 強制將傳入的 RegimeConfig 與 assets 列表綁定至實例變數 self.config 與 self.assets。
第二步： 必須為 assets 列表中的每一檔資產，獨立實例化 HMMCore(config) 與 StateFreezer(config)，並分別綁定至 self.models 與 self.freezers 字典。
嚴禁多檔資產共用同一個模型執行個體或記憶體緩衝區。
5.3.2 step() 實盤推論執行順序：第一步 [資產合規性檢查]： 必須驗證 repr_output.asset_id 是否存在於 self.models。
若不存在，必須立即拋出 ValueError 攔截，嚴禁帶著未知標的進入後續矩陣運算。
第二步 [休市防呆攔截紅線]： 必須優先檢查 is_trading 布林值。如果 is_trading == False，必須立刻執行短路求值 (Short-circuit Evaluation)，直接呼叫 StateFreezer.get_frozen_state() 並回傳其 MarketState 結果。
此時嚴禁呼叫或觸發 HMM 的任何 Forward 演算法或貝氏更新。
第三步 [正常交易推論]： 當 is_trading == True 時，呼叫對應資產的 HMMCore.step() 計算當前時間步的狀態機率向量 state_probs。
第四步 [語意封裝]： 計算 dominant_state = int(np.argmax(state_probs))，並自 HMM 實例獲取物理語意標籤 aligned_labels，封裝為 MarketState 實例。
第五步 [記憶體狀態更新紅線]： 必須在回傳 MarketState 之前，將此有效狀態傳遞給該資產的 StateFreezer 進行內部緩存更新 (update_last_valid_state)，確保未來突發斷線或休市時，系統具備精確的 $T-1$ 基準狀態。
5.3.3 batch() 離線回測執行順序：
第一步： 檢查 repr_outputs 長度，若為空列表，必須立刻回傳空列表。
第二步： 提取 repr_outputs[0].asset_id 並路由至對應的 HMMCore 實例。
第三步： 向量化提取所有時間步的特徵向量，建構 Shape (T, 128) 的巨大特徵矩陣。
第四步： 呼叫 hmm.batch_predict_proba(embeddings)，一次性產出全時段原始狀態機率矩陣 Shape (T, n_components)。
第五步 [因果律向後填充紅線]： 依據時間序列正向順序（從 t=0 到 T-1）遍歷 masks 遮罩。若 masks[t] == False，必須強制覆蓋該時步的機率為最新一筆有效交易日之狀態機率分佈（即執行時間序列 Forward Fill），且必須將該步的 is_frozen 標記鎖死為 True。嚴禁反向遍歷或使用未來數據進行填補，徹底根除前視偏差。
第六步： 將處理完畢的矩陣逐列封裝為 List[MarketState] 並回傳。
5.4 張量/矩陣防爆邊界 (Shape & NaN Guards)輸入維度斷言： 在進入 HMM 計算前，自 repr_output.embedding 提取之向量，其 Shape 必須嚴格斷言為 (128,)，任何因上游維度錯置導致的形狀異常，必須在 pipeline 層級直接拋錯崩潰，絕對不允許進入底層底層數學庫。
輸出分佈保障： step() 方法產出的 MarketState.state_probs 矩陣，其 Shape 必須精確等於 (n_components,)。且在寫入合約前，其元素總和必須滿足 schemas.py 定義的歸一化校驗。極端起點防禦 (Cold Start Guard)： 在 batch() 模式下，若歷史序列的第 0 筆資料即遭遇 masks[0] == False（即開局即休市），系統必須強制初始化 last_valid_prob 為一個一維熱點向量（Index 0 為 1.0，其餘為 0.0），以此作為基準狀態向後遞延，絕對不允許產生包含 NaN 或全零的空矩陣。
5.5 實盤與回測行為一致性保證 (Streaming vs Batch)無狀態協調原則 (Stateless Pipeline)： RegimeLayer 類別本身不維護任何時序滾動的特徵歷史緩存（此職責已於 Layer 1 與 Layer 2 之 FIFO 緩存中鎖死）。RegimeLayer 僅維護多資產的「模型字典」與「最後有效狀態指針 (Last Valid State Pointer)」。
數學決定性要求 (100% Deterministic Reproducibility)： 對於相同的上游 128 維潛在空間表徵與相同的交易遮罩狀態，step() 引擎在實盤推論出的 MarketState 機率分佈，與 batch() 引擎在對應歷史時間點切片產出的分佈，其數學微差必須小於 $10^{-7}$。
休市行為對齊線： 實盤中透過 is_trading 攔截以阻斷馬可夫鏈演化的行為，與回測中透過 masks 矩陣進行正向條件填充 (Vectorized Forward Fill) 的行為，其數學本質完全對齊。
這確保了系統在長達 14 小時的台指期休市期間，不論在模擬回測還是實盤環境中，皆能展現 100% 一致的物理凍結邊界。


6. hmm_core.py 高斯隱馬爾可夫核心規範
hmm_core.py 負責執行底層的高斯發射機率計算與對數空間 (Log-space) 的馬可夫鏈前向遞迴。它必須具備極高穩健度，全盤阻斷數值下溢 (Numerical Underflow) 與奇異矩陣 (Singular Matrix) 拋錯。

6.1 類別命名與依賴絕對約定
類別名稱： 必須精確命名為 HMMCore。

允許引入庫清單 (Whitelist)：

Python
import numpy as np
import logging
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import hmmlearn.hmm as hmm
from ..schemas import RegimeConfig
【嚴格禁用紅線】 嚴禁呼叫 multivariate_normal.pdf（機率密度函數原值），必須全網鎖死呼叫對數版本 .logpdf。嚴禁在線上 step() 推論中引入任何形式的 .fit() 或 .partial_fit() 邏輯。

6.2 核心方法簽名 (Signatures)
本類別必須且只能實作以下三個核心方法：

建構子 (Constructor)：

Python
def __init__(self, config: RegimeConfig) -> None
權重與對數轉移矩陣注入器 (Weight Injector)：

Python
def set_fitted_model(self, model: hmm.GaussianHMM) -> None
對數空間前向步進推論器 (Log-space Forward Step)：

Python
def step_forward_log(self, x_t: np.ndarray, prev_log_probs: np.ndarray) -> np.ndarray
6.3 執行順序絕對紅線
6.3.1 step_forward_log() 線上推論執行順序：
第一步 [初始化與狀態斷言]： 必須檢查 self.is_fitted 是否為 True。若為 False，必須立即拋出 RuntimeError 攔截。

第二步 [抗毒化輸入審查]： 強制檢驗輸入向量 x_t 的 Shape 是否精確等於 (128,)，且必須使用 np.isnan(x_t).any() 進行空值掃描。若觸發異常，必須立即拋出 ValueError 阻斷。

第三步 [多型別發射對數幾率計算]： 進入狀態循環（從 0 到 n_components-1），依據配置的 covariance_type 提取對應的共變異數矩陣，加上安全脊項後，呼叫 multivariate_normal.logpdf 計算發射對數幾率。

第四步 [數值裝甲防護]： 計算出之發射對數幾率必須立刻執行 np.clip(log_pdf, -700.0, 700.0) 硬性裁切。嚴禁省略此步驟，以防止極端離群值產出 -inf 導致後續貝氏更新產生數學發散。

第五步 [轉移機率前向更新]： 將前一時刻的對數狀態機率 prev_log_probs 透過新增維度 [:, np.newaxis] 與預計算的對數轉移矩陣 self.log_transmat 進行廣播相加，並沿著 axis=0 (前一狀態) 執行 logsumexp，產出當前時刻的預測先驗對數幾率。

第六步 [貝氏後驗更新與歸一化]： 將發射對數幾率與預測先驗對數幾率直接相加，獲得未歸一化的對數後驗機率。隨後減去其全域的 logsumexp 邊際對數似然值，完成 Log 空間內的歸一化減法，並回傳。

6.4 張量/矩陣防爆邊界 (Shape & NaN Guards)
轉移矩陣對數防護： 在 set_fitted_model 中，計算對數轉移矩陣時，必須使用 np.log(np.clip(self.model.transmat_, 1e-15, 1.0))。強制設定下限 1e-15，從根源消滅 log(0) = -inf 所引發的數值崩潰。

共變異數矩陣安全脊 (Epsilon Ridge) 約束： 在計算發射幾率時，針對不同的 covariance_type，必須實作以下剛性防爆機制：

tied: cov = self.model.covars_ + np.eye(128) * 1e-6

full: cov = self.model.covars_[i] + np.eye(128) * 1e-6

diag: cov = self.model.covars_[i] + 1e-6

spherical: cov = self.model.covars_[i] + 1e-6

任何未定義的型別，必須在 else 分支直接拋出 NotImplementedError。

7. alignment.py 狀態語意對齊器規範
alignment.py 是解決無監督學習標籤切換問題 (Label Switching) 的剛性鐵律。它必須確保 HMM 內部狀態 ID 的排列順序，與現實世界中的物理度量（如真實波動率）具備絕對單調遞增關係。

7.1 類別命名與依賴絕對約定
類別名稱： 必須精確命名為 StateAligner。

允許引入庫清單 (Whitelist)：

Python
import numpy as np
import logging
import hmmlearn.hmm as hmm
from typing import List
from ..schemas import RegimeConfig
7.2 核心方法簽名 (Signatures)
本類別包含兩個靜態工廠方法，參數型別與回傳合約鎖死如下：

底層參數重映射置換器 (底層參數矩陣重新映射)：

Python
@staticmethod
def align_states(hmm_model: hmm.GaussianHMM, config: RegimeConfig, metric_values: np.ndarray) -> np.ndarray
語意標籤生成器 (Label Generator)：

Python
@staticmethod
def generate_labels(config: RegimeConfig) -> List[str]
7.3 執行順序與置換紅線
7.3.1 align_states() 矩陣置換絕對執行順序：
第一步 [模型擬合狀態審查]： 必須檢查 hmm_model 是否包含 means_ 與 transmat_ 屬性。若不包含，必須拋出 RuntimeError 攔截。

第二步 [排序指針計算]： 呼交 sort_idx = np.argsort(metric_values)。此指針定義了新舊狀態 ID 的映射對齊軌跡（由小到大）。

第三步 [均值矩陣置換]： 強制更新均值：hmm_model.means_ = hmm_model.means_[sort_idx]。

第四步 [共變異數型別判定與置換紅線]： * 如果 config.covariance_type == "tied"，必須直接跳過 (pass) 共變異數矩陣的切片操作，絕對禁止對其進行排序切片，因為 Tied 矩陣為全狀態共用。

如果為 "full", "diag", "spherical"，則執行 hmm_model.covars_ = hmm_model.covars_[sort_idx]。

第五步 [轉移矩陣雙軸同步置換紅線]： 置換轉移矩陣時，必須同時且分步對其 Row (來源狀態軸) 與 Col (目標狀態軸) 進行置換。標準表達式必須鎖死為：hmm_model.transmat_ = hmm_model.transmat_[sort_idx][:, sort_idx]。嚴禁漏掉任何一個維度軸。

第六步 [初始機率置換]： 檢查若存在 startprob_ 且不為空，必須執行 hmm_model.startprob_ = hmm_model.startprob_[sort_idx]。


8. state_freezer.py 休市狀態凍結器規範本模組為跨市場非同步時間軸的物理防護層。當 Layer 1 傳遞休市或斷線遮罩時，此模組必須絕對阻斷馬可夫鏈的演化，並提供安全的 $T-1$ 狀態，防止假性零波動引發狀態矩陣錯亂。8.1 類別命名與依賴絕對約定類別名稱： 必須精確命名為 StateFreezer。允許引入庫清單 (Whitelist)：Pythonimport numpy as np
import logging
from typing import List
from ..schemas import MarketState
8.2 核心方法簽名 (Signatures)本類別必須實作以下靜態核心方法：靜態凍結器 (Static Freezer)：Python@staticmethod
def freeze(timestamp: int, asset_id: str, prev_state_probs: np.ndarray, aligned_labels: List[str]) -> MarketState
8.3 執行順序與記憶體隔離紅線8.3.1 freeze() 方法絕對執行順序：第一步 [縱深防禦 - 狀態存在斷言]： 必須檢查 prev_state_probs 是否為空 (None 或長度為 0)。若為空，必須拋出 ValueError，嚴禁從虛無中製造狀態。第二步 [縱深防禦 - NaN 毒化斷言]： 必須檢查 np.isnan(prev_state_probs).any()。若前一狀態已被污染，必須拋出 ValueError，拒絕複製毒化矩陣。第三步 [縱深防禦 - 數學歸一斷言]： 必須檢查前置狀態的總和是否為 1.0 (np.isclose(np.sum, 1.0, atol=1e-5))。若發散，必須立即拋錯。第四步 [記憶體安全複製紅線]： 嚴禁直接使用 = 賦值。必須使用 safe_probs = np.copy(prev_state_probs) 進行深層拷貝。此物理意義在於徹底阻斷 Python 的傳址污染 (Pass by Reference)，確保下游 Layer 4 在操作向量時不會意外改寫 Layer 3 的歷史緩存。第五步 [語意封裝]： 重新計算 dominant_idx = int(np.argmax(safe_probs))。第六步 [強制凍結標記]： 回傳 MarketState 實例，並且必須強制將 is_frozen 參數鎖死為 True。9. prior_injector.py 外部先驗機率融合器規範本模組為 Layer 3 與外界 (如 NLP 新聞情緒分析 Agent) 溝通的唯一閘道。外部系統被視為「不可信且極具破壞性」，因此必須實作最嚴格的抗毒化 (Anti-Poisoning) 審查與數值下溢保護。9.1 類別命名與依賴絕對約定類別名稱： 必須精確命名為 PriorInjector。允許引入庫清單 (Whitelist)：Pythonimport numpy as np
import logging
from scipy.special import logsumexp
【嚴格禁用紅線】 嚴禁在此模組實作任何非對數空間 (Non-Log Space) 的機率相乘。9.2 核心方法簽名 (Signatures)本類別必須實作以下靜態核心方法：外部先驗注入器 (Agent Belief Injector)：Python@staticmethod
def inject_agent_belief(current_log_probs: np.ndarray, agent_log_probs: np.ndarray, confidence: float = 0.5) -> np.ndarray
9.3 執行順序與貝氏更新紅線9.3.1 inject_agent_belief() 方法絕對執行順序：第一步 [維度對齊斷言]： 必須比對 current_log_probs.shape 與 agent_log_probs.shape。維度若不匹配，拋出 ValueError。第二步 [外源 NaN/Inf 毒化攔截紅線]： 必須強制掃描外部輸入 agent_log_probs 中是否包含 NaN 或 Inf (np.isnan / np.isinf)。若偵測到毒化數據，必須僅記錄 Logger Warning 並安全地退回原模型的預測 return current_log_probs，嚴禁讓系統因此崩潰停機。第三步 [信心水準裁切]： 檢查 confidence 是否介於 [0.0, 1.0]，若超出範圍，必須強制 np.clip(confidence, 0.0, 1.0)。第四步 [短路求值優化]：若 confidence == 0.0：直接回傳 current_log_probs。若 confidence == 1.0：直接回傳歸一化後的 Agent 預測 (agent_log_probs - logsumexp(agent_log_probs))。第五步 [對數空間貝氏加權紅線]：必須計算權重：log_w_current = np.log(1.0 - confidence + 1e-15) 與 log_w_agent = np.log(confidence + 1e-15)。加上的 1e-15 為不可省略之數學裝甲，用以防禦 log(0)。將原始預測與 Agent 預測分別加上權重後，堆疊為矩陣 stacked。使用 fused_log_probs = logsumexp(stacked, axis=0) 執行凸組合。第六步 [後驗歸一化]： 必須再次執行對數空間歸一化：fused_log_probs = fused_log_probs - logsumexp(fused_log_probs)，確保機率和嚴格等於 1.0 後回傳。10. 全模組輸出與部署邊界保證當本 Layer 3 實作完畢並佈署後，我們對下游 Layer 4 (狀態轉移) 與 Layer 6 (決策優化) 提出以下剛性保證：維度與長度絕對防爆： 進入下游的 state_probs，其維度必定為 (n_components,) 的合法 NumPy 一維張量，且總和永遠為 1.0 (容許 $10^{-5}$ 誤差)。零幻覺與零 NaN 保證： 在遭遇斷線、極端報價、週末休市等情境，輸出之狀態機率分佈必定經過了 Freezer 的深度拷貝保護或是 HMM 內部的數值裁切，絕對不存在 NaN、Inf 等破壞性數值。物理語意單調性保證： State 0 永遠對應實體波動率最低的死水期，State N-1 永遠對應波動極端擴張期。系統已永久免疫 Label Switching 的隨機跳動。