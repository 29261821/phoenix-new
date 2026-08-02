Phoenix Core: Layer 5 Prediction Layer - META 工程文件

系統版本: v1.5-Lite (100-Point Standard)
底層架構: Python (Logic) + torch / xgboost (Compute)
核心依賴: pydantic, numpy, torch, xgboost
上游輸入: 1. Layer 4: TransitionState (下一時刻的狀態機率 $P(S_{t+1})$)
2. Layer 2: RepresentationOutput (當前市場表徵向量 $X_t$)
下游輸出: ForecastState (預期收益率分佈、預測波動率、VaR)

系統版本: v2.0-Pro (100-Point Standard, Multi-Asset Joint Distribution)
底層架構: Python (Logic) + torch / xgboost (Compute)
核心依賴: pydantic, numpy, torch, scipy
上游輸入: 1. Layer 4: TransitionState (下一時刻的狀態機率 $P(S_{t+1})$)
2. Layer 2: RepresentationOutput (128維市場潛在空間)
下游輸出: MarketForecast (全市場預期收益向量、共變異數矩陣、VaR)

📂 實體目錄樹狀圖 (Directory Tree)

此層級從「單一資產預測」躍升為「全市場聯合分佈預測」。必須同時輸出個體的邊際分佈與資產間的共變異數矩陣。

phoenix/
└── core/
    └── layer5_prediction/            
        ├── __init__.py
        ├── schemas.py                # [介面契約] 多資產聯合分佈結構
        ├── pipeline.py               # 全市場預測協調器 (聚合多資產特徵)
        ├── models/                   
        │   ├── __init__.py
        │   ├── mdn_marginal.py       # [邊際分佈] 混合密度網路 (預測各資產 mu, sigma)
        │   ├── covariance_net.py     # [SOTA核心] 動態條件相關性網路 (預測 Covariance)
        │   └── frozen_masker.py      # [防爆核心] 休市預測歸零器
        └── losses/                   
            ├── __init__.py
            └── joint_distribution.py # 多元高斯負對數概似損失 (Multivariate NLL)


📜 介面契約與超參數配置 (Interface Contracts & Config) - ⚠️ 100分級約束

Layer 5 輸出的不再是單一數值，而是針對全市場的數學向量與矩陣。維度必須與註冊的資產數量 ($M$) 嚴格對齊。

schemas.py 嚴格定義：

from pydantic import BaseModel, Field
import numpy as np
from typing import List

class PredictionConfig(BaseModel):
    """Layer 5 超參數配置契約"""
    forecast_horizon: int = Field(1, description="預測未來的 K 棒數量 (e.g., 1 代表 T+1)")
    target_assets: List[str] = Field(..., description="必須與 Layer 1 註冊的資產池嚴格一致，決定矩陣維度")
    cov_method: str = Field("dynamic_dcc", description="共變異數估計法 (dynamic_dcc 或 rolling_empirical)")

class MarketForecast(BaseModel):
    """Layer 5 輸出契約 (傳遞給 Layer 6 進行多資產優化)"""
    timestamp: int 
    assets: List[str]                 # 長度為 M 的資產列表
    expected_returns: np.ndarray      # Shape: (M,) - 各資產預期收益率均值
    expected_vols: np.ndarray         # Shape: (M,) - 各資產預期波動率
    covariance_matrix: np.ndarray     # Shape: (M, M) - SOTA: 動態跨資產共變異數矩陣
    value_at_risk_95: np.ndarray      # Shape: (M,) - 95% VaR (下行尾部風險)
    is_frozen_mask: np.ndarray        # Shape: (M,) - Boolean: True 表示該資產當前休市

    class Config:
        arbitrary_types_allowed = True


📄 檔案權責與 META 規範 (File Responsibilities & META Specs)

1. 介面契約與超參數配置 (schemas.py)系統定位: Layer 5 的 I/O 絕對物理邊界。任何不符合此契約的資料輸入，必須在進入模型前拋出 ValueError。🚨 類別與依賴絕對約定允許依賴: pydantic (BaseModel, Field), numpy (np.ndarray), typing (List)。禁止依賴: 嚴禁引入任何神經網路庫 (torch)，維持純粹的資料契約性質。📐 核心契約紅線PredictionConfig:target_assets: 必須為 List[str]，這決定了後續所有輸出矩陣的 $M$ 維度。cov_method: 必須為 "dynamic_dcc" 或 "rolling_empirical"。MarketForecast (輸出契約):expected_returns, expected_vols, value_at_risk_95: 必須為 np.ndarray，Shape 嚴格固定為 (M,)。covariance_matrix: 必須為 np.ndarray，Shape 嚴格固定為 (M, M)。is_frozen_mask: 必須為 np.ndarray (Boolean)，長度 M。配置約定: 必須宣告 class Config: arbitrary_types_allowed = True 以允許 NumPy 型別。2. 全市場預測協調器 (pipeline.py)系統定位: 數據流聚合中樞，負責執行特徵拼接與模型調度。🚨 類別命名與依賴約定類別名稱: PredictionLayer。允許依賴: torch, numpy (np), typing (Dict, Any)。內部實例化約定: __init__ 必須依序綁定 MDNMarginal, CovarianceNet, FrozenMasker。嚴禁在此處新增未定義的狀態物件。⚙️ 核心方法約定: step(...)輸入參數: transitions: Dict[str, TransitionState], reprs: Dict[str, RepresentationOutput]。執行順序絕對紅線 (不可顛倒):時間戳對齊 (Timestamp Alignment): 檢查所有資產的時間戳是否與首項一致，若不一致直接 RuntimeError 拋錯。特徵空間拼接 (Concat): 循環資產池，執行 torch.cat([reprs[asset].embedding, transitions[asset].next_state_probs])。嚴禁改變拼接順序 (Layer 2 前，Layer 4 後)。邊際分佈推論: 呼叫 self.mdn.predict() 獲取 $\mu$ 與 $\sigma$。共變異數預測: 呼叫 self.cov_net.predict() 獲取 $\Sigma$。休市歸零 (Final Guard): 必須最後呼叫 self.frozen_masker.apply()。嚴禁在歸零前將預測結果暴露給下游。🛡️ 實盤與回測行為一致性無狀態推論: step 函數必須為純函數 (Pure Function)，嚴禁在內部維護除了模型參數以外的緩存。Batch 模式約定: 在處理回測時，pipeline 必須接收 (Batch, M, D) 的張量，並確保 frozen_masker 處理的是對應批次的遮罩，嚴禁全局覆蓋。

3. 混合密度邊際網路 (models/mdn_marginal.py)系統定位: 產出連續概率密度分佈，處理不同市場政權下的收益預期。🚨 類別與依賴絕對約定類別名稱: MDNMarginal (繼承 torch.nn.Module)。允許依賴: torch, torch.nn, typing (Tuple)。⚙️ 核心數學與運算紅線結構約束: 必須包含 pi_head (混合權重), mu_head (均值), sigma_head (波動率) 三個線性輸出頭。激活函數數學紅線:pi: 必須執行 torch.softmax(..., dim=-1)，嚴禁遺漏。sigma: 絕對數學紅線，必須執行 torch.nn.functional.elu(x) + 1.0 + 1e-6。嚴禁使用 exp 函數，防止梯度爆炸。MoE 融合紅線 (predict 方法):必須執行 expected_returns = torch.sum(pi * mu, dim=-1)。總方差計算紅線 (Law of Total Variance): expected_vols 必須透過公式 torch.sqrt(torch.sum(pi * (sigma2 + mu2), dim=-1) - (expected_returns2)) 計算。嚴禁使用簡化的直接加權平均。🛡️ 張量防爆邊界輸入: Shape 必須為 (M, 128 + N) 或 (Batch, M, 128 + N)。輸出: sigma 必須透過 torch.clamp(..., min=1e-6) 進行數值下限保護。4. 動態共變異數網路 (models/covariance_net.py)系統定位: 捕捉資產間動態相關性的 SOTA 核心，負責輸出具備物理合法性的共變異數矩陣。🚨 類別與依賴絕對約定類別名稱: CovarianceNet (繼承 torch.nn.Module)。允許依賴: torch, torch.nn。⚙️ 核心演算法與數學紅線Cholesky 結構紅線: 網路預測的向量元素必須通過 torch.tril_indices 重構為下三角矩陣 $L$。半正定性 (PSD) 防爆紅線:對角線必須執行 torch.nn.functional.softplus(diag_elements) + 1e-4。共變異數矩陣構造公式必須嚴格為 Sigma = torch.bmm(L, L.transpose(1, 2))。嶺懲罰紅線 (Ridge Adjustment): 矩陣輸出前，必須強制加上 identity * 1e-6，確保在極端市況下矩陣永遠可逆，嚴禁輸出奇異矩陣 (Singular Matrix)。⚙️ 核心方法約定: predict(...)簽名: predict(self, x: torch.Tensor, expected_vols: torch.Tensor) -> torch.Tensor。動態 DCC 邏輯紅線:先計算標準相關性矩陣 $R = D^{-1} \Sigma D^{-1}$ (其中 $D$ 為 $L$ 的對角化波動項)。必須使用 expected_vols 進行重縮放 ($D_{pred} R D_{pred}$)，嚴禁直接使用網路輸出的原始 $\Sigma$。此過程必須對應 batch 模式使用 torch.diag_embed 與 torch.bmm 進行矩陣廣播，嚴禁使用 Python 迴圈。5. 休市預測歸零器 (models/frozen_masker.py)系統定位: 隔離休市雜訊的最後物理防線。🚨 類別與依賴絕對約定類別名稱: FrozenMasker。允許依賴: torch, numpy。⚙️ 執行紅線歸零法則: 必須使用 torch.where(mask_tensor, ...) 進行向量化處理。expected_returns 歸零值必須為 0.0。expected_vols 歸零值必須為 1e-6。協方差矩陣隔離紅線:必須建立 cov_mask = mask_row | mask_col，對 (M, M) 矩陣執行全域篩選。休市資產對應的 Row 和 Column 必須強制設為 0.0 (對角線設為 1e-6)。

6. 聯合分佈損失函數 (losses/joint_distribution.py)系統定位: 模型訓練的動力核心，負責監督全市場聯合預測的準確度與穩健性。🚨 類別與依賴絕對約定類別名稱: JointDistributionLoss (繼承 torch.nn.Module)。允許依賴: torch, torch.nn, math。⚙️ 核心數學與防爆紅線動態遮罩隔離 (Dynamic Masking):每一筆 Batch 中的每一個時間步，必須獨立檢查 is_frozen_mask。嚴禁對全局矩陣直接執行 Loss 計算；必須動態提取 active_indices 並執行子矩陣切割。紅線: 若全市場資產在該時間點皆休市，必須輸出 0.0 損失，防止梯度發散。數值穩定性約定 (Cholesky NLL):計算 NLL 時，嚴禁使用矩陣求逆運算 (torch.inverse)。必須使用 torch.linalg.cholesky 分解後，配合 torch.linalg.solve_triangular 進行馬氏距離計算，確保在 $M$ 維度較大時保持 CUDA 運算穩定。退避防禦紅線 (Fallback Mechanism):必須在 try...except 區塊中封裝 Cholesky 運算。若遭遇奇異矩陣拋錯，必須觸發對角線退避邏輯 (Fallback to Diagonal Variance)，嚴禁系統在訓練階段因為個別數據異常而崩潰。7. Layer 5 復現檢查總表 (Checklist)若需進行 100% 準確的工程復現，開發者或 AI 必須逐項核對以下清單：檢查項目規範細節驗證標準張量維度是否對齊 $M$ 維度 (Asset Pool)$M$ 必須與 config.target_assets 一致NaN 防護歸零器是否已在所有輸出路徑生效檢查 is_frozen 覆蓋後的數據分布矩陣 PSDcovariance_net 輸出是否通過 $LL^T$ 重構檢查是否包含 Ridge Adjustment ($1e-6$)數學正確性mdn_marginal 方差公式是否含 Law of Total Var檢查是否減去了均值的平方項異步對齊pipeline.step 是否強斷言時間戳一致性檢查 ValueError 拋錯邏輯回測一致性Batch 模式是否全數向量化處理檢查是否存在 for 迴圈遍歷序列8. 系統憲法結語：工程紅線聲明無未來函數污染 (Zero Look-ahead): Layer 5 的所有運算，僅能依賴 t 時刻的特徵與 t+1 時刻的轉移機率預測，嚴禁讀取 t+1 的實際價格。休市絕對凍結: 當 is_frozen 為 True，該資產的收益預期必須為 0，且該資產在共變異數矩陣中的影響力必須降至 EPS ($1e-6$)，確保不干擾 Layer 6 優化器的運算空間。無條件崩潰保護: 任何涉及矩陣求逆或 Cholesky 分解的運算，必須伴隨 Ridge 加權與 Fallback 機制，確保系統在 100 毫秒內給出穩定的數值結果，而非拋出無解 Error。📢 META 規範完成聲明本文件為 phoenix/core/layer5_prediction 層級的最高憲法。任何對模型參數、維度計算、遮罩邏輯的修改，皆需先更新此檔案內容。至此，Layer 5 的所有開發與定義工作已達 SOTA 完成度。