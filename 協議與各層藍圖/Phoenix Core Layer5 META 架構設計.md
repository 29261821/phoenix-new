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

1. pipeline.py (全市場預測協調器)

權責: 將各資產獨立的 Layer 2 特徵與 Layer 4 轉移機率進行聚合 (Gather)，進行全局運算。

META 約束: * 必須實作 step(transitions: Dict[str, TransitionState], reprs: Dict[str, RepresentationOutput]) -> MarketForecast。

特徵拼接紅線: 在進入神經網路前，每檔資產的輸入向量必須嚴格定義為 Concat[ X_t, P(S_{t+1}) ]。預測結果必須是「條件依賴於 Layer 4 下一個可能政權」的產物。

2. models/mdn_marginal.py (混合密度網路)

權責: 產出各個資產連續的機率密度邊際分佈（預測 $\mu$ 與 $\sigma$）。

META 約束:

必須輸出 pi (混合權重), mu (預期收益率), sigma (預期波動率)。

數學紅線 (MoE 融合): 單一資產最終的 expected_return 必須是 $\sum_{i=1}^{N} pi_i \times \mu_i$，完美契合「不同市場政權下的專家預測」。

3. models/covariance_net.py (動態條件共變異數網路) - ⚡ SOTA 核心

權責: 解決資產間的相關性突變（如：平時股債雙拼，崩盤時股債雙殺）。

META 約束:

必須根據全市場的 Latent Space 與政權狀態，動態生成 $M \times M$ 的共變異數矩陣 $\Sigma_t$。

矩陣正定性 (Positive Semi-Definite) 紅線: 神經網路直接輸出的矩陣通常不具備數學上的合法性。必須在最後一層套用 Cholesky 分解反推 $\Sigma = L L^T$ 或類似的激活機制，確保輸出的共變異數矩陣永遠是半正定的，否則下游 Layer 6 執行反矩陣運算時會直接崩潰拋錯 (Singular Matrix Error)。

4. models/frozen_masker.py (休市預測歸零器) - ⚠️ 跨市場防呆紅線

權責: 處理休市資產的無效預測。

META 約束:

在神經網路輸出全部預測後，必須介入檢查 is_frozen_mask。

歸零法則: 對於 is_frozen == True 的資產，強制將其 expected_returns 設為 0.0，expected_vols 設為一個極小值 (如 1e-6，防止除以零)。

共變異數隔離: 在 covariance_matrix 中，該休市資產對應的 Row 和 Column，除了對角線保留 1e-6 外，其餘與其他資產的協方差必須強制抹零 (0.0)。這確保休市資產在 Layer 6 分配時被視為「完全無關的零收益死水」，進而被自動捨棄。

🛡️ 實盤與回測行為一致性保證 (Streaming vs. Batch Consistency)

Target 構建的嚴格對齊 (Batch 模式): 在離線訓練準備 Label (真實 $\mu, \sigma, \Sigma$) 時，目標變數必須嚴格對應未來的真實收益率。且此 Label 矩陣在送入 Dataset 前，必須強制剔除最後 forecast_horizon 筆資料，防止模型在訓練的最後幾步看見 NaN 或產生錯位的未來污染。

無狀態矩陣推論 (Stateless Vectorized Inference): 預測層本身不維護跨時步的狀態 (由 L2/L4 負責)。對於相同的 $X_t$ 與 $P(S_{t+1})$ 矩陣組合，必須給出 100% 決定性的輸出。休市歸零器在 Batch 模式下必須使用 torch.where 進行矩陣級別的高速遮罩，嚴禁使用 for 迴圈處理。