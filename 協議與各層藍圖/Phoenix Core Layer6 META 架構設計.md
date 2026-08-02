Phoenix Core: Layer 6 Decision Layer - META 工程文件

系統版本: v1.5-Lite (100-Point Standard)
底層架構: Python (Logic) + numpy / scipy (Compute)
核心依賴: pydantic, numpy, scipy
上游輸入: 1. Layer 5: ForecastState (預期收益、波動率、VaR)
2. Layer 3: MarketState (當前市場政權，用於硬規則 Veto)
下游輸出: ActionState (最終目標部位權重、停損線、執行信心)

系統版本: v2.0-Pro (100-Point Standard, Multi-Asset Portfolio Optimization)
底層架構: Python (Logic) + cvxpy / numpy / scipy (Compute)
核心依賴: pydantic, numpy, cvxpy, scipy
上游輸入: 1. Layer 5: MarketForecast (全市場預期收益向量、共變異數矩陣、VaR、休市標記)
2. Layer 3: MarketState (各資產當前市場政權，用於硬規則 Veto)
下游輸出: PortfolioAction (全市場目標部位權重向量)

📂 實體目錄樹狀圖 (Directory Tree)

此層級完全脫離了價格 K 線，純粹基於多元機率分佈與風險矩陣進行「最優化投資組合分配」。在 v2.0-Pro 中，全面升級為支援約束最佳化 (Constrained Optimization) 的架構。

phoenix/
└── core/
    └── layer6_decision/              
        ├── __init__.py
        ├── schemas.py                # [介面契約] 多資產資金分配結構
        ├── pipeline.py               # 決策管線協調器 (Hybrid Brain)
        ├── allocators/               
        │   ├── __init__.py
        │   └── markowitz_kelly.py    # [SOTA核心] 多資產分數凱利/均值-變異數優化
        └── rules/                    
            ├── __init__.py
            ├── regime_veto.py        # [防爆] 針對單一資產的狀態否決器
            ├── frozen_constraint.py  # [防呆] 休市資產部位凍結約束器
            └── risk_overlay.py       # [防爆] 投資組合級別 (Portfolio-level) 降槓桿器


📜 介面契約與超參數配置 (Interface Contracts & Config) - ⚠️ 100分級約束

資金管理是量化交易的生命線。在多資產環境下，必須鎖死系統最大總槓桿、單一資產最大權重，以及整體的投資組合風險上限。

schemas.py 嚴格定義：

from pydantic import BaseModel, Field
import numpy as np
from typing import List

class DecisionConfig(BaseModel):
    """Layer 6 超參數配置契約"""
    max_gross_leverage: float = Field(1.0, description="投資組合最大總槓桿 (sum(abs(weights)))")
    max_asset_weight: float = Field(0.5, description="單一資產的最大絕對權重，防止單點重倉爆倉")
    kelly_fraction: float = Field(0.5, description="分數凱利，通常設為 0.5 (Half-Kelly) 避免破產")
    veto_states: List[int] = Field([0], description="禁止交易的 HMM 狀態 (e.g., State 0=低波死水)")
    max_portfolio_var_95: float = Field(0.05, description="投資組合級別允許的最大 95% VaR")
    turnover_penalty: float = Field(0.001, description="切換部位的摩擦成本預估")

class PortfolioAction(BaseModel):
    """Layer 6 輸出契約 (傳遞給 Execution Gateway)"""
    timestamp: int 
    assets: List[str]               # 長度為 M 的資產列表
    target_weights: np.ndarray      # Shape: (M,) - 最終目標部位 (-1.0 到 1.0)
    portfolio_leverage: float       # 當前總槓桿
    veto_triggered: np.ndarray      # Shape: (M,) - Boolean: 該資產是否被強制歸零
    reasoning: str                  # 決策溯源 (e.g., "Opt converged -> VaR Capped")

    class Config:
        arbitrary_types_allowed = True


📄 檔案權責與 META 規範 (File Responsibilities & META Specs)

1. pipeline.py (混合決策協調器)

權責: 實作 Rule Alpha (剛性規則) 與 ML Alpha (數學優化) 的融合。

META 約束: * 必須實作 step(forecast: MarketForecast, regimes: Dict[str, MarketState], current_weights: np.ndarray) -> PortfolioAction。

執行順序絕對紅線: 流水線必須嚴格遵守 Rules (產生邊界約束) -> Allocator (在約束內尋優) -> Risk Overlay (最終裁定) 的順序。

2. rules/frozen_constraint.py (休市部位凍結器) - ⚠️ 跨市場防呆紅線

權責: 處理「週末加密貨幣可以交易，但美股休市」的錯綜複雜狀況。

META 約束:

讀取 MarketForecast.is_frozen_mask。

物理限制法則: 如果某資產當下休市 (is_frozen == True)，我們無法買進也無法賣出。因此，必須向優化器 (Allocator) 輸出一個強制約束：target_weight[i] == current_weight[i]。

確保優化器在分配資金時，會將休市資產的既有風險（與佔用的資金）考慮進去，然後只對「目前開盤的資產」進行權重優化。

3. rules/regime_veto.py (狀態否決器)

權責: 阻斷在不適合的市場狀態下進行交易。

META 約束:

檢查每檔資產的 MarketState.dominant_state 是否存在於 veto_states。

如果觸發 Veto，且該資產未休市，向優化器輸出強制約束：target_weight[i] == 0.0 (強制平倉)。

4. allocators/markowitz_kelly.py (多資產優化大腦) - ⚡ SOTA 核心

權責: 根據 Layer 5 給出的預期收益率矩陣 $\mu$ 與共變異數矩陣 $\Sigma$，計算最優投資組合。

META 約束:

必須放棄簡單的公式解析解，全面採用 凸優化套件 (如 cvxpy)，因為我們有大量的不等式約束。

目標函數 (Objective): 最大化分數凱利期望增長率：$\max_{w} (w^T \mu - \frac{1}{2 f} w^T \Sigma w)$，其中 $f$ 是 kelly_fraction。

約束條件 (Constraints):

總槓桿約束：$\sum |w_i| \le max\_gross\_leverage$

單一資產約束：$-max\_asset\_weight \le w_i \le max\_asset\_weight$

傳入的硬規則約束：休市資產 $w_i = current\_weight_i$；Veto 資產 $w_i = 0$。

5. rules/risk_overlay.py (投資組合風險覆蓋器)

權責: 確保最優化出來的組合，整體的尾部風險不會超標。

META 約束:

接收 Allocator 算出的 Optimal_Weights ($w$)。

投資組合 VaR 計算: 透過 Layer 5 的共變異數矩陣計算組合波動率：$\sigma_p = \sqrt{w^T \Sigma w}$，並推導組合層級的 95% VaR。

降槓桿紅線: 如果 Portfolio_VaR > max_portfolio_var_95，必須按比例等距縮小所有「非休市」資產的權重，直到整體風險降至安全水位。

🛡️ 實盤與回測行為一致性保證 (Streaming vs. Batch Consistency)

摩擦成本與 Turnover 鉗制:

在優化目標函數中，必須加入 L1 正則化懲罰項：$- \lambda \sum |w_i - current\_weight_i|$。

這確保了系統不會因為 $\mu$ 發生了 0.01% 的微小改變，就去頻繁洗盤換倉，從數學根源上消滅 Ping-Pong 效應。

Singular Matrix 防禦:

雖然 Layer 5 已經保證了共變異數矩陣 $\Sigma$ 是半正定的，但在極端休市或資料共線性高時，$\Sigma$ 可能會接近奇異矩陣 (Singular)。

cvxpy 在求解時，必須傳入 psd_wrap(\Sigma) 或在對角線加入微小的 Ridge 懲罰項 ($\epsilon I$)，確保優化器在任何極端市況下都能在 100 毫秒內給出收斂的最優解 (Optimal Solution)，絕對不允許實盤拋出無解 Error。

🔌 系統對接邊界 (Phase 1 終點)

至此，輸出的 PortfolioAction 即為 Phoenix Core (6 層核心) 的最終產物。
這份包含所有資產權重 $w$ 的向量，將離開大腦的領域，傳送給 Execution Gateway (執行閘道器 / CCXT) 轉換為實際的子單 (Child Orders) 進行 TWAP/VWAP 拆單與執行。