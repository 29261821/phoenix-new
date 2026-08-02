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

1. 介面契約與超參數配置 (schemas.py)系統定位: Layer 6 的 I/O 絕對物理邊界與超參數鋼印。所有進入 Layer 6 運算的變數，必須無條件符合此合約，任何維度錯置必須在此層直接攔截。🚨 類別命名與依賴絕對約定允許依賴: pydantic (BaseModel, Field), numpy, typing (List)。禁止依賴: 嚴禁引入 torch、pandas 或 cvxpy，維持純粹的資料合約性質。類別名稱: 必須精確命名為 DecisionConfig 與 PortfolioAction。📐 核心屬性與張量防爆邊界DecisionConfig 絕對約束 (包含預設值紅線):max_gross_leverage (float): 必須預設為 1.0。此為系統總資金的物理極限，嚴禁預設超過 1.0 (無槓桿狀態)。max_asset_weight (float): 必須預設為 0.5。防止單一資產單點重倉的爆倉紅線。kelly_fraction (float): 必須預設為 0.5。分數凱利約束，避免黑天鵝破產。veto_states (List[int]): 必須預設為 [0]。代表極端低波動死水狀態。max_portfolio_var_95 (float): 必須預設為 0.05。turnover_penalty (float): 必須預設為 0.001。PortfolioAction 絕對約束 (輸出合約):timestamp (int): 必須為 Unix Timestamp (毫秒)，嚴格對齊 Layer 5 輸入。assets (List[str]): 註冊資產池列表，長度為 $M$。target_weights (np.ndarray): 最終目標權重。防爆約定: 必須為一維 NumPy 矩陣，Shape 嚴格鎖死為 (M,)。數值必須介於 [-max_gross_leverage, max_gross_leverage] 之間。portfolio_leverage (float): 總部位絕對值加總。veto_triggered (np.ndarray): Boolean 矩陣，Shape 嚴格為 (M,)。reasoning (str): 決策溯源。Pydantic 防呆: 必須宣告 class Config: arbitrary_types_allowed = True 以允許 NumPy 型別檢驗。2. 混合決策協調器 (pipeline.py)系統定位: 實作 Rule Alpha (剛性規則) 與 ML Alpha (數學凸優化) 的總指揮，絕不允許任何越權調用。🚨 類別命名與依賴絕對約定允許依賴: numpy (np), typing (Dict, Any)。內部依賴: 必須精確匯入自身目錄下的 DecisionConfig, PortfolioAction, FrozenConstraint, RegimeVeto, MarkowitzKellyAllocator, RiskOverlay。類別名稱: 必須精確命名為 DecisionLayer。⚙️ 核心方法與實例綁定約定__init__(self, config: DecisionConfig)內部綁定絕對紅線: 必須且只能實例化以下物件，嚴禁在此維護跨時間步的狀態矩陣（如 self.prev_weights 等），確保系統為無狀態 (Stateless) 純函數架構：self.config = configself.frozen_constraint = FrozenConstraint()self.regime_veto = RegimeVeto(veto_states=config.veto_states)self.allocator = MarkowitzKellyAllocator(config=config)self.risk_overlay = RiskOverlay(max_portfolio_var_95=config.max_portfolio_var_95)step(self, forecast: Any, regimes: Dict[str, Any], current_weights: np.ndarray) -> PortfolioAction介面對齊約束: forecast 必須對齊 Layer 5 的 MarketForecast 結構，regimes 必須包含 asset_id 映射到 Layer 3 MarketState 的結構。current_weights Shape 必須為 (M,)。⚡ 執行順序與防爆絕對紅線 (Chronological Execution Guard)在實盤推論 step() 內部，必須嚴格遵守以下三個 Phase 的執行順序，嚴禁顛倒、平行處理或跨區塊污染：Phase 1: Rules (產生邊界約束)第一步：呼叫 self.frozen_constraint.get_mask(forecast)，產出 frozen_mask (Shape: (M,))。第二步：呼叫 self.regime_veto.get_mask(regimes, forecast.assets)，產出 veto_mask (Shape: (M,))。紅線: 必須先取得物理與狀態遮罩，絕不能先做最佳化。Phase 2: Allocator (約束內尋優)必須呼叫 self.allocator.optimize(...)。傳入順序必須精確為：expected_returns, covariance_matrix, current_weights, frozen_mask, veto_mask。產出中繼變數：optimal_weights。Phase 3: Risk Overlay (最終裁定)必須呼叫 self.risk_overlay.apply(...)。傳入參數必須包含 optimal_weights, covariance_matrix, frozen_mask。紅線: 降槓桿器必須擁有最終裁量權，其產出的 final_weights 才是唯一合法的持倉目標。🔄 實盤與回測行為一致性保證無狀態純函數約束 (Stateless Pure Function): step 方法內部絕不可依賴 self 存放的歷史權重。當前權重 (current_weights) 必須由外部 (Execution Gateway 或 Backtest Engine) 動態餵入，確保回測時的部位計算與實盤的帳戶真實部位完全一致，徹底根絕「帳面權重與物理權重脫鉤」的致命錯誤。


3. 休市部位凍結器 (rules/frozen_constraint.py)系統定位: 跨市場非同步休市隔離閘道。負責處理「週末加密貨幣開盤，但美股休市」等錯綜複雜情境，為優化器提供強制部位鎖死的遮罩。🚨 類別命名與依賴絕對約定允許依賴: numpy, typing (Any)。禁止依賴: 嚴禁引入 pandas 或任何高階資料處理庫，確保記憶體操作在最底層的 C 陣列級別進行。類別名稱: 必須精確命名為 FrozenConstraint。⚙️ 核心方法簽名與約定get_mask(self, forecast: Any) -> np.ndarray輸入型別約定: 接收 Layer 5 的 MarketForecast 實例（此處標註 Any 是為了解耦，但必須預期其具備 is_frozen_mask 屬性）。輸出型別約定: 必須回傳一維 numpy.ndarray。⚡ 物理狀態隔離與張量防爆邊界 (Mask Generation Guard)強制轉型紅線: 方法內部提取遮罩時，必須使用 np.asarray(forecast.is_frozen_mask, dtype=bool) 進行強制轉型與拷貝。嚴禁直接回傳原始的可能是 List 或 Pandas Series 的物件，徹底杜絕下游 cvxpy 優化時發生索引錯位 (Index Mismatch) 導致的維度崩潰。Shape 絕對邊界: 輸出的 boolean 陣列長度必須嚴格等於 $M$ (資產數量)。物理語意鎖死: 陣列中的 True 嚴格代表該資產「當下無法進行任何交易 (包含買入、賣出與平倉)」，其目標權重必須在優化器階段被鎖死為當前權重 (current_weights)。4. 狀態否決器 (rules/regime_veto.py)系統定位: 防禦性交易狀態攔截網。基於無監督學習辨識的政權 (Regime) 狀態，強制阻斷在「死水」或「極端不可測」市場環境下的部位曝險。🚨 類別命名與依賴絕對約定允許依賴: numpy, typing (Dict, Any, List)。類別名稱: 必須精確命名為 RegimeVeto。⚙️ 核心方法簽名與約定__init__(self, veto_states: List[int])初始化約束: 必須在建構子中將傳入的 veto_states 轉換為 Python set (self.veto_states = set(veto_states))，以保證在大量資產輪詢時，狀態比對的時間複雜度被壓制在 $O(1)$。get_mask(self, regimes: Dict[str, Any], assets: List[str]) -> np.ndarray輸入型別約定:regimes: 字典，Key 為 asset_id，Value 為 Layer 3 的 MarketState 實例。assets: 來自 Layer 5 的資產列表（長度為 $M$）。輸出型別約定: 必須回傳一維 numpy.ndarray (dtype=bool)。⚡ 執行邏輯與極端防爆紅線 (Defensive Default Logic)在此方法的 for i, asset in enumerate(assets): 迴圈內，必須嚴格遵守以下防禦與裁定順序：無狀態防爆攔截 (Default Veto Red Line):必須安全提取狀態物件：state_obj = regimes.get(asset)。絕對紅線: 如果 state_obj is None (例如：某資產 API 斷線導致 Layer 3 未能給出預測)，系統嚴禁瞎猜或放行。必須立刻觸發「預設否決」，將 veto_mask[i] 設為 True 並 continue。政權比對裁定:如果狀態存在，檢查 state_obj.dominant_state in self.veto_states。若為真，則將 veto_mask[i] = True。Shape 絕對邊界:初始化的遮罩必須為 np.zeros(len(assets), dtype=bool)，保證輸出的維度必然精確為 (M,)，絕不會引發陣列越界。物理衝突處理協議 (Veto vs Frozen):設計綱領: 若一檔資產同時被標記為 Frozen (休市) 與 Veto (狀態否決)，在物理現實中，休市資產絕對無法被強制平倉。因此 RegimeVeto 僅負責標示 Veto，真正的衝突仲裁必須交由下游的 MarkowitzKellyAllocator 進行 (即 veto_mask & ~frozen_mask)，此處嚴禁擅自改寫 Frozen 狀態。🔄 實盤與回測行為一致性保證FrozenConstraint 與 RegimeVeto 的所有判斷邏輯，皆為基於「當下 Timestamp 單一切片」的純函數操作。無論在 step() 實盤單步推進，或是在回測引擎的 for 迴圈中被呼叫，記憶體中均不可存留上一根 K 棒的布林遮罩，保證 100% 的無污染推論。

5. 投資組合風險覆蓋器 (rules/risk_overlay.py)系統定位: 投資組合級別 (Portfolio-level) 尾部風險覆蓋器。負責防範最優化出來的組合波動率超標，利用嚴謹的二次方程式計算非休市部位的最佳縮放比例。  🚨 類別命名與依賴絕對約定允許依賴: numpy, typing (Tuple)。禁止依賴: 嚴禁引入任何機器學習庫或優化器套件。類別名稱: 必須精確命名為 RiskOverlay。⚙️ 核心方法簽名與約定__init__(self, max_portfolio_var_95: float)轉換約定: 必須在初始化時，將傳入的 95% VaR 基準 (假設常態分佈單尾 $Z \approx 1.64485$) 轉換並預先計算出系統允許的最大變異數 self.max_var，減少實盤單步運算時間。apply(self, optimal_weights: np.ndarray, covariance_matrix: np.ndarray, frozen_mask: np.ndarray) -> Tuple[np.ndarray, str]輸入: 優化器計算的權重 (Shape: M,)、共變異數矩陣 (Shape: M, M)、休市遮罩 (Shape: M,)。輸出: 經過尾部風險裁定的最終安全權重與裁定字串。⚡ 降槓桿數學與防爆絕對紅線 (Quadratic De-leveraging Guard)投資組合變異數計算: 必須使用 $w^T \Sigma w$ 計算組合波動率平方。若結果小於等於 self.max_var，直接放行 w。休市分離紅線: 若風險超標，嚴禁對全體權重等比例縮放。必須利用 np.where 將部位嚴格拆解為 w_froz (休市部位) 與 w_act (活躍部位)。二次方程式求解約定: 必須建立方程式求解縮放比例 $k$ ($0 \le k \le 1$)：$$(w_{froz} + k \cdot w_{act})^T \Sigma (w_{froz} + k \cdot w_{act}) = max\_var$$必須將其展開並定義係數：$A = w_{act}^T \Sigma w_{act}$$B = 2(w_{act}^T \Sigma w_{froz})$$C = w_{froz}^T \Sigma w_{froz} - max\_var$極端數值防禦 (Discriminant Guard):若 $A < 1e-8$ (活躍部位幾無變異數)，必須直接強制令 $k = 0.0$。若 $C > 0$ (僅休市部位的變異數就已經超標)，必須直接強制令 $k = 0.0$。若計算判別式 $B^2 - 4AC < 0$，為防護數值誤差，強制令 $k = 0.0$。邊界鉗制紅線: 最終解出的 $k$ 必須透過 np.clip(k, 0.0, 1.0) 強制鎖死在合法比例區間內，保證不發生反向建倉或加槓桿。6. 多資產優化大腦 (allocators/markowitz_kelly.py)系統定位: SOTA Sizing Engine。徹底放棄解析解，全面採用凸優化套件 (CVXPY)，並支援包含 $L1$ 正則化在內的複雜不等式約束。  🚨 類別命名與依賴絕對約定允許依賴: numpy, cvxpy (cp), typing (Tuple), ..schemas (DecisionConfig)。禁止約定: 嚴禁使用 scipy.optimize.minimize，因其對大規模不等式約束與稀疏矩陣的求解穩定度不及內建 QP (Quadratic Programming) 求解器。類別名稱: 必須精確命名為 MarkowitzKellyAllocator。⚙️ 核心方法簽名與約定__init__(self, config: DecisionConfig)微小值注入: 必須綁定超參數，並定義 self.epsilon_ridge = 1e-6，作為矩陣正定防護的超級常數。optimize(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray, current_weights: np.ndarray, frozen_mask: np.ndarray, veto_mask: np.ndarray) -> Tuple[np.ndarray, str]⚡ 凸優化目標函數約定 (遵循核心鋼印：公式標誌與物理定義)程式碼註解必須包含以下嚴格物理定義，確保開發者不發生維度幻覺：目標方程式:$$\max_{w} \left( w^T \mu - \frac{1}{2f} w^T \Sigma w - \lambda \sum |w_i - w_{prev, i}| \right)$$物理單位與變數定義:  $w$ (權重): 單位 - 無單位比例 (如 0.15 = 15%)。$\mu$ (預期收益): 單位 - 收益率 (每 K 棒預期百分比)。$f$ (凱利分數): 單位 - 無單位常數。$\Sigma$ (共變異數): 單位 - 變異數 (收益率的平方)。$\lambda$ (換倉懲罰): 單位 - 摩擦耗損率。$w_{prev}$ (前次權重): 單位 - 無單位比例。🛡️ 矩陣防爆與求解器絕對紅線 (Solver Constraints & Fallbacks)奇異矩陣絕對防禦 (Singular Matrix Guard):進入目標函數前，必須對 $\Sigma$ 加上 Ridge 懲罰項：safe_sigma = covariance_matrix + np.eye(n_assets) * self.epsilon_ridge。紅線: 必須使用 cp.psd_wrap(safe_sigma) 將矩陣強制標記為半正定，否則 cvxpy 在處理神經網路輸出的極端矩陣時會直接拒絕構建模型。  約束條件優先級 (Constraint Hierarchy):加入總槓桿 $L1$ 約束與單一資產上下界約束。  Frozen 約束紅線: 休市資產權重強制鎖死 w[idx] == current_weights[idx]。  Veto 約束衝突防護: 若資產觸發 Veto，強制 w[idx] == 0.0。紅線：必須使用 veto_mask & ~frozen_mask，保證「無法交易的休市資產」絕不被賦予平倉約束，防止系統產生無解條件 (Infeasible)。  求解器退避機制 (Solver Fallback Protocol):第一段: 必須優先呼叫 problem.solve(solver=cp.OSQP, warm_start=True) 以榨取極致效能。第二段 (退避): 若回傳狀態非 optimal，必須在 try-except 區塊內攔截，並切換至 solver=cp.ECOS 重新求解。終極防爆紅線: 若所有求解器拋出異常，或解出狀態仍無效 (Infeasible / None)，必須直接回傳 current_weights.copy() (保持原部位不動)，嚴禁實盤系統拋錯崩潰。浮點誤差清理 (Floating Dust Cleanup):最終回傳的數值陣列，必須經過 np.where(np.abs(w) < 1e-6, 0.0, w) 過濾，防止產生如 1.42e-15 的假性微小部位，引發執行端的微結構無限下單錯誤。📢 META 規範完成聲明本文件 (Part 1 至 Part 3) 為 phoenix/core/layer6_decision 層級的最高憲法。任何對資金分配規則、優化器目標函數或降槓桿機制的修改，皆需先檢驗是否違反此檔案宣告的物理限制。至此，Layer 6 的 SOTA 開發與定義工作已具備 100% 絕對復現能力。