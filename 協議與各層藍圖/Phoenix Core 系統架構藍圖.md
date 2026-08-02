鳳凰計畫 (Phoenix Project) - 核心六層架構藍圖

版本: v2.0-Pro (Phoenix Core - Multi-Asset SOTA Edition)
設計原則: 跨市場異步對齊、多世代模型工廠 (Strategy Pattern)、極致防禦前視偏差、多資產聯合最佳化。

🏗️ 核心六層架構 (The Core 6 Layers)

📊 Layer 1: Data Layer (數據管線層)

職責: 處理跨市場、跨時區的異步數據，將物理時間轉化為市場感知時間，並保留完整的「遺失資訊」。

核心機制 (v2.0 升級):

全局心跳 (Global Heartbeat): 以基準時間尺度 (如 1m) 推進全市場。

三維遺失編碼 (Missingness Encoding): 徹底棄用單純的 ffill。將特徵拆分為 Value (數值), Mask (布林值), Reason (休市/斷線/熔斷)，讓模型學會辨識「假性零波動」。

時間變形 (Time Deformation): 計算 $d\tau_t$，捕捉市場的高密度時刻（如 CPI 發布）與死水時刻。

輸出產物: MarketObservation (包含 3D 狀態與時間語意特徵)。

🧠 Layer 2: Representation Layer (市場表徵層)

職責: 神經網路路由器，將多通道高維數據壓縮為統一的 128 維潛在空間 (Latent Space)。

核心機制 (v2.0 升級):

多模型工廠 (Model Factory): 隨插即用的網路架構。

LSTM (基準對照組)

Causal TCN (Phoenix Lite 核心，處理時間感知閘門)

PatchTST (輕量 Transformer，處理 Attention Mask)

Mamba SSM (Phoenix Ultimate 核心，$O(1)$ 複雜度，直接將時間變形特徵映射為步長參數 $\Delta$)

元數據編碼 (Metadata Embedding): 注入資產種類、時區等物理屬性。

輸出產物: RepresentationVector (統一為 128 維的市場數學表徵)。

👁️ Layer 3: Regime Layer (市場政權辨識層)

職責: 在連續的 Latent Space 中，為每檔資產辨識離散的市場狀態（如：低波死水、高波趨勢）。

核心機制 (v2.0 升級):

狀態凍結器 (State Freezer): 當接收到 Layer 1 的「休市 Mask」時，強制凍結 HMM 前向機率的演化，防止狀態矩陣在非交易時段發生錯亂飄移。

物理語意對齊 (Alignment): 根據真實波動率由小到大排序隱藏狀態，解決 Label Switching 致命問題。

輸出產物: MarketState (當前市場狀態機率分佈 + 休市凍結標記)。

🔄 Layer 4: State Transition Layer (狀態轉移層)

職責: 計算市場「即將走向哪裡」，動態捕捉變盤的切換邊緣。

核心機制 (v2.0 升級):

時變轉移網路 (Dynamic TVTP): 打破靜態轉移矩陣，根據當前 128 維特徵動態生成 NxN 的轉移矩陣 $T_t$。

單位矩陣防呆 (Identity Freezer): 遇休市資產時，強制輸出單位矩陣 $I$，確保 $P(S_{t+1}) \equiv P(S_t)$，鎖死轉移機率。

輸出產物: TransitionState (下一刻的轉移矩陣與預測機率)。

🎯 Layer 5: Prediction Layer (聯合分佈預測層)

職責: 預測全市場未來的預期收益率、風險（波動）以及資產間的聯動性。

核心機制 (v2.0 升級):

全市場聚合 (Joint Distribution): 從單一資產預測躍升為多資產聯合預測。

動態條件共變異數網路 (Covariance Net): 動態輸出 $M \times M$ 的共變異數矩陣 $\Sigma_t$，並透過 Cholesky 分解保證矩陣的半正定性 (Positive Semi-Definite)。

休市歸零法則: 將休市資產的預期收益與跨資產協方差強制抹零，隔離雜訊。

輸出產物: MarketForecast (包含 Expected Returns 向量與 Covariance 矩陣的預測包)。

⚖️ Layer 6: Decision Layer (混合決策與資金大腦)

職責: 整合全市場預測與狀態約束，求解出數學上最優的投資組合權重。

核心機制 (v2.0 升級):

多資產凸優化 (Multi-Asset Kelly/Markowitz): 放棄單兵凱利，使用 cvxpy 求解最大化分數凱利期望的二次規劃問題。

約束條件聯動 (Constraints): * Regime Veto: 處於死水狀態的資產強制權重為 0。

Frozen Constraint: 休市資產權重強制鎖死為當前持倉 (無法買賣)。

投資組合尾部覆蓋 (Portfolio Risk Overlay): 計算整體組合的 95% VaR，若超標則按比例全局降槓桿。

輸出產物: PortfolioAction (目標部位向量：如 BTC 15%, ETH -5%, SPY 0% 等)。

💻 架構虛擬碼：v2.0-Pro 多資產狀態機 (Multi-Asset OOP Design)

以下展示 v2.0-Pro 架構如何在一個 step() 內，優雅地處理異步多資產與模型切換：

class PhoenixSystemV2:
    def __init__(self, config):
        # 1. 建立基礎管線
        self.l1_data = DataLayer(config)
        # 2. 工廠模式載入指定的網路 (e.g., 'mamba', 'tcn')
        self.l2_repr = RepresentationLayer(config.model_type) 
        self.l3_regime = RegimeLayer()
        self.l4_transition = StateTransitionLayer()
        self.l5_predict = PredictionLayer()
        self.l6_decision = DecisionLayer()

    def on_heartbeat(self, raw_market_stream):
        """全局心跳觸發 (例如每分鐘)"""
        
        # [Layer 1] 產出包含 Mask, Reason, Time Delta 的多資產觀測字典
        obs_dict = self.l1_data.step(raw_market_stream)

        # [Layer 2] 降維並注入資產 Metadata，輸出 128 維向量字典
        repr_dict = {
            asset: self.l2_repr.encode(obs) 
            for asset, obs in obs_dict.items()
        }

        # [Layer 3] 多資產狀態機 (自動處理休市凍結)
        regime_dict = {
            asset: self.l3_regime.detect(rep, obs.is_trading) 
            for asset, rep in repr_dict.items()
        }

        # [Layer 4] TVTP 轉移矩陣 (休市輸出單位矩陣 I)
        trans_dict = {
            asset: self.l4_transition.predict(regime_dict[asset], rep) 
            for asset, rep in repr_dict.items()
        }

        # [Layer 5] 聚合！預測全市場聯合分佈與共變異數矩陣
        forecast = self.l5_predict.forecast_joint(repr_dict, trans_dict, obs_dict)

        # [Layer 6] 多資產凸優化求解 (cvxpy)
        portfolio_action = self.l6_decision.allocate(forecast, regime_dict)
        
        return portfolio_action
