Phoenix Core: Layer 2 Representation Layer - META 工程文件

系統版本: v1.5-Lite (100-Point Standard)
底層架構: WSL + CUDA (Compute) + Python (Logic)
核心依賴: torch, polars, numpy, pydantic
上游輸入: Layer 1: MarketObservation (Polars DataFrame)
下游輸出: RepresentationVector (低維度密集向量)

📂 實體目錄樹狀圖 (Directory Tree)

phoenix/
└── core/
    └── layer2_repr/
        ├── __init__.py
        ├── schemas.py                # [介面契約]
        ├── pipeline.py               # 表徵管線協調器 (負責切換濾網)
        ├── preprocessors/
        │   ├── __init__.py
        │   ├── rolling_scaler.py
        │   └── tensor_builder.py     # [更新] 多輸入通道拼接
        ├── models/                   # 🚀 Representation Family (濾網庫)
        │   ├── __init__.py
        │   ├── model_factory.py      # 模型路由器 (根據 Config 決定載入哪個網路)
        │   ├── metadata_embedder.py  # [新增] 資產元數據靜態編碼器
        │   ├── lstm_baseline.py      # [論文對照組] 標準雙向或單向 LSTM
        │   ├── tcn_encoder.py        # [Lite版核心] 時間感知因果網路
        │   ├── patchtst_encoder.py   # [Pro版預留] 輕量級時間序列 Transformer
        │   └── mamba_encoder.py      # [Ultimate預留] 狀態空間模型 (SSM)
        └── weights/
            ├── __init__.py
            └── checkpoint_mgr.py     


📜 介面契約與超參數配置 (Interface Contracts & Config)

神經網路的輸入維度大幅擴增，必須精確定義哪些維度對應數值，哪些對應 Mask 與 時間。
透過 model_type 鎖死切換開關。無論底層是 Transformer 還是 LSTM，對外輸出的 embedding_dim 永遠一致，確保 Layer 3 不會崩潰。

schemas.py 嚴格定義：

from pydantic import BaseModel, Field
import numpy as np
from typing import Literal

class RepresentationConfig(BaseModel):
    """Layer 2 超參數配置契約"""
    # 決定當前掛載哪一種「濾網」
    model_type: Literal["lstm", "tcn", "patchtst", "mamba", "mamba_attention"] = Field("tcn")
    
    value_channels: int = Field(..., description="原始特徵數量")
    metadata_dim: int = Field(16, description="資產類型 Embeddings 的輸出維度")
    time_channels: int = Field(3, description="時間變形特徵數量")
    
    # 【絕對紅線】無論是哪個模型，最終吐出的降維向量必須是這個長度
    embedding_dim: int = Field(128, description="最終融合輸出的 Latent Space 維度")
    window_size: int = Field(128, description="推論所需的序列長度 (Mamba 可設為1)")

class RepresentationOutput(BaseModel):
    """Layer 2 輸出契約 (統一介面)"""
    timestamp: int
    asset_id: str
    embedding: np.ndarray # Shape: (embedding_dim,)
    model_family: str     # 記錄是由哪個模型產出的 (e.g., "tcn", "mamba")，供除錯與回測比對

    class Config:
        arbitrary_types_allowed = True


📄 檔案權責與 META 規範 (File Responsibilities & META Specs)

1. pipeline.py (表徵管線協調器與記憶體管理) & models/model_factory.py (模型工廠)

權責: 實踐「濾網抽插」的軟體工程模式，並精準管理不同網路的狀態記憶。

META 約束 (工廠模式):

必須實作 class RepresentationLayer。

pipeline.py 初始化時，讀取 config.model_type，並呼叫 model_factory.build(config)。工廠會根據字串，回傳對應的 PyTorch 模型實例 (如 TCNEncoder 或 MambaEncoder)。

所有的 Encoder 模型都必須繼承一個基礎類別 BaseRepresentationModel，該類別強制定義了 forward(value, mask, reason, time) -> embedding 的函數簽名。

META 約束 (狀態記憶與效能):

狀態記憶 (Stateful Memory): 實盤中，模型不能總是只吃最新一根 K 線。pipeline.py 必須根據模型類型動態維護：

若為 TCN/PatchTST：必須維護一個長度等於 RepresentationConfig.window_size 的 FIFO 緩衝區 (Buffer/Deque)。每次呼叫 step(obs: MarketObservation) 時，將最新特徵推入 Buffer，然後將整個 Buffer 送進 Encoder。

若為 Mamba：因其具備 $O(1)$ 的內部隱藏狀態，step() 僅需餵入最新一根 K 線。這段狀態機的差異必須在 pipeline.py 中被優雅地封裝，對上層 (Layer 3) 隱藏。

效能紅線: step() 過程中，必須使用 with torch.no_grad():，並確保 Tensor 處於 .eval() 模式。


2. preprocessors/rolling_scaler.py (防爆標準化與 Mask 感知) - ⚠️ 系統絕對紅線
權責: 解決神經網路對特徵尺度 (Scale) 敏感的問題，同時避開前視偏差與休市雜訊污染。
META 約束:
禁止全局 Fit: 嚴禁使用 sklearn.preprocessing.StandardScaler().fit_transform(all_data) 偷看未來。
Mask 感知滾動 (Mask-Aware Rolling): 必須實作基於 Welford 演算法的「在線標準化 (Online Z-Score)」。

數學紅線: $Z_t = \frac{x_t - \mu_{t-w:t}}{\sigma_{t-w:t}}$。在更新均值 $\mu$ 與標準差 $\sigma$ 的滑動視窗時，必須排除 features_mask == False (休市或遺失) 的數據。絕不能讓 ffill 產生的「假性零波動」拉低了真實市場的波動率計算，否則會導致 Z-score 在開盤瞬間發生極端暴衝。

3. models/metadata_embedder.py (元數據靜態編碼器)
權責: 讓模型知道它現在看的是「24 小時的 BTC」還是「有夜盤的台指期」，提供跨資產的物理語境。
META 約束:類別對映: 必須使用 torch.nn.Embedding 來處理 asset_type、timezone 與 region 這類類別型變數 (Categorical Variables)。
架構解耦 (Agnostic Injection): 這是一個 Static Embedding，不隨時間 $T$ 改變。無論管線後端掛載的是 LSTM、TCN 還是 Mamba，這個靜態向量都必須在進入時序 Encoder 核心之前，透過 Broadcasting (廣播機制) 擴展至與時間序列相同長度，並進行 Concatenation (拼接) 或加法融合，確保所有的 Representation Family 都能感知資產屬性。

4. preprocessors/tensor_builder.py (多通道張量建構與字典封裝)
權責: 將 Layer 1 的 3D Missingness 矩陣與時間變形矩陣，打包為下游多模型相容的標準格式。
META 約束:統一字典輸出: 為了相容 LSTM、Transformer、SSM 等不同網路對輸入的要求，嚴禁將所有特徵硬拼接成單一維度。
必須輸出標準化 Dictionary 格式：{ "value": tensor, "mask": tensor, "time_delta": tensor, "reason": tensor }。
這讓下游模型可以各取所需（例如決定要把 mask 當作 Attention Bias，還是直接忽略）。
防爆紅線 (Zero-out): 對於 value 張量（形狀為 (Batch, Channels, Seq_Len)），在 mask == False 的時間點，Scaled_Value 必須強制歸零。
讓網路在該時間點完全喪失對價格數值的依賴，只能透過 mask 與 reason 的 Embedding 來學習「休市」或「斷線」的語意，徹底避免吸收虛假的 0 波動雜訊。

5. 模型家族規範 (Representation Families) - ⚡ 核心演算法庫
權責: 根據 Config 掛載對應的網路架構，將多通道特徵融合，產出具備物理與時間語意的 128 維市場表徵。
META 約束:lstm_baseline.py (論文對照組): 最陽春的基準線 (Benchmark)。不處理時間變形 ($d\tau$)，單純將所有 Channel 拼接後餵給 LSTM 進行訓練。
tcn_encoder.py (Phoenix Lite): 實作 Causal Conv1d，並需要維護 window_size 的緩存。Time Deformation Injection: 必須將字典中的 time_delta (時間變形係數) 結合進網路。
實作方式規定為：在每一個 Residual Block 的 Activate Function 之前，將輸入矩陣乘上 $d\tau_t$ 轉換而來的時間感知閘門 (Time-aware Gate)。
當市場進入「休市 ($d\tau=0$)」時，網路的隱藏狀態會自動凍結（不衰減）；當進入「數據發布 ($d\tau$ 飆高)」時，模型會自然賦予該 K 線極高的激活權重。
patchtst_encoder.py (輕量 Transformer - Pro 版預留): 將時間序列切割成 Patch，利用 Attention 機制學習。對於 mask == False 的位置，必須將其轉換為 Attention Mask，強迫模型在自注意力計算時，完全忽略該區塊的權重。
mamba_encoder.py (Phoenix Core/Ultimate): 狀態空間模型 (SSM) 實作。優勢紅線: Mamba 的核心靈魂是 $\Delta$ (步長參數)。這裡必須將傳入的 time_delta 直接 Mapping 給 Mamba 的 $\Delta$。當市場休市，$\Delta \to 0$，Mamba 狀態精準凍結；市場劇震，$\Delta$ 放大，狀態快速更新。這是 Mamba 輾壓 Transformer 的終極物理意義。

🔌 下游依賴 (Layer 3 & 4 預告)權責: 規範 Layer 2 的產出如何在後續的馬可夫政權與預測管線中被安全消費。
META 約束:產出的 RepresentationOutput.embedding（依據 v2.0-Pro 規範強制對齊為 128 維度，修正原草案中的 64 維度不一致）是目標資產在當前心跳時刻的「潛在空間切片 (Latent Space Slice)」。
該向量將被分發至下游進行解耦運算：Layer 3 (Regime Layer): 作為高斯隱馬爾可夫模型 (Gaussian HMM) 的直接觀測輸入，用以判斷這 128 個非線性特徵當前落在哪個已進行語意對齊的「政權聚類 (Regime Cluster)」。
Layer 4 & 5 (Transition & Prediction Layers): 作為動態時變轉移網路 (TVTP) 與混合密度網路 (MDN) 的最核心條件特徵 ($X_t$)，用以推導下一個時間步的狀態轉移機率矩陣與連續型的預期收益率分佈曲線。
🛡️ 實盤與回測行為一致性保證 (Streaming vs. Batch Consistency)權責: 確保模型在離線平行運算 (Batch) 與線上逐 K 線推進 (Streaming) 時，內部的隱藏狀態、時間步長與遮罩處理在數學上完全等價。META 約束:介面多型化: 無論管線底層掛載的是哪一個模型家族，pipeline.py 都必須對外提供完全一致的 step() (逐 K 線實盤) 與 batch() (大範圍回測) 介面。動態記憶體隱蔽紅線: * 若激活 TCN / PatchTST：step() 必須從 Layer 1 接收當前心跳的 MarketObservation，將其推入長度為 window_size (128) 的 FIFO 隊列，通過 tensor_builder 打包成 3D 張量後送入網路，並抽取最後一幀。
若激活 Mamba：因其具備 $O(1)$ 的循環狀態 (Recurrent State)，step() 僅需將當前心跳的單一特徵字典傳入。
Mamba 核心會利用 Layer 1 算出的 time_delta（即當前時間變形係數）直接映射為 $\Delta$，自動完成內部隱藏狀態的動態衰減與即時更新，不需在外部維護巨型時間窗口 Buffer。這段底層狀態機的本質差異必須在 pipeline.py 內部被完全封裝。
數學一致性邊界: 必須確保在相同資料輸入下，Batch 模式的平行掃描 (Parallel Scan / Convolutional Invariance) 結果，與 Streaming 模式的逐步流式迭代結果，在 PyTorch 中能通過 torch.allclose(atol=1e-6) 的嚴格驗證，從源頭徹底掐死回測賺錢、實盤虧錢的「代碼非對齊漏洞」。