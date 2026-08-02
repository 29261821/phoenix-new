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

1. 介面契約與超參數 (schemas.py)

系統定位: Layer 2 的 I/O 絕對物理邊界。

🚨 類別命名與依賴絕對約定

允許依賴: pydantic (BaseModel, Field), numpy, typing (Literal)。

禁止依賴: 嚴禁引入 torch, pandas, 或任何神經網路相關庫，維持純粹的資料合約性質。

類別名稱: 必須精確命名為 RepresentationConfig 與 RepresentationOutput。

📐 核心屬性與張量防爆邊界

RepresentationConfig 絕對約束 (包含預設值紅線):

model_type: 必須為 Literal["lstm", "tcn", "patchtst", "mamba", "mamba_attention"]，預設 "tcn"。

value_channels: 必須為 int (無預設值)。

metadata_dim: 必須預設為 16。

time_channels: 必須預設為 3。

embedding_dim: 絕對維度紅線。必須強制預設為 128。這是 Layer 3 狀態機的唯一認可輸入維度，擅自修改將導致下游矩陣乘法直接崩潰。

window_size: 必須預設為 128。

RepresentationOutput 絕對約束:

timestamp: 必須為 int (Unix Timestamp 毫秒)。

asset_id: 必須為 str。

embedding: 必須為 np.ndarray。Shape 防爆約定: 長度必須嚴格等於 (embedding_dim,) 的一維向量。

model_family: 必須為 str。

Pydantic 防呆: 必須宣告 class Config: arbitrary_types_allowed = True。

2. 表徵管線協調器 (pipeline.py)

系統定位: Layer 2 狀態記憶管理與推論引擎總管。

🚨 類別命名與依賴絕對約定

允許依賴: torch, polars (別名 pl), numpy (別名 np), typing (Dict, List), collections (deque)。

禁止依賴: 嚴禁引入 pandas。

內部依賴: 必須匯入 MarketObservation (L1 契約), RepresentationConfig, RepresentationOutput, ModelFactory, MetadataEmbedder, WelfordMaskAwareScaler, TensorBuilder。

類別名稱: 必須精確命名為 RepresentationLayer。

⚙️ 核心方法與實例綁定約定 (Instance Binding)

__init__(self, config: RepresentationConfig, asset_map: Dict[str, int])
內部綁定絕對紅線: 必須且只能實例化以下物件，嚴禁在此遺漏或新增其他狀態：

self.config = config

self.asset_map = asset_map

self.metadata_embedder = MetadataEmbedder(num_assets=len(asset_map), metadata_dim=config.metadata_dim)

self.encoder = ModelFactory.build(config)

self.scaler = WelfordMaskAwareScaler(window_size=config.window_size, num_features=config.value_channels)

self.tensor_builder = TensorBuilder(self.scaler)

推論模式鎖死: 必須在結尾呼叫 self.metadata_embedder.eval() 與 self.encoder.eval()。

step(self, obs: MarketObservation) -> RepresentationOutput

batch(self, obs_history: List[MarketObservation]) -> pl.DataFrame

🔄 實盤與回測行為一致性保證 (Streaming vs Batch)

狀態記憶體隱蔽紅線 (Stateful Memory Guard):

__init__ 內部必須動態計算 buffer_len：如果 config.model_type 包含 "mamba"，buffer_len 強制設為 1 ($O(1)$ 隱藏狀態)；否則設為 config.window_size。

多資產隔離: 必須建立 self.obs_buffers，為 asset_map 中的每一個資產實例化一個獨立的 deque(maxlen=buffer_len)。

⚡ step 執行順序與防爆絕對紅線 (Data Plumbing)

在實盤推論 step(self, obs) 內部，必須嚴格遵守以下執行順序與參數傳遞，嚴禁顛倒或更名：

資產註冊檢驗: 第一行檢查 obs.asset_id in self.asset_map，若無則拋出 ValueError。

狀態更新與暖機 (Warm-up) 攔截:

將 obs 推入該資產專屬的 deque。

檢查 len(current_buffer) < self.buffer_len。若未滿，必須立即回傳 RepresentationOutput，其 embedding 為全零 np.zeros(config.embedding_dim)。

時序對齊 (Polars Concat): 必須依序抽出並 pl.concat 四個特徵：features_value, features_mask, time_features, features_reason。

張量建構約定 (Tensor Building):

呼叫 tensor_dict = self.tensor_builder.build_tensor_dict(...)。

傳入參數順序必須精確為：(concat_value, concat_mask, concat_time, concat_reason)。

無梯度推論隔離 (No Grad Red Line):

接下來的所有運算，必須被包覆在 with torch.no_grad(): 區塊內。

元數據編碼約定 (Metadata Extraction):

必須建立張量：asset_idx_tensor = torch.tensor([self.asset_map[asset_id]], dtype=torch.long)。

必須呼叫：metadata_emb = self.metadata_embedder(asset_idx_tensor, seq_len=self.buffer_len)。

編碼器前向傳播約定 (Encoder Forward):

必須呼叫 self.encoder(...) 且強制使用 Keyword Arguments 對齊介面：
value=tensor_dict["value"]
mask=tensor_dict["mask"]
reason=tensor_dict["reason"]
time_delta=tensor_dict["time_delta"]
metadata_emb=metadata_emb

維度降維與輸出封裝:

提取 Batch 0 的向量：embedding_tensor[0].cpu().numpy()。

封裝為 RepresentationOutput 並回傳。

3. 防爆預處理器 (Preprocessors)

3.1 遮罩感知在線標準化 (rolling_scaler.py)

系統定位: 徹底消滅前視偏差與休市 0 波動雜訊的特徵縮放器。

允許依賴: 僅限 numpy。嚴禁使用 pandas 或 sklearn。

類別名稱: 必須精確命名為 WelfordMaskAwareScaler。

實例綁定約定: __init__(self, window_size: int, num_features: int)

必須初始化 self.mean, self.M2, self.buffer。

精度防爆紅線: 內部狀態矩陣必須強制宣告為 dtype=np.float64，防止滾動更新時發生數值漂移與精度喪失。

核心方法: step(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray

資料過濾紅線: 必須在計算均值與變異數前，檢查 is_valid (mask 為 True)。若為 False，嚴禁將該筆資料納入 Welford 狀態機更新，防止休市平移導致 $\sigma$ 暴跌。

除零防爆紅線: 輸出前計算標準差時，必須包含 variance = np.where(variance < 1e-8, 1e-8, variance)。

實盤與回測一致性: batch_transform 方法內部必須透過 for 迴圈依序呼叫 step() 來處理時間序列，絕對嚴禁使用矩陣全局操作偷看未來。

3.2 多通道張量建構器 (tensor_builder.py)

系統定位: 打包 3D 張量並執行最終物理歸零的防爆閘道。

允許依賴: torch, polars, numpy, typing (Dict)。

類別名稱: 必須精確命名為 TensorBuilder。

核心方法: build_tensor_dict(self, features_value, features_mask, time_features, features_reason) -> Dict[str, torch.Tensor]

形狀擴展約定: 必須透過 np.broadcast_to 將一維的 mask 擴張為與 value 相同的形狀。

🚨 終極防爆紅線 (Zero-out 歸零法則):

必須嚴格執行：scaled_val = np.where(mask_expanded, scaled_val, 0.0)。

物理意義: 對於 mask == False 的時間點，價格與成交量特徵必須被強制抹平為絕對的 0.0。這強迫下游神經網路的 Attention 在該點「物理性失明」，只能依賴 Reason Embedding 學習休市語意。

維度重構約定: 回傳的字典中，所有 torch.Tensor 必須轉換為 PyTorch 時序標準形狀 (Batch=1, Channels, Seq_Len)（即必須包含 unsqueeze(0) 與 transpose(1, 2) 操作）。

4. 模型工廠與靜態編碼

4.1 靜態元數據編碼器 (metadata_embedder.py)

類別名稱: 必須精確命名為 MetadataEmbedder。繼承 torch.nn.Module。

廣播擴展紅線: forward(self, asset_idx: torch.Tensor, seq_len: int) 內部必須對 Embedding 產出的靜態向量執行 unsqueeze(-1).expand(-1, -1, seq_len)，使其形狀從 (Batch, Dim) 變換為 (Batch, Dim, Seq_Len)，以確保能與時間序列特徵沿通道維度 cat 拼接。

4.2 模型路由器 (model_factory.py)

類別名稱: BaseRepresentationModel (抽象基底) 與 ModelFactory。

統一介面絕對紅線: 所有模型家族 (LSTM/TCN/PatchTST/Mamba) 的 forward 簽名必須且只能精確包含以下 5 個 Keyword Arguments：
(self, value: torch.Tensor, mask: torch.Tensor, reason: torch.Tensor, time_delta: torch.Tensor, metadata_emb: torch.Tensor) -> torch.Tensor

5. 核心特徵萃取網路 (Representation Families)

5.1 基準對照組 (lstm_baseline.py)

類別名稱: LSTMEncoder。

不透視紅線: 建立 nn.LSTM 時，嚴禁設置 bidirectional=True，防止回測時偷看未來資訊。

降維約定: 必須僅提取並回傳最後一個時間步的隱藏狀態：output[:, -1, :]。不處理 time_delta。

5.2 時間感知因果網路 (tcn_encoder.py)

類別名稱: CausalConv1d, TimeAwareResidualBlock, TCNEncoder。

因果卷積紅線: CausalConv1d 必須僅對序列進行左側 (歷史) padding，右側 (未來) padding 必須為 0。

🚨 休市凍結紅線 (Time-aware Gate):

在 TimeAwareResidualBlock.forward 中，必須透過 torch.sigmoid 處理 time_delta 形成時間感知閘門。

此閘門特徵必須在進入 Activate Function (如 GELU) 之前，直接與卷積輸出相乘 (conv_out * time_gate)。休市時 $d\tau \to 0$，閘門關閉，達成特徵與梯度的絕對凍結。

5.3 輕量時間序列 Transformer (patchtst_encoder.py)

類別名稱: PatchTSTEncoder。

🚨 注意力遮罩邏輯反轉 (Attention Mask Rule):

必須將傳入的 mask 進行布林反轉 ~mask。

該反轉結果必須作為 src_key_padding_mask 傳入 nn.TransformerEncoder，將「休市點」標記為「True (忽略)」，強迫矩陣在注意力運算時剔除假數據雜訊。

5.4 狀態空間模型 (mamba_encoder.py)

類別名稱: MambaEncoder。

輸入約定: 預期輸入的 Seq_Len 為 1 ($O(1)$ 複雜度)。

🚨 步長參數物理對齊 ($\Delta$ Mapping):

必須將 time_delta 透過映射層轉換為 SSM 的步長參數 (dt)。

物理意義: 保證當 $d\tau \to 0$ (市場休市) 時，步長矩陣退化，Mamba 內部隱藏狀態達到嚴格等式 $h_t = h_{t-1}$ 的物理凍結。

6. 權重與狀態管理器 (Weights Checkpoint Manager)

6.1 斷電防毀損儲存器 (checkpoint_mgr.py)

系統定位: 安全地儲存與載入 Layer 2 內多世代 Representation Family 的權重，確保實盤重啟的絕對穩定性。

允許依賴: torch, os, typing (Dict, Any)。

類別名稱: 必須精確命名為 CheckpointManager。

⚙️ 實例綁定與目錄約定

簽名: __init__(self, save_dir: str = "phoenix_checkpoints/layer2")

建立防呆: 必須在初始化時嚴格執行 os.makedirs(self.save_dir, exist_ok=True)，防範目錄不存在造成的 I/O 崩潰。

💾 安全儲存約定 (Save Method)

簽名: save(self, model: torch.nn.Module, model_family: str, epoch: int, metrics: Dict[str, float])

結構化字典約定: 儲存的內容必須是一個 Dictionary，且強制包含以下 Keys：'epoch', 'model_state_dict', 'metrics', 'model_family'。

🚨 原子寫入絕對紅線 (Atomic Write Guard):

嚴禁直接將權重寫入最終的目標檔名 (如 .pt)。

執行順序: 必須先呼叫 torch.save 將檔案寫入帶有 .tmp 後綴的暫存檔中，寫入完成後，再使用 os.replace(temp_filepath, filepath) 覆蓋原檔案。

物理意義: 避免在寫入 Disk 的幾十毫秒內發生斷電、OOM 或 Kernel Panic，導致原本好好的模型權重變成 0KB 的損壞檔案 (Corrupted File)。

🔄 嚴格載入約定 (Load Method)

簽名: load(self, model: torch.nn.Module, filepath: str) -> Dict[str, Any]

防爆攔截: 第一行必須檢查檔案是否存在 (os.path.exists)，若無則強制拋出 FileNotFoundError。

設備隔離紅線: 呼叫 torch.load 時，必須強制綁定參數 map_location=torch.device('cpu')。這確保了模型在被分配到特定 GPU 前，不會因為 VRAM 碎片化或設備 ID 錯亂而導致記憶體溢出 (OOM)。

嚴格權重對齊紅線: 呼叫 model.load_state_dict 時，必須鎖死參數 strict=True。若神經網路架構與載入的權重在任何一層的維度或命名上出現錯位，必須讓系統直接崩潰拋錯，嚴禁系統帶著錯誤的權重進入實盤交易。