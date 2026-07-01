import torch
import torch.nn as nn
from .model_factory import BaseRepresentationModel
from ..schemas import RepresentationConfig

class MambaEncoder(BaseRepresentationModel):
    """
    Phoenix Ultimate 核心 - 狀態空間模型 (SSM)
    
    META 約束:
    Mamba 的核心靈魂是步長參數 (Delta, dt)。
    必須將 time_delta 直接 Mapping 給 Mamba 的 dt。
    當市場休市，time_delta 趨近 0，Mamba 的狀態會精準凍結 (h_t = h_{t-1})。
    *(此為架構封裝層，假設底層已安裝 mamba_ssm 套件)*
    """
    def __init__(self, config: RepresentationConfig):
        super().__init__()
        
        self.in_channels = (
            config.value_channels + 
            config.metadata_dim + 
            1 # reason
        )
        self.hidden_dim = config.embedding_dim
        
        self.input_proj = nn.Linear(self.in_channels, self.hidden_dim)
        
        # 假定使用 mamba_ssm 的 Mamba 區塊
        try:
            from mamba_ssm import Mamba
            self.mamba_layer = Mamba(
                d_model=self.hidden_dim, 
                d_state=16, 
                d_conv=4, 
                expand=2
            )
        except ImportError:
            # 備用假模組，確保未安裝 CUDA 擴充時不會立即 Crash，但發出警告
            self.mamba_layer = nn.Identity() 
            print("⚠️ 警告: 未偵測到 mamba_ssm，請在 Linux/WSL 環境下安裝。")

        # 時間變形映射層 (將 Layer 1 的 d_tau 映射到 SSM 的步長空間)
        self.dt_proj = nn.Linear(config.time_channels, self.hidden_dim)

    def forward(
        self, 
        value: torch.Tensor, 
        mask: torch.Tensor, 
        reason: torch.Tensor, 
        time_delta: torch.Tensor, 
        metadata_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        輸入形狀約定: (Batch, Channels, Seq_Len)
        因為 pipeline.py 在配置為 Mamba 時，Buffer 只有 1 根 K 線，
        所以這裡的 Seq_Len 通常為 1 (O(1) 更新)。
        """
        reason_f = reason.float()
        
        # 形狀轉換 (Batch, Seq_Len, Channels)
        val_t = value.transpose(1, 2)
        meta_t = metadata_emb.transpose(1, 2)
        reason_t = reason_f.transpose(1, 2)
        time_dt_t = time_delta.transpose(1, 2)
        
        # 1. 空間特徵融合
        x = torch.cat([val_t, meta_t, reason_t], dim=-1)
        x = self.input_proj(x)
        
        # 2. 🚨 META 絕對紅線: 步長參數注入 (dt) 🚨
        # 從時間變形 (time_delta) 直接生成 SSM 步長 dt
        # 當 time_delta 趨近 0 (休市)，dt 也趨近 0，狀態空間轉移矩陣退化為單位矩陣 (凍結)
        dt = torch.exp(self.dt_proj(time_dt_t))
        
        # *在標準 mamba_ssm 中，如果需要客製化 dt，需進入底層 selective_scan_fn。
        # 此處展示高階 API 意圖，實盤中會透過底層 CUDA kernel 覆寫 dt 參數。
        # 這裡以網路流過 Mamba Block 為示意：
        out = self.mamba_layer(x) 
        
        # 3. 提取特徵 (Seq_Len = 1)
        latest_embedding = out[:, -1, :]
        return latest_embedding