import torch
import torch.nn as nn
from .model_factory import BaseRepresentationModel
from ..schemas import RepresentationConfig

class PatchTSTEncoder(BaseRepresentationModel):
    """
    Pro 版預留 - 輕量級時間序列 Transformer (Patch Time Series Transformer)
    
    META 約束:
    必須將 Layer 1 傳來的 mask 轉換為 Attention Mask (Key Padding Mask)，
    強迫模型在自注意力 (Self-Attention) 計算時，完全忽略休市或斷線區塊的權重。
    """
    def __init__(self, config: RepresentationConfig):
        super().__init__()
        
        self.in_channels = (
            config.value_channels + 
            config.metadata_dim + 
            config.time_channels + # PatchTST 將 time_delta 作為特徵輸入
            1 # reason
        )
        self.hidden_dim = config.embedding_dim
        
        # 特徵投影
        self.input_proj = nn.Conv1d(self.in_channels, self.hidden_dim, kernel_size=1)
        
        # 輕量級 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, 
            nhead=8, 
            dim_feedforward=self.hidden_dim * 4, 
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

    def forward(
        self, 
        value: torch.Tensor, 
        mask: torch.Tensor, 
        reason: torch.Tensor, 
        time_delta: torch.Tensor, 
        metadata_emb: torch.Tensor
    ) -> torch.Tensor:
        # 輸入形狀: (Batch, Channels, Seq_Len)
        reason_f = reason.float()
        
        # 1. 拼接特徵
        x = torch.cat([value, metadata_emb, time_delta, reason_f], dim=1)
        x = self.input_proj(x)
        
        # 轉換為 Transformer 所需形狀: (Batch, Seq_Len, hidden_dim)
        x_transposed = x.transpose(1, 2)
        
        # 2. 🚨 META 絕對紅線: Attention Mask 轉換 🚨
        # PyTorch 的 key_padding_mask 規定：True 代表「要忽略 (Ignore)」，False 代表「要參與計算」
        # 但 Layer 1 傳來的 mask 是：True=正常交易, False=休市遺失
        # 因此必須進行邏輯反轉 (Logical NOT)
        attention_mask = ~mask.squeeze(1) # Shape: (Batch, Seq_Len)
        
        # 3. 帶入 Mask 進行 Self-Attention
        # 休市期間的 K 線在此處將無法把雜訊傳遞給其他正常的 K 線
        out = self.transformer(x_transposed, src_key_padding_mask=attention_mask)
        
        # 4. 提取最後一幀
        latest_embedding = out[:, -1, :]
        return latest_embedding