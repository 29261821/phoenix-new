import torch
import torch.nn as nn
from .model_factory import BaseRepresentationModel
from ..schemas import RepresentationConfig

class LSTMEncoder(BaseRepresentationModel):
    """
    論文對照組 (Baseline) - 傳統 LSTM
    
    META 約束:
    最陽春的基準線。不處理時間變形 ($d\\tau$)，單純將所有 Channel 拼接後餵給 LSTM。
    """
    def __init__(self, config: RepresentationConfig):
        super().__init__()
        
        # 計算拼接後的總輸入維度
        # value_channels + metadata_dim + 1(mask) + 1(reason)
        # 依照 META，基準組「不吃 time_delta」
        self.input_dim = (
            config.value_channels + 
            config.metadata_dim + 
            1 + # mask (轉為 float)
            1   # reason (轉為 float)
        )
        
        self.hidden_dim = config.embedding_dim
        
        # 標準單向 LSTM (因為實盤不能偷看未來，嚴格禁止 bidirectional=True)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

    def forward(
        self, 
        value: torch.Tensor, 
        mask: torch.Tensor, 
        reason: torch.Tensor, 
        time_delta: torch.Tensor, 
        metadata_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        張量形狀轉換與前向傳播
        輸入字典的形狀皆為: (Batch, Channels, Seq_Len)
        """
        # 1. 確保輔助通道的資料型別為 float32
        mask_f = mask.float()
        reason_f = reason.float()
        
        # 2. 暴力拼接所有特徵 (Channel 維度拼接)
        # Shape: (Batch, Total_Channels, Seq_Len)
        combined_features = torch.cat([
            value, 
            metadata_emb, 
            mask_f, 
            reason_f
        ], dim=1)
        
        # 3. 形狀重構以符合 PyTorch LSTM 標準輸入
        # (Batch, Channels, Seq_Len) -> (Batch, Seq_Len, Channels)
        lstm_input = combined_features.transpose(1, 2)
        
        # 4. LSTM 推論
        # output shape: (Batch, Seq_Len, Hidden_Dim)
        # h_n shape: (Num_Layers, Batch, Hidden_Dim)
        output, (h_n, c_n) = self.lstm(lstm_input)
        
        # 5. O(1) 空間約束: 嚴格只提取最後一個時間步的隱藏狀態 (Last Frame)
        # 取 output 的最後一個序列位置 (Seq_Len = -1)
        # Shape: (Batch, embedding_dim)
        latest_embedding = output[:, -1, :]
        
        return latest_embedding