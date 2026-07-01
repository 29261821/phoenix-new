import torch
import torch.nn as nn
from .model_factory import BaseRepresentationModel
from ..schemas import RepresentationConfig

class CausalConv1d(nn.Conv1d):
    """
    因果卷積層 (Causal Convolution)
    
    權責: 
    透過非對稱的 Padding 確保時間步 T 只能看到 <= T 的資料。
    徹底消滅傳統 Conv1d 會「向右看」導致的前視偏差 (Look-ahead bias)。
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        self.left_padding = (kernel_size - 1) * dilation
        # 初始化底層 Conv1d，但在呼叫時手動加上左側 padding
        super().__init__(
            in_channels, out_channels, kernel_size, 
            padding=0, dilation=dilation, **kwargs
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Seq_Len)
        # 在序列的左側 (過去) 補零，右側 (未來) 不補
        x_padded = torch.nn.functional.pad(x, (self.left_padding, 0))
        return super().forward(x_padded)

class TimeAwareResidualBlock(nn.Module):
    """
    時間感知殘差塊 (Time-Aware Residual Block)
    
    META 約束:
    必須在 Activate Function 之前，將輸入矩陣乘上由 time_delta 轉換而來的時間感知閘門。
    物理意義: 休市時 d_tau 趨近 0，閘門關閉，狀態凍結；劇震時 d_tau 放大，賦予極高激活權重。
    """
    def __init__(self, channels: int, kernel_size: int, dilation: int, time_channels: int):
        super().__init__()
        
        # 1. 空間卷積 (Causal)
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        
        # 2. 時間閘門映射 (將 d_tau 映射到與特徵相同的維度)
        self.time_gate_proj = nn.Conv1d(time_channels, channels, kernel_size=1)
        
        # 3. 激活與正規化
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, time_delta: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Channels, Seq_Len)
        
        # --- 卷積提取空間特徵 ---
        conv_out = self.conv(x)
        
        # --- 時間變形注入 (Time Deformation Injection) ---
        # 產生時間感知閘門 (Time-aware Gate)，使用 Sigmoid 將其壓在 0~1 之間
        # time_delta shape: (Batch, time_channels, Seq_Len)
        time_gate = torch.sigmoid(self.time_gate_proj(time_delta))
        
        # 🚨 META 絕對紅線: 必須在 Activation 之前乘上閘門 🚨
        gated_out = conv_out * time_gate
        
        # --- 激活與殘差連接 ---
        activated = self.activation(gated_out)
        activated = self.dropout(activated)
        
        # Residual Connection
        res = x + activated
        
        # LayerNorm 通常在 Channel 維度做，需轉換形狀
        res_transposed = res.transpose(1, 2)
        normed = self.layer_norm(res_transposed).transpose(1, 2)
        
        return normed

class TCNEncoder(BaseRepresentationModel):
    """
    Phoenix Lite 核心 - 時間感知因果網路 (Time-aware Causal Network)
    """
    def __init__(self, config: RepresentationConfig):
        super().__init__()
        
        # 1. 計算輸入通道總數 (排除 time_delta，因為它有專屬的閘門通道)
        in_channels = (
            config.value_channels + 
            config.metadata_dim + 
            1 + # mask
            1   # reason
        )
        self.hidden_dim = config.embedding_dim
        
        # 2. 特徵升維投影 (Input Projection)
        self.input_proj = nn.Conv1d(in_channels, self.hidden_dim, kernel_size=1)
        
        # 3. 建立 TCN 殘差塊堆疊 (使用指數增長的 Dilation 擴大感受野)
        self.num_layers = 4
        self.blocks = nn.ModuleList([
            TimeAwareResidualBlock(
                channels=self.hidden_dim, 
                kernel_size=3, 
                dilation=2**i,  # 1, 2, 4, 8...
                time_channels=config.time_channels
            )
            for i in range(self.num_layers)
        ])

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
        """
        mask_f = mask.float()
        reason_f = reason.float()
        
        # 1. 初始特徵拼接 (沿著 Channel 維度)
        x = torch.cat([value, metadata_emb, mask_f, reason_f], dim=1)
        
        # 2. 映射至隱藏維度
        x = self.input_proj(x)
        
        # 3. 通過時間感知 TCN 區塊
        for block in self.blocks:
            x = block(x, time_delta)
            
        # 4. O(1) 空間約束: 提取時間軸上的最後一個表徵向量
        # x shape: (Batch, hidden_dim, Seq_Len) -> 取 Seq_Len 的最後一格
        latest_embedding = x[:, :, -1]
        
        # 回傳 Shape: (Batch, embedding_dim)
        return latest_embedding