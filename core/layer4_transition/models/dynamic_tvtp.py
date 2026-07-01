# phoenix/core/layer4_transition/models/dynamic_tvtp.py

import torch
import torch.nn as nn

class DynamicTVTP(nn.Module):
    """
    動態時變轉移網路 (Time-Varying Transition Probability Network)
    
    系統定位: SOTA 核心 (打破靜態馬可夫假設)
    權責: 接收 128 維市場潛在空間特徵，動態生成下一刻的狀態轉移矩陣。
    """
    
    def __init__(self, input_dim: int = 128, n_components: int = 4, hidden_dim: int = 64):
        """
        Args:
            input_dim (int): 嚴格對齊 Layer 2 的 embedding_dim (預設 128)
            n_components (int): 嚴格對齊 Layer 3 的狀態數量
            hidden_dim (int): 輕量級 MLP 的隱藏層維度
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_components = n_components
        
        # 輕量級 2 層 MLP (捕捉 128 維特徵的非線性映射)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, n_components * n_components)
        )

        # 權重初始化：確保初始轉移矩陣傾向於對角線 (維持狀態慣性)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        防爆初始化策略：
        我們希望在未經訓練前，網路輸出的預設行為接近「靜態單位矩陣」或「高對角線機率」。
        這能避免模型一開局就瘋狂切換狀態 (Ping-Pong Effect)。
        """
        final_layer = self.net[-1]
        nn.init.xavier_uniform_(final_layer.weight, gain=0.1)
        
        # 偏置項 (Bias) 初始化：給對角線元素較高的初始權重
        bias_matrix = torch.zeros(self.n_components, self.n_components)
        bias_matrix.fill_diagonal_(2.0) # 給予對角線較高初始 Logit
        final_layer.bias.data = bias_matrix.view(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        因果限制: 絕對只能依賴 X_t (時間 T 的表徵)
        
        Args:
            x (torch.Tensor): RepresentationVector, Shape: [Batch, 128]
            
        Returns:
            torch.Tensor: 動態轉移矩陣 T_t, Shape: [Batch, N, N]
        """
        batch_size = x.size(0)
        
        # 輸出 Shape: [Batch, N * N]
        logits = self.net(x)
        
        # 重塑為矩陣 Shape: [Batch, N, N]
        logits_reshaped = logits.view(batch_size, self.n_components, self.n_components)
        
        # 數學紅線 (Softmax 約束): 
        # 轉移矩陣 T_t 其每一列 (Row) 的總和必須嚴格等於 1.0。
        # dim=-1 代表對最後一個維度 (也就是每一列) 進行 Softmax 運算。
        transition_matrix_t = torch.softmax(logits_reshaped, dim=-1)
        
        return transition_matrix_t