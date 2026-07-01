import torch
import torch.nn as nn

class StaticTransitionModel(nn.Module):
    """
    靜態馬可夫轉移矩陣 (Static Markov Transition Matrix)
    
    系統定位: 基準模型 (Baseline) 與降級防禦機制
    權責: 提供非時變的固定狀態轉移機率矩陣，作為 TVTP SOTA 網路的對照組。
    """
    
    def __init__(self, n_components: int = 4):
        """
        Args:
            n_components (int): 嚴格對齊 Layer 3 的狀態數量
        """
        super().__init__()
        self.n_components = n_components
        
        # 數學紅線: 矩陣初始化策略
        # 我們不直接儲存機率，而是儲存 Logits。這保證了在神經網路的計算圖中，
        # 經過 Softmax 轉換後的矩陣絕對是一個合法的機率分佈 (Row sum = 1.0)。
        
        # 初始化傾向於對角線 (狀態慣性)，符合真實市場「狀態具有延續性」的物理現象
        init_logits = torch.zeros(n_components, n_components)
        init_logits.fill_diagonal_(2.0)
        
        # 註冊為可學習參數 (Learnable Parameter)
        # 允許在離線訓練階段透過最大概似估計 (MLE) 或交叉熵損失被優化器更新
        self.transition_logits = nn.Parameter(init_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        靜態轉移矩陣的推論。
        
        介面約束: 為了與 dynamic_tvtp.py 保持絕對的介面對齊，
        雖然本模型不依賴輸入特徵 x 進行動態運算，但仍必須接收 x，
        並依據其 Batch Size 進行張量廣播 (Broadcast)。
        
        Args:
            x (torch.Tensor): RepresentationVector, Shape: [Batch, 128]
            
        Returns:
            torch.Tensor: 靜態轉移矩陣 T_t, Shape: [Batch, N, N]
        """
        batch_size = x.size(0)
        
        # 核心公式約束: 確保每一列總和為 1.0
        # static_matrix Shape: [N, N]
        static_matrix = torch.softmax(self.transition_logits, dim=-1)
        
        # 向量化廣播 (Vectorized Broadcasting)
        # 為了榨乾 GPU 效能，嚴禁使用 for 迴圈複製矩陣。
        # 透過 unsqueeze 與 expand 將 [N, N] 擴張為 [Batch, N, N]
        transition_matrix_t = static_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        return transition_matrix_t