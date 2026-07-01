# phoenix/core/layer4_transition/models/identity_freezer.py

import torch
import numpy as np
from typing import Union

class IdentityFreezer:
    """
    休市單位矩陣映射器 (Identity Matrix Mapper)
    
    系統定位: 跨市場防呆紅線
    權責: 在資產休市或數據斷線時，徹底凍結狀態的演化。
    """
    
    @staticmethod
    def get_identity_matrix(n_components: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        產出標準單位矩陣 I。
        數學意義: P(S_{t+1}) = P(S_t) * I = P(S_t)
        """
        return torch.eye(n_components, dtype=torch.float32, device=device)

    @staticmethod
    def apply_vectorized_freeze(
        is_frozen_mask: torch.Tensor, 
        tvtp_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch 模式的向量化條件遮罩 (Vectorized Conditional Masking)
        
        為榨乾 CUDA 效能，避免使用 Python for 迴圈。
        透過 torch.where 將休市時間點的轉移矩陣一次性替換為單位矩陣 (I)。
        
        Args:
            is_frozen_mask: Boolean Tensor, Shape: (Batch,)
            tvtp_output: TVTP 網路輸出的動態轉移矩陣, Shape: (Batch, N, N)
            
        Returns:
            torch.Tensor: 防爆處理後的轉移矩陣, Shape: (Batch, N, N)
        """
        batch_size, n_components, _ = tvtp_output.shape
        device = tvtp_output.device
        
        # 生成與 Batch 對齊的單位矩陣 (Batch, N, N)
        identity_batch = torch.eye(n_components, dtype=torch.float32, device=device)
        identity_batch = identity_batch.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 將 mask 擴張至 (Batch, 1, 1) 以進行矩陣級別的 Broadcasting
        mask_expanded = is_frozen_mask.unsqueeze(-1).unsqueeze(-1)
        
        # 執行遮罩：True (休市) 給予 I，False (正常) 保留 TVTP 輸出
        safe_transition_matrix = torch.where(mask_expanded, identity_batch, tvtp_output)
        
        return safe_transition_matrix

    @staticmethod
    def apply_streaming_freeze(is_frozen: bool, transition_matrix: np.ndarray) -> np.ndarray:
        """
        Streaming 實盤模式的單步凍結防禦。
        """
        if is_frozen:
            n_components = transition_matrix.shape[0]
            return np.eye(n_components, dtype=np.float64)
        return transition_matrix