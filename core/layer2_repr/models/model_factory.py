import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from ..schemas import RepresentationConfig

class BaseRepresentationModel(nn.Module, ABC):
    """
    Layer 2 統一模型基底合約
    強制規定所有的 Encoder (LSTM, TCN, Mamba) 必須實作這個 forward 簽名。
    """
    @abstractmethod
    def forward(
        self, 
        value: torch.Tensor, 
        mask: torch.Tensor, 
        reason: torch.Tensor, 
        time_delta: torch.Tensor,
        metadata_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        輸出必須永遠是 Shape: (Batch, embedding_dim) 的 Latent Vector
        """
        pass

class ModelFactory:
    """
    模型路由器 (Model Factory)
    根據 Config 決定載入哪個具體的網路架構。
    """
    @staticmethod
    def build(config: RepresentationConfig) -> BaseRepresentationModel:
        if config.model_type == "lstm":
            from .lstm_baseline import LSTMEncoder
            return LSTMEncoder(config)
        
        # 未來其他複雜模型將在此處註冊
        elif config.model_type == "tcn":
            # from .tcn_encoder import TCNEncoder
            # return TCNEncoder(config)
            raise NotImplementedError("TCN 架構即將實作")
            
        elif config.model_type == "mamba":
            # from .mamba_encoder import MambaEncoder
            # return MambaEncoder(config)
            raise NotImplementedError("Mamba 架構即將實作")
            
        else:
            raise ValueError(f"不支援的模型類型: {config.model_type}")