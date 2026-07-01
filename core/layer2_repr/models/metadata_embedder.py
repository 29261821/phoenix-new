import torch
import torch.nn as nn

class MetadataEmbedder(nn.Module):
    """
    資產元數據靜態編碼器 (Metadata Embedder)
    
    權責: 
    將類別型變數 (Categorical Variables) 如資產類別、時區等，
    轉化為連續的靜態 Embedding 向量。
    """
    def __init__(self, num_assets: int, metadata_dim: int):
        super().__init__()
        # 使用標準 Embedding 層處理資產 ID (例如 0: BTC, 1: SPY, 2: TSLA)
        self.asset_embedding = nn.Embedding(
            num_embeddings=num_assets, 
            embedding_dim=metadata_dim
        )
        
        # 可擴充：如果未來有 region (美股/亞股/加密貨幣) 或 asset_class (股票/外匯/Crypto)
        # self.region_embedding = nn.Embedding(...)

    def forward(self, asset_idx: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        輸入: 
            asset_idx: 形狀為 (Batch,) 的資產索引 Tensor
            seq_len: 當前時間序列的長度 (為了對齊後續的 Broadcasting)
        輸出:
            形狀為 (Batch, metadata_dim, Seq_Len) 的擴展張量
        """
        # 1. 取得靜態向量: Shape (Batch, metadata_dim)
        static_emb = self.asset_embedding(asset_idx)
        
        # 2. Broadcasting (廣播擴展)
        # 為了能與時間序列特徵拼接，必須將靜態特徵沿著時間軸 (Seq_Len) 複製
        # 變換形狀: (Batch, metadata_dim, 1) -> (Batch, metadata_dim, Seq_Len)
        expanded_emb = static_emb.unsqueeze(-1).expand(-1, -1, seq_len)
        
        return expanded_emb