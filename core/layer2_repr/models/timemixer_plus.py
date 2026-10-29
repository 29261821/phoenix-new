import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_factory import BaseRepresentationModel
from ..schemas import RepresentationConfig


class ChannelMixing(nn.Module):
    """
    最粗尺度通道混合 (Channel Mixing at Coarsest Scale)
    依據 TimeMixer++ 論文 Eq. (2)：在最粗尺度 x_M 上進行 Variate-wise Self-Attention。
    """
    def __init__(self, num_channels: int, d_model: int):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形狀: (B, C, d_model)
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_model) # (B, C, C)
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, V) # (B, C, d_model)
        return self.norm(x + self.out_proj(out))


class DualAxisAttentionTID(nn.Module):
    """
    雙軸注意力時間影像解耦 (Dual-Axis Attention Time Image Decomposition, TID)
    依據 TimeMixer++ 論文 Eq. (7)：
    - Column-axis Attention: 捕捉週期內「季節性/波動模式」(Seasonality)
    - Row-axis Attention: 捕捉跨週期「趨勢模式」(Trend)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # 2D 卷積計算 Query, Key, Value
        self.conv_col = nn.Conv2d(d_model, d_model * 3, kernel_size=1)
        self.conv_row = nn.Conv2d(d_model, d_model * 3, kernel_size=1)
        
        self.proj_col = nn.Linear(d_model, d_model)
        self.proj_row = nn.Linear(d_model, d_model)

    def forward(self, z: torch.Tensor):
        # z: (B, d_model, P, F) - 2D Time Image
        B, D, P, F_dim = z.shape
        
        # 1. Column-axis Attention (Seasonality 週期內模式)
        # 目標: 對 F_dim 維度進行注意力，將 P 維度視為 Batch 空間
        z_col = self.conv_col(z) # (B, 3*D, P, F_dim)
        q_c, k_c, v_c = torch.chunk(z_col, 3, dim=1)
        # 轉置為 (B*P, F_dim, D)
        q_c = q_c.permute(0, 2, 3, 1).reshape(B * P, F_dim, D)
        k_c = k_c.permute(0, 2, 3, 1).reshape(B * P, F_dim, D)
        v_c = v_c.permute(0, 2, 3, 1).reshape(B * P, F_dim, D)
        
        scores_col = torch.bmm(q_c, k_c.transpose(1, 2)) / math.sqrt(D)
        attn_col = F.softmax(scores_col, dim=-1)
        seasonal_flat = torch.bmm(attn_col, v_c) # (B*P, F_dim, D)
        seasonal = seasonal_flat.reshape(B, P, F_dim, D).permute(0, 3, 1, 2) # (B, D, P, F_dim)
        
        # 2. Row-axis Attention (Trend 跨週期總體趨勢)
        # 目標: 對 P 維度進行注意力，將 F_dim 維度視為 Batch 空間
        z_row = self.conv_row(z) # (B, 3*D, P, F_dim)
        q_r, k_r, v_r = torch.chunk(z_row, 3, dim=1)
        # 轉置為 (B*F_dim, P, D)
        q_r = q_r.permute(0, 3, 2, 1).reshape(B * F_dim, P, D)
        k_r = k_r.permute(0, 3, 2, 1).reshape(B * F_dim, P, D)
        v_r = v_r.permute(0, 3, 2, 1).reshape(B * F_dim, P, D)
        
        scores_row = torch.bmm(q_r, k_r.transpose(1, 2)) / math.sqrt(D)
        attn_row = F.softmax(scores_row, dim=-1)
        trend_flat = torch.bmm(attn_row, v_r) # (B*F_dim, P, D)
        trend = trend_flat.reshape(B, F_dim, P, D).permute(0, 3, 2, 1) # (B, D, P, F_dim)
        
        return seasonal, trend


class TimeMixerPlusBlock(nn.Module):
    """
    TimeMixer++ 單個 MixerBlock
    整合：
    1. Multi-Resolution Time Imaging (MRTI, FFT 萃取週期)
    2. Time Image Decomposition (TID, 雙軸注意力解構)
    3. Multi-Scale Mixing (MCM, 季節自下而上，趨勢自上而下)
    4. Multi-Resolution Mixing (MRM, 振幅頻率加權)
    """
    def __init__(self, d_model: int, num_scales: int = 3, top_k: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        self.top_k = top_k
        
        self.tid = DualAxisAttentionTID(d_model)
        
        # Bottom-up 季節項混合 (2D-Conv)
        self.seasonal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                nn.GELU()
            ) for _ in range(num_scales - 1)
        ])
        
        # Top-down 趨勢項混合 (2D-TransConv)
        self.trend_transconvs = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(d_model, d_model, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
                nn.GELU()
            ) for _ in range(num_scales - 1)
        ])

    def forward(self, scale_features: list):
        # scale_features: [x_0, x_1, ..., x_M], 每個為 (B, d_model, T_m)
        coarsest = scale_features[-1] # (B, d_model, T_M)
        B, D, T_coarse = coarsest.shape
        
        # 1. FFT 在最粗尺度獲取 Top-K 顯著頻率與週期
        # 計算實數 FFT 振幅
        fft_vals = torch.fft.rfft(coarsest, dim=-1).abs().mean(dim=[0, 1]) # (Freq_bins,)
        fft_vals[0] = 0 # 忽略直流分量 DC (頻率為 0)
        
        k = min(self.top_k, max(1, len(fft_vals) - 1))
        topk_amplitudes, topk_indices = torch.topk(fft_vals, k=k)
        
        periods = []
        for idx in topk_indices:
            p = max(2, int(T_coarse / max(idx.item(), 1)))
            periods.append(p)
        
        # 振幅加權歸一化
        weights = F.softmax(topk_amplitudes.float(), dim=-1) # (k,)
        
        # 2. 對每個 Period 進行 Multi-Resolution Time Imaging & TID
        mixed_scales_by_period = []
        for p_idx, p in enumerate(periods):
            seasonal_images = []
            trend_images = []
            pad_lens = []
            
            for m in range(self.num_scales):
                xm = scale_features[m] # (B, D, T_m)
                Tm = xm.shape[-1]
                F_dim = math.ceil(Tm / p)
                pad_target = p * F_dim
                pad_len = pad_target - Tm
                pad_lens.append((pad_len, Tm))
                
                if pad_len > 0:
                    xm_padded = F.pad(xm, (0, pad_len))
                else:
                    xm_padded = xm
                
                # 轉為 2D Time Image: (B, D, P, F_dim)
                zm = xm_padded.view(B, D, p, F_dim)
                s_m, t_m = self.tid(zm)
                seasonal_images.append(s_m)
                trend_images.append(t_m)
            
            # 3. Multi-Scale Mixing
            # (A) Seasonal: Bottom-Up (由細到粗 m: 0 -> M-1)
            for m in range(self.num_scales - 1):
                s_next = self.seasonal_convs[m](seasonal_images[m])
                # 對齊維度
                if s_next.shape[-1] != seasonal_images[m + 1].shape[-1]:
                    s_next = F.interpolate(s_next, size=seasonal_images[m + 1].shape[2:], mode='nearest')
                seasonal_images[m + 1] = seasonal_images[m + 1] + s_next

            # (B) Trend: Top-Down (由粗到細 m: M-1 -> 0)
            for m in range(self.num_scales - 1, 0, -1):
                t_prev = self.trend_transconvs[m - 1](trend_images[m])
                if t_prev.shape[-1] != trend_images[m - 1].shape[-1]:
                    t_prev = F.interpolate(t_prev, size=trend_images[m - 1].shape[2:], mode='nearest')
                trend_images[m - 1] = trend_images[m - 1] + t_prev

            # 4. 聚合季節與趨勢，並由 2D 還原回 1D
            mixed_1d_scales = []
            for m in range(self.num_scales):
                combined_2d = seasonal_images[m] + trend_images[m] # (B, D, P, F_dim)
                pad_len, orig_len = pad_lens[m]
                unfolded = combined_2d.reshape(B, D, -1)
                reconstructed = unfolded[:, :, :orig_len]
                mixed_1d_scales.append(reconstructed)

            mixed_scales_by_period.append((weights[p_idx], mixed_1d_scales))

        # 5. Multi-Resolution Mixing (加權融合各個 Period)
        out_scales = []
        for m in range(self.num_scales):
            m_res = torch.zeros_like(scale_features[m])
            for w, period_scales in mixed_scales_by_period:
                m_res = m_res + w * period_scales[m]
            # 殘差連接與 LayerNorm
            m_res = scale_features[m] + m_res
            out_scales.append(m_res)

        return out_scales


class TimeMixerPlusEncoder(BaseRepresentationModel):
    """
    TimeMixer++ 時序表徵學習編碼器 (Phoenix Core Layer 2 SOTA)
    
    技術特性：
    1. 最粗尺度變數注意力 (Coarsest-scale Channel Mixing)
    2. 多尺度下採樣 (Multi-scale Downsampling)
    3. 頻域多解析度時序影像化 (MRTI)
    4. 雙軸注意力季節-趨勢解構 (TID)
    5. 非對稱多尺度混合 (MCM)
    6. 尺度自適應 Ensemble 預測頭
    """
    def __init__(self, config: RepresentationConfig, num_scales: int = 3, top_k: int = 3, num_layers: int = 2):
        super().__init__()
        self.config = config
        self.num_scales = num_scales
        self.d_model = config.embedding_dim

        self.in_channels = (
            config.value_channels + 
            config.metadata_dim + 
            config.time_channels +
            1 # reason
        )

        # 1. 深度特徵映射
        self.input_embed = nn.Conv1d(self.in_channels, self.d_model, kernel_size=1)
        
        # 2. 最粗尺度變數混合
        self.channel_mixing = ChannelMixing(self.in_channels, self.d_model)

        # 3. 多尺度下採樣器 (stride=2 conv)
        self.downsamplers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=2, stride=2) for _ in range(num_scales - 1)
        ])

        # 4. 堆疊 MixerBlocks
        self.blocks = nn.ModuleList([
            TimeMixerPlusBlock(self.d_model, num_scales=num_scales, top_k=top_k)
            for _ in range(num_layers)
        ])

        # 5. 各尺度表徵頭與最後 Ensemble 投影
        self.scale_heads = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model) for _ in range(num_scales)
        ])
        
        self.out_norm = nn.LayerNorm(self.d_model)

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
        
        # 1. 特徵組合 (拼接)
        # Shape: (Batch, Total_Channels, Seq_Len)
        x = torch.cat([value, metadata_emb, time_delta, reason_f], dim=1)
        
        # 2. 應用 Mask (將異常/休市訊號置零防滲透)
        x = x * mask.float()
        
        # 3. 映射到 d_model
        x_0 = self.input_embed(x) # (B, d_model, T_0)
        
        # 4. 多尺度下採樣產生 Xinit = [x_0, x_1, ..., x_M]
        scales = [x_0]
        cur = x_0
        for down in self.downsamplers:
            cur = down(cur)
            scales.append(cur)
            
        # 5. 在最粗尺度進行 Channel Mixing (Variate-wise Attention)
        # 最粗尺度特徵: (B, d_model, T_M) -> 轉為 (B, T_M, d_model)
        coarse_in = scales[-1].transpose(1, 2)
        coarse_mixed = self.channel_mixing(coarse_in)
        scales[-1] = coarse_mixed.transpose(1, 2)

        # 6. 通過堆疊的 TimeMixer++ MixerBlocks
        for block in self.blocks:
            scales = block(scales)

        # 7. 各尺度提取最後時間步 (Latest Frame) 並進行 Ensemble
        scale_embeddings = []
        for m, head in enumerate(self.scale_heads):
            # 取各尺度最後一幀: (B, d_model)
            latest_m = scales[m][:, :, -1]
            scale_embeddings.append(head(latest_m))

        # 均值聚合 (Ensemble)
        final_latent = torch.stack(scale_embeddings, dim=0).mean(dim=0)
        final_latent = self.out_norm(final_latent)
        
        # 嚴格符合協議: (Batch, embedding_dim)
        return final_latent
