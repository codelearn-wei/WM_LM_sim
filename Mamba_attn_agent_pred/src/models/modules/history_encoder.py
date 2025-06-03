# Mamba编码时序轨迹数据
import torch.nn as nn
from models.layers.mamba import Mamba

class MambaTemporalEncoder(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        """
        Args:
            d_model: 模型维度
            d_state: Mamba状态维度
            d_conv: 卷积维度
            expand_factor: 扩展因子
        """
        super().__init__()
        self.d_model = d_model
        d_inner = int(expand_factor * d_model)
        
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand_factor
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, N, D)
        Returns:
            out: (B, T, N, D)
        """
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)  # (B*N, T, D)
        
        # Apply Mamba block
        x = self.mamba(x)  # (B*N, T, D)
        x = self.norm(x)
        x = self.dropout(x)
        
        # Reshape back
        x = x.view(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)
        return x
