# 建模智能体之间的状态attention
import torch
import torch.nn as nn

class AttentionSpatialInteraction(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 增强版本：添加前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, N, D)
        Returns:
            out: (B, T, N, D)
        """
        B, T, N, D = x.shape
        out = []
        for t in range(T):
            x_t = x[:, t]  # (B, N, D)
            
            # 多头注意力
            attn_out, _ = self.attn(x_t, x_t, x_t)  # (B, N, D)
            attn_out = self.dropout(attn_out)
            attn_out = self.norm(attn_out + x_t)  # 残差连接 + 层归一化
            
            # 前馈网络
            ffn_out = self.ffn(attn_out)
            ffn_out = self.final_norm(ffn_out + attn_out)  # 残差连接 + 层归一化
            
            out.append(ffn_out)
            
        out = torch.stack(out, dim=1)  # (B, T, N, D)
        return out