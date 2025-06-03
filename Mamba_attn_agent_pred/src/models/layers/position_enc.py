import torch
import math
import torch.nn as nn
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, ...)
        Returns:
            x with positional encoding added
        """
        T = x.size(1)
        pos_enc = self.pe[:T].unsqueeze(0)  # (1, T, D)
        
        # Adapt dimensions for broadcasting
        if x.dim() == 4:  # (B, T, N, D)
            pos_enc = pos_enc.unsqueeze(2)  # (1, T, 1, D)
            
        return x + pos_enc