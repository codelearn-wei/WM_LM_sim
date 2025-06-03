import torch
import torch.nn as nn
from models.layers.position_enc import PositionalEncoding

class MapEncoder(nn.Module):
    def __init__(self, map_feature_dim, d_model, num_heads=4, dropout=0.1):
        """
        Initialize map encoder.
        
        Args:
            map_feature_dim: Dimension of input map features
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Map feature embedding
        self.map_embed = nn.Linear(map_feature_dim, d_model)
        
        # Positional encoding for map tokens
        self.pos_enc = PositionalEncoding(d_model)
        
        # Self-attention for map features
        self.map_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Cross-attention between map and trajectory features
        self.map_traj_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, map_features, traj_features):
        """
        Forward pass.
        
        Args:
            map_features: (B, N_map, map_feature_dim) - Map features
            traj_features: (B, T, N, D) - Trajectory features
            
        Returns:
            enhanced_features: (B, T, N, D) - Enhanced trajectory features
        """
        B, T, N, D = traj_features.shape
        N_map = map_features.shape[1]
        
        # Embed map features
        map_emb = self.map_embed(map_features)  # (B, N_map, D)
        map_emb = self.pos_enc(map_emb)  # Add positional encoding
        
        # Self-attention on map features
        map_attn_out, _ = self.map_self_attn(map_emb, map_emb, map_emb)
        map_emb = self.norm1(map_emb + map_attn_out)
        
        # Reshape trajectory features for cross-attention
        traj_reshaped = traj_features.reshape(B, T*N, D)
        
        # Cross-attention between map and trajectory features
        cross_attn_out, _ = self.map_traj_cross_attn(
            traj_reshaped,  # Query
            map_emb,        # Key
            map_emb         # Value
        )
        
        # Add residual connection and normalize
        enhanced = self.norm2(traj_reshaped + cross_attn_out)
        
        # Apply feed-forward network
        enhanced = enhanced + self.ffn(enhanced)
        enhanced = self.final_norm(enhanced)
        
        # Reshape back to original dimensions
        enhanced_features = enhanced.reshape(B, T, N, D)
        
        return enhanced_features 