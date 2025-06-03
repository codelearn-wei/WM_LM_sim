from models.modules.agent_encoder import AttentionSpatialInteraction
from models.modules.history_encoder import MambaTemporalEncoder
from models.modules.map_encoder import MapEncoder
from models.layers.position_enc import PositionalEncoding
from models.layers.embeding import Embedding
import torch.nn as nn

class TrajectoryEncoder(nn.Module):
    def __init__(self, in_features, d_model, map_feature_dim=None, use_map=True, agent_type_num=None, use_agent_type=True, num_attn_layers=2):
        """
        Args:
            in_features: 原始输入特征数
            d_model: 模型维度
            map_feature_dim: 地图特征维度
            use_map: 是否使用地图信息
            agent_type_num: agent类型数量
            use_agent_type: 是否使用agent type
            num_attn_layers: 注意力层数
        """
        super().__init__()
        
        self.use_map = use_map
        
        self.embed = Embedding(
            in_features=in_features,
            d_model=d_model,
            agent_type_num=agent_type_num,
            use_agent_type=use_agent_type
        )
        self.pos_enc = PositionalEncoding(d_model)
        self.temporal_encoder = MambaTemporalEncoder(d_model)
        
        # 堆叠多层注意力模块
        self.attn_layers = nn.ModuleList([
            AttentionSpatialInteraction(d_model) for _ in range(num_attn_layers)
        ])
        
        # 地图编码器
        if use_map and map_feature_dim is not None:
            self.map_encoder = MapEncoder(
                map_feature_dim=map_feature_dim,
                d_model=d_model
            )
    
    def forward(self, x, map_features=None):
        """
        Args:
            x: (B, T, N, F) - 轨迹输入
            map_features: (B, N_map, map_feature_dim) - 地图特征
        Returns:
            out: (B, T, N, D) - 编码后特征
        """
        x = self.embed(x)  # (B, T, N, D)
        x = self.pos_enc(x)  # 添加位置编码
        x = self.temporal_encoder(x)  # Mamba时序建模
        
        # 应用多层注意力
        for attn_layer in self.attn_layers:
            x = attn_layer(x)  # 纯注意力交互建模
        
        # 如果使用地图信息，进行地图特征融合
        if self.use_map and map_features is not None:
            x = self.map_encoder(map_features, x)
            
        return x