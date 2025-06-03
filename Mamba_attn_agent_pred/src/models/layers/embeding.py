import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, in_features, d_model, agent_type_num=None, use_agent_type=False, dropout=0.1):
        """
        Args:
            in_features: 原始输入特征数（例如 x, y, vx, vy, heading, ...）
            d_model: 嵌入维度
            agent_type_num: agent 类型的数量（如车、人、自行车等），如果用 one-hot 或 embedding 表示
            use_agent_type: 是否使用 agent type 作为特征之一
        """
        super(Embedding, self).__init__()
        self.use_agent_type = use_agent_type
        self.base_feat_dim = in_features - 1 if use_agent_type else in_features
        
        # 基础特征投影
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 如果使用agent类型，添加类型嵌入
        if use_agent_type and agent_type_num is not None:
            self.agent_type_emb = nn.Embedding(agent_type_num, d_model)
        else:
            self.agent_type_emb = None
    
    def forward(self, x):
        """
        Args:
            x: (B, T, N, F) - Batch, Time, Num_agents, Features
        Returns:
            emb: (B, T, N, D) - Embedded features
        """
        B, T, N, F = x.shape
        
        if self.use_agent_type and self.agent_type_emb is not None:
            # 分离基础特征和agent类型
            base_feat = x[..., :-1]  # (B, T, N, F-1)
            agent_type = x[..., -1].long()  # (B, T, N)
            
            # 重塑张量以适应线性层
            base_feat = base_feat.reshape(-1, F-1)  # (B*T*N, F-1)
            
            # 投影基础特征
            base_emb = self.input_proj(base_feat)  # (B*T*N, D)
            
            # 获取agent类型嵌入
            type_emb = self.agent_type_emb(agent_type.reshape(-1))  # (B*T*N, D)
            
            # 合并特征
            emb = base_emb + type_emb
            
            # 重塑回原始维度
            emb = emb.reshape(B, T, N, -1)  # (B, T, N, D)
        else:
            # 重塑张量以适应线性层
            x_reshaped = x.reshape(-1, F)  # (B*T*N, F)
            
            # 直接投影所有特征
            emb = self.input_proj(x_reshaped)  # (B*T*N, D)
            
            # 重塑回原始维度
            emb = emb.reshape(B, T, N, -1)  # (B, T, N, D)
        
        return emb