import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv , GATConv , SAGPooling 
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import math  


class VehicleEncoder(nn.Module):
    def __init__(self, vehicle_feature_dim, gnn_out_dim, num_heads, attention_out_dim, latent_dim, lstm_hidden_dim):
        """
        初始化 VehicleEncoder 模型

        参数：
            vehicle_feature_dim: 每个车辆的特征维度（如位置x, y和速度vx, vy）
            gnn_out_dim: GNN 输出的特征维度
            num_heads: 注意力机制中的头数
            attention_out_dim: 注意力机制输出后的特征维度
            latent_dim: 最终潜在表示的维度
            lstm_hidden_dim: LSTM 隐藏层的维度
        """
        super(VehicleEncoder, self).__init__()
        assert gnn_out_dim % num_heads == 0, "gnn_out_dim must be divisible by num_heads"
        
        # 图神经网络层，用于建模车辆间的空间关系
        self.gnn = GCNConv(vehicle_feature_dim, gnn_out_dim)
        # 多头注意力机制，捕捉车辆间的依赖关系
        self.attention = nn.MultiheadAttention(embed_dim=gnn_out_dim, num_heads=num_heads)
        # 车辆聚合注意力层，计算每辆车的权重
        self.vehicle_attention = nn.Linear(gnn_out_dim, 1)
        # LSTM，用于时序建模
        self.lstm = nn.LSTM(gnn_out_dim, lstm_hidden_dim, batch_first=True)
        # 线性层，生成最终编码
        self.linear1 = nn.Linear(lstm_hidden_dim, attention_out_dim)
        self.linear2 = nn.Linear(attention_out_dim, latent_dim)
        # Dropout 和激活函数，防止过拟合
        # self.dropout1 = nn.Dropout(0.2)
        # self.dropout2 = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, trajectories):
        """
        前向传播

        参数：
            trajectories: 车辆轨迹数据，形状为 (B, T, N, F)
                - B: 批次大小
                - T: 时间步数
                - N: 车辆数
                - F: 特征维度

        返回：
            encoded: 编码后的潜在表示，形状为 (B, T, latent_dim)
        """
        B, T, N, F = trajectories.shape
        device = trajectories.device
        
        # 创建全连接图的边索引，表示车辆间无向连接
        edge_index = torch.combinations(torch.arange(N, device=device), r=2).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # 准备批次化的图数据
        data_list = []
        for b in range(B):
            for t in range(T):
                node_features = trajectories[b, t]  # (N, F)
                data = Data(x=node_features, edge_index=edge_index)
                data_list.append(data)
        
        # 通过 GNN 处理图数据
        batch = Batch.from_data_list(data_list).to(device)
        gnn_output = self.gnn(batch.x, batch.edge_index)
        # gnn_output = self.dropout1(gnn_output)
        gnn_output = self.relu(gnn_output)
        gnn_output = gnn_output.view(B, T, N, -1)  # (B, T, N, gnn_out_dim)
        
        # 对每个时间步应用多头注意力机制
        attn_outputs = []
        for t in range(T):
            attn_input = gnn_output[:, t, :, :].permute(1, 0, 2)  # (N, B, gnn_out_dim)
            attn_output, _ = self.attention(attn_input, attn_input, attn_input)
            attn_output = attn_output.permute(1, 0, 2)  # (B, N, gnn_out_dim)
            attn_outputs.append(attn_output)
        
        attn_output = torch.stack(attn_outputs, dim=1)  # (B, T, N, gnn_out_dim)
        
        # 计算车辆注意力权重并归一化
        vehicle_weights = self.vehicle_attention(attn_output).squeeze(-1)  # (B, T, N)
        vehicle_weights = torch.softmax(vehicle_weights, dim=2)  # 归一化权重
        
        # 加权求和聚合车辆特征
        aggregated = torch.einsum('btnd,btn->btd', attn_output, vehicle_weights)  # (B, T, gnn_out_dim)
        
        # 通过 LSTM 建模时间依赖
        lstm_output, _ = self.lstm(aggregated)  # (B, T, lstm_hidden_dim)
        
        # 生成最终编码
        encoded = self.linear1(lstm_output)
        encoded = self.relu(encoded)
        # encoded = self.dropout2(encoded)
        encoded = self.linear2(encoded)
        
        return encoded

class ActionEmbedding(nn.Module):
    def __init__(self, action_dim, latent_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(action_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),  # 添加层归一化
            nn.ReLU(),
            nn.Dropout(0.1),               # 添加dropout
            nn.Linear(latent_dim * 2, latent_dim)
        )
    
    def forward(self, actions):
        """
        Transform actions to latent space
        actions: (B, H, action_dim)
        Returns: (B, H, latent_dim)
        """
        return self.embedding(actions)


class PositionalEncoding(nn.Module):
    """位置编码器，为序列数据添加位置信息"""
    
    def __init__(self, d_model, max_len=100):
        super().__init__()
        
        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        参数:
            x: 输入张量 [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len]
    
class PolicyNetwork(nn.Module):
    def __init__(self, latent_dim, hidden_dim, nhead=8, num_layers=4, dropout=0.1):
        """
        基于Transformer的策略网络
        
        参数:
            latent_dim: 隐状态维度
            hidden_dim: 隐藏层维度
            nhead: 多头注意力中的头数
            num_layers: Transformer编码器层数
            dropout: Dropout比率
        """
        super().__init__()
        
        # 位置编码器，为序列添加位置信息
        self.pos_encoder = PositionalEncoding(latent_dim)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 动作编码映射层
        self.action_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 动作与隐藏状态的融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 额外的cross-attention层，关注历史轨迹和当前动作之间的关系
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 预测未来状态的多层次MLP
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 历史状态的自适应权重层
        self.history_attention = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, historical_encoding, action_embedding):
        """
        预测下一个状态
        
        参数:
            historical_encoding: 历史轨迹编码 [batch_size, seq_len, latent_dim]
            action_embedding: 动作嵌入 [batch_size, seq_len, latent_dim]
            
        返回:
            predicted_state: 预测的下一个状态 [batch_size, 1, latent_dim]
        """
        batch_size, seq_len, _ = historical_encoding.shape
        
        # 添加位置编码
        historical_encoding = self.pos_encoder(historical_encoding)
        
        # 通过Transformer编码器处理历史序列
        # 创建注意力掩码，使模型在每个位置只关注当前和之前的时间步
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(historical_encoding.device)
        transformer_output = self.transformer_encoder(historical_encoding, mask=mask)
        
        # 计算历史轨迹的注意力权重
        history_weights = self.history_attention(transformer_output)  # [batch_size, seq_len, 1]
        
        # 加权聚合历史信息
        weighted_history = torch.sum(transformer_output * history_weights, dim=1)  # [batch_size, latent_dim]
        
        # 处理最后一个时间步的动作
        last_action = action_embedding[:, -1, :]  # [batch_size, latent_dim]
        
        # 投影动作到高维空间
        action_projected = self.action_projection(last_action)  # [batch_size, hidden_dim]
        
        # 融合历史轨迹和动作信息
        combined = torch.cat([weighted_history, last_action], dim=-1)  # [batch_size, latent_dim + latent_dim]
        fused_state = self.fusion_layer(combined)  # [batch_size, hidden_dim]
        
        # 使用cross-attention进一步建模历史轨迹和当前动作的关系
        # 将fused_state扩展为序列形式，作为query
        query = fused_state.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 将transformer_output作为key和value
        transformer_output_projected = self.action_projection(transformer_output)  # [batch_size, seq_len, hidden_dim]
        cross_attention_output, _ = self.cross_attention(
            query=query,
            key=transformer_output_projected,
            value=transformer_output_projected
        )  # [batch_size, 1, hidden_dim]
        
        # 结合cross-attention输出和融合状态
        enhanced_state = cross_attention_output.squeeze(1) + fused_state  # [batch_size, hidden_dim]
        
        # 预测下一个时间步的状态
        predicted_state = self.predictor(enhanced_state).unsqueeze(1)  # [batch_size, 1, latent_dim]
        
        return predicted_state
class WorldModel(nn.Module):
    def __init__(self, vehicle_feature_dim, gnn_out_dim, num_heads, 
                 attention_out_dim, latent_dim, action_dim,lstm_hidden_dim):
        super().__init__()
        
        self.encoder = VehicleEncoder(
            vehicle_feature_dim=vehicle_feature_dim, 
            gnn_out_dim=gnn_out_dim, 
            num_heads=num_heads,
            attention_out_dim=attention_out_dim, 
            latent_dim=latent_dim,
            lstm_hidden_dim=lstm_hidden_dim
        )
        
        self.action_embedding = ActionEmbedding(
            action_dim=action_dim, 
            latent_dim=latent_dim
        )
        
        self.policy_network = PolicyNetwork(latent_dim,lstm_hidden_dim)
    
    def forward(self, historical_trajectories, actions):
        """
        historical_trajectories: (B, H, N, F)
        actions: (B, H, action_dim)
        Returns: predicted next state
        """
        # Encode historical trajectories
        historical_encoding = self.encoder(historical_trajectories)
        
        # Embed actions
        action_embedding = self.action_embedding(actions)
        
        # Predict next state
        predicted_next_state = self.policy_network(historical_encoding, action_embedding)
        
        return predicted_next_state


class TrajectoryDecoder(nn.Module):
    def __init__(self, latent_dim, attention_out_dim, num_heads, gnn_out_dim, vehicle_feature_dim, num_vehicles,lstm_hidden_dim):
        super(TrajectoryDecoder, self).__init__()
        self.num_vehicles = num_vehicles
        
        # Increase complexity of expansion from latent space
        self.expand = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.LayerNorm(latent_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(latent_dim * 2, attention_out_dim)
        )
        
        # Add intermediate layer for vehicle projection
        self.vehicle_projection = nn.Sequential(
            nn.Linear(attention_out_dim, attention_out_dim * 2),
            nn.LayerNorm(attention_out_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(attention_out_dim * 2, num_vehicles * gnn_out_dim),
            nn.LayerNorm(num_vehicles * gnn_out_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Multiple attention layers to capture complex vehicle relationships
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=gnn_out_dim,
                num_heads=num_heads
            ) for _ in range(3)  # Stacking 3 attention layers
        ])
        
        # Add normalization and skip connections for attention
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(gnn_out_dim) for _ in range(3)
        ])
        
        # More complex feature projection with residual connections
        self.feature_projection = nn.Sequential(
            nn.Linear(gnn_out_dim, gnn_out_dim * 2),
            nn.LayerNorm(gnn_out_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(gnn_out_dim * 2, gnn_out_dim),
            nn.LayerNorm(gnn_out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gnn_out_dim, gnn_out_dim // 2),
            nn.LayerNorm(gnn_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gnn_out_dim // 2, vehicle_feature_dim)
        )
        
        # Add additional MLP for refining vehicle features
        self.refinement = nn.Sequential(
            nn.Linear(vehicle_feature_dim, vehicle_feature_dim * 2),
            nn.LayerNorm(vehicle_feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(vehicle_feature_dim * 2, vehicle_feature_dim)
        )
    
    def forward(self, latent_state):
        """
        将隐状态解码为车辆特征
        
        参数:
        latent_state (torch.Tensor): 来自WorldModel的隐状态 (B, 1, latent_dim)
        
        返回:
        torch.Tensor: 解码后的车辆特征 (B, 1, N, F)
        """
        batch_size = latent_state.size(0)
        device = latent_state.device
        
        # 扩展维度
        expanded = self.expand(latent_state)  # (B, 1, attention_out_dim)
        
        # 投影到所有车辆的特征
        all_vehicles = self.vehicle_projection(expanded)  # (B, 1, N*gnn_out_dim)
        all_vehicles = all_vehicles.view(batch_size, 1, self.num_vehicles, -1)  # (B, 1, N, gnn_out_dim)
        
        # 重塑以应用注意力
        attention_input = all_vehicles.squeeze(1).permute(1, 0, 2)  # (N, B, gnn_out_dim)
        
        # 应用多层多头注意力机制
        attn_output = attention_input
        for i, (attention, norm) in enumerate(zip(self.attention_layers, self.attention_norms)):
            residual = attn_output
            attn_output, _ = attention(
                query=attn_output,
                key=attn_output,
                value=attn_output
            )
            attn_output = norm(attn_output + residual)  # 添加残差连接和层归一化
            
        # 重塑回原始格式
        attn_output = attn_output.permute(1, 0, 2)  # (B, N, gnn_out_dim)
        attn_output = attn_output.unsqueeze(1)  # (B, 1, N, gnn_out_dim)
        
        # 投影回原始特征维度
        vehicle_features = self.feature_projection(attn_output)  # (B, 1, N, F)
        
        # 额外的特征细化
        refined_features = self.refinement(vehicle_features)
        
        # 残差连接 - 确保最终输出保持稳定
        final_features = vehicle_features + refined_features
        
        return final_features
    
class WorldModelWithDecoder(nn.Module):
    """Combined world model with decoder for trajectory prediction"""
    
    def __init__(self, vehicle_feature_dim, gnn_out_dim, num_heads, 
                 attention_out_dim, latent_dim, lstm_hidden_dim , action_dim, num_vehicles):
        super().__init__()
        
        # Use original WorldModel
        self.world_model = WorldModel(
            vehicle_feature_dim=vehicle_feature_dim, 
            gnn_out_dim=gnn_out_dim, 
            num_heads=num_heads,
            attention_out_dim=attention_out_dim, 
            latent_dim=latent_dim,
            action_dim=action_dim,
            lstm_hidden_dim = lstm_hidden_dim
        )
        
        # Add decoder
        self.decoder = TrajectoryDecoder(
            latent_dim=latent_dim,
            attention_out_dim=attention_out_dim,
            num_heads=num_heads,
            gnn_out_dim=gnn_out_dim,
            vehicle_feature_dim=vehicle_feature_dim,
            num_vehicles=num_vehicles,
            lstm_hidden_dim = lstm_hidden_dim
        )
    
    def forward(self, historical_trajectories, actions):
        """
        Forward pass through the combined model
        
        Args:
            historical_trajectories: (B, H, N, F)
            actions: (B, H, action_dim)
            
        Returns:
            predicted next trajectories (B, 1, N, F)
        """
        # Use WorldModel to predict next state latent representation
        predicted_latent = self.world_model(historical_trajectories, actions)
        
        
        # Decode latent state to vehicle features
        predicted_trajectories = self.decoder(predicted_latent)
        
        return predicted_trajectories

    