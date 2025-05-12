# TODO: 修改feature学习的模型，按照WM_dino的方式修改现有模型
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
import torchvision
from torch.optim import AdamW

# 编码解码任务头（提升任务模型的可解释性）



class FeatureEncoder(nn.Module):
    """特征编码器"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

class ActionEncoder(nn.Module):
    """动作编码器"""
    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

class FeaturePredictor(nn.Module):
    """特征预测器"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.predictor(x)

class PolicyPredictor(nn.Module):
    """基于时空注意力的策略预测器，用于实现 p_θ(enc_θ(o_{t-H:t}), φ(a_{t-H:t}))
    
    结构:
    1. 时空注意力机制处理观察序列
    2. 动作编码处理
    3. 特征融合和预测
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 特征转换层 - 将输入维度转换为hidden_dim
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入维度应该是dino_dim
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 时间注意力 - 捕捉时序依赖
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 空间注意力 - 捕捉空间依赖
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 时序处理
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 特征融合层 - 将hidden_dim转换为output_dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, history_encodings, action_encodings):
        """
        Args:
            history_encodings: 历史观察序列的编码 [batch_size, seq_len, dino_dim]
            action_encodings: 动作序列编码 [batch_size, seq_len, hidden_dim]
            
        Returns:
            pred_encoding: 预测的下一个观察的编码
            attention_info: 注意力权重信息，用于可视化
        """
        batch_size, seq_len, _ = history_encodings.shape
        
        # 1. 特征转换 - 只转换历史编码
        history_features = self.feature_transform(history_encodings)  # [B, T, H]
        
        # 2. 时间注意力处理观察序列
        temporal_out, temporal_weights = self.temporal_attention(
            history_features, 
            history_features, 
            history_features
        )
        temporal_features = self.norm1(history_features + temporal_out)
        
        # 3. LSTM处理时序信息
        lstm_out, _ = self.lstm(temporal_features)
        
        # 4. 空间注意力
        # 重塑特征以应用空间注意力
        spatial_features = lstm_out.reshape(batch_size * seq_len, -1, self.hidden_dim)
        spatial_out, spatial_weights = self.spatial_attention(
            spatial_features, spatial_features, spatial_features
        )
        spatial_features = self.norm2(spatial_features + spatial_out)
        
        # 重塑回原始维度
        spatial_features = spatial_features.reshape(batch_size, seq_len, -1)
        
        # 5. 特征融合
        # 使用最后一个时间步的特征和动作
        final_temporal = spatial_features[:, -1]  # [B, H]
        final_action = action_encodings[:, -1]    # [B, H]
        
        # 连接特征并生成预测
        combined = torch.cat([final_temporal, final_action], dim=-1)
        # combined = torch.cat([spatial_features, final_action], dim=-1)
        pred_encoding = self.fusion(combined)
        
        # 返回预测和注意力信息
        attention_info = {
            'temporal_attention': temporal_weights,
            'spatial_attention': spatial_weights.reshape(batch_size, seq_len, -1)
        }
        
        return pred_encoding, attention_info
   
# 需要一个预测方法用于验证
class WM_Policy(nn.Module):
    def __init__(self, action_dim, history_steps, hidden_dim, device, mode='feature',
                road_weight=3.0, vehicle_weight=15.0, boundary_weight=0.01,
                other_losses_weight=0.05, weight_schedule=None, current_epoch=0):
        super().__init__()
        self.device = device
        self.mode = mode
        self.history_steps = history_steps
        
        # 储存权重配置
        self.road_weight = road_weight
        self.vehicle_weight = vehicle_weight
        self.boundary_weight = boundary_weight
        self.other_losses_weight = other_losses_weight
        self.weight_schedule = weight_schedule
        self.current_epoch = current_epoch
        
        # DINOv2输出维度
        self.dino_dim = 768
        
 

        self.action_encoder = ActionEncoder(action_dim, hidden_dim)
        # 输入维度是DINO特征维度
        self.policy_predictor = PolicyPredictor(
            input_dim=self.dino_dim,  # 只使用DINO特征维度
            hidden_dim=hidden_dim,
            output_dim=self.dino_dim
        )
    
        
        # 特征损失函数
        self.feature_loss_fn = nn.MSELoss()
        
       
        # 初始化优化器
        self.optimizer = AdamW(self.parameters(), lr=1e-4)
        
    def compute_loss(self, pred_features, target_features, pred_image=None, target_image=None, road_mask=None):
        """计算损失函数"""
        # 特征预测损失
        feature_loss = self.feature_loss_fn(pred_features, target_features)
        
      
        return feature_loss
    
    def predict_next_state(self, current_state, action):
        """
        预测下一个潜在状态
        :param current_state: 当前状态特征，形状为 [B, dino_dim] 或 [dino_dim]
        :param action: 当前动作，形状为 [B, action_dim] 或 [action_dim]
        :return: 预测的下一个状态特征，形状为 [B, dino_dim] 或 [dino_dim]
        """
        # 确保输入是批量形式
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)  # [1, dino_dim]
        if action.dim() == 1:
            action = action.unsqueeze(0)  # [1, action_dim]
        
        # 移动到正确的设备
        current_state = current_state.to(self.device)
        action = action.to(self.device)
        
        # 编码动作
        action_encoding = self.action_encoder(action)  # [B, hidden_dim]
        
 
        
        # 预测下一个状态
        pred_features, _ = self.policy_predictor(current_state, action_encoding.unsqueeze(1))  # [B, dino_dim], _
        
        # 返回单样本或批量结果
        return pred_features.squeeze(0) if pred_features.size(0) == 1 else pred_features
        
    

    
    def forward(self, history_features, actions, target_features):
        """前向传播"""
        # 确保输入在正确的设备上
        history_features = history_features.to(self.device)
        actions = actions.to(self.device)
        # next_frame = next_frame.to(self.device)
        
        target_features = target_features.to(self.device)
        
        # 编码动作
        batch_size, seq_len, _ = actions.shape
        actions_flat = actions.reshape(-1, actions.size(-1))
        action_encodings = self.action_encoder(actions_flat)  # [B*T, hidden_dim]
        action_encodings = action_encodings.reshape(batch_size, seq_len, -1)  # [B, T, hidden_dim]
        
        pred_features, attention_info = self.policy_predictor(history_features, action_encodings)
        self.attention_maps = attention_info
        return pred_features, target_features

    