import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mamba import Mamba

# Make sure F is imported correctly
functional = F

class TrajectoryEmbedding(nn.Module):
    def __init__(self, in_features, d_model, agent_type_num=None, use_agent_type=True, dropout=0.1):
        """
        Args:
            in_features: 原始输入特征数（例如 x, y, vx, vy, heading, ...）
            d_model: 嵌入维度
            agent_type_num: agent 类型的数量（如车、人、自行车等），如果用 one-hot 或 embedding 表示
            use_agent_type: 是否使用 agent type 作为特征之一
        """
        super(TrajectoryEmbedding, self).__init__()
        self.use_agent_type = use_agent_type
        self.base_feat_dim = in_features - 1 if use_agent_type else in_features
        
        self.input_proj = nn.Sequential(
            nn.Linear(self.base_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        if use_agent_type:
            self.agent_type_emb = nn.Embedding(agent_type_num, d_model)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, N, F) - Batch, Time, Num_agents, Features
        Returns:
            emb: (B, T, N, D) - Embedded features
        """
        if self.use_agent_type:
            base_feat = x[..., :-1]
            agent_type = x[..., -1].long()
            base_emb = self.input_proj(base_feat)
            type_emb = self.agent_type_emb(agent_type)
            emb = base_emb + type_emb
        else:
            emb = self.input_proj(x)
        return emb

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

class TrajectoryEncoder(nn.Module):
    def __init__(self, in_features, d_model, agent_type_num=None, use_agent_type=True, num_attn_layers=2):
        """
        Args:
            in_features: 原始输入特征数
            d_model: 模型维度
            agent_type_num: agent类型数量
            use_agent_type: 是否使用agent type
            num_attn_layers: 注意力层数
        """
        super().__init__()
        
        self.embed = TrajectoryEmbedding(in_features, d_model, agent_type_num, use_agent_type)
        self.pos_enc = PositionalEncoding(d_model)
        self.temporal_encoder = MambaTemporalEncoder(d_model)
        
        # 堆叠多层注意力模块
        self.attn_layers = nn.ModuleList([
            AttentionSpatialInteraction(d_model) for _ in range(num_attn_layers)
        ])
    
    def forward(self, x):
        """
        Args:
            x: (B, T, N, F) - 轨迹输入
        Returns:
            out: (B, T, N, D) - 编码后特征
        """
        x = self.embed(x)  # (B, T, N, D)
        x = self.pos_enc(x)  # 添加位置编码
        x = self.temporal_encoder(x)  # Mamba时序建模
        
        # 应用多层注意力
        for attn_layer in self.attn_layers:
            x = attn_layer(x)  # 纯注意力交互建模
            
        return x

class TrajectoryDecoder(nn.Module):
    def __init__(self, d_model, pred_len, output_dim=2, num_modes=6):
        """
        Args:
            d_model: 模型维度
            pred_len: 预测长度
            output_dim: 输出维度 (通常是2，表示x,y坐标)
            num_modes: 多模态预测的模式数量
        """
        super().__init__()
        self.pred_len = pred_len
        self.num_modes = num_modes
        
        # 模式选择头 - 为每种可能的轨迹生成置信度分数
        self.mode_selection_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_modes),
        )
        
        # 轨迹预测头 - 为每种模式生成轨迹预测
        self.trajectory_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, pred_len * output_dim),
            ) for _ in range(num_modes)
        ])
    
    def forward(self, x):
        """
        Args:
            x: (B, T, N, D) - 编码后特征
        Returns:
            trajectories: (B, N, M, T_pred, output_dim) - 多模态轨迹预测
            confidences: (B, N, M) - 各模态置信度
        """
        B, T, N, D = x.shape
        
        # 使用最后一个时间步作为预测输入
        x_last = x[:, -1]  # (B, N, D)
        
        # 预测各模态置信度
        confidences = self.mode_selection_head(x_last)  # (B, N, M)
        confidences = functional.softmax(confidences, dim=-1)
        
        # 每个模态生成轨迹
        trajectories = []
        for i in range(self.num_modes):
            traj = self.trajectory_head[i](x_last)  # (B, N, pred_len*output_dim)
            traj = traj.view(B, N, self.pred_len, -1)  # (B, N, pred_len, output_dim)
            trajectories.append(traj)
            
        trajectories = torch.stack(trajectories, dim=2)  # (B, N, M, pred_len, output_dim)
        
        return trajectories, confidences

class TrajPlanningModel(nn.Module):
    def __init__(self, 
                 in_features, 
                 d_model=128, 
                 agent_type_num=4, 
                 use_agent_type=True,
                 num_attn_layers=2,
                 pred_len=30,
                 output_dim=2,
                 num_modes=6):
        """
        Args:
            in_features: 输入特征维度
            d_model: 模型隐藏维度
            agent_type_num: agent类型数量
            use_agent_type: 是否使用agent类型
            num_attn_layers: 注意力层数
            pred_len: 预测长度
            output_dim: 输出维度
            num_modes: 多模态轨迹数量
        """
        super().__init__()
        
        self.encoder = TrajectoryEncoder(
            in_features=in_features,
            d_model=d_model,
            agent_type_num=agent_type_num,
            use_agent_type=use_agent_type,
            num_attn_layers=num_attn_layers
        )
        
        self.decoder = TrajectoryDecoder(
            d_model=d_model,
            pred_len=pred_len,
            output_dim=output_dim,
            num_modes=num_modes
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, N, F) - 轨迹输入
        Returns:
            trajectories: (B, N, M, T_pred, output_dim) - 多模态轨迹预测
            confidences: (B, N, M) - 各模态置信度
        """
        encoded = self.encoder(x)
        trajectories, confidences = self.decoder(encoded)
        return trajectories, confidences

class TrajPlanningLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        轨迹规划损失函数
        Args:
            reduction: 损失聚合方式 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_trajs, pred_conf, gt_trajs, gt_mask=None):
        """
        多模态轨迹预测损失（使用MinADE/MinFDE方法）
        
        Args:
            pred_trajs: (B, N, M, T, 2) - 预测轨迹
            pred_conf: (B, N, M) - 预测置信度
            gt_trajs: (B, N, T, 2) - 真实轨迹
            gt_mask: (B, N) - 可选，指示哪些agent有效的掩码
            
        Returns:
            loss_dict: 包含各种损失的字典
        """
        B, N, M, T, D = pred_trajs.shape
        
        # 将GT轨迹扩展到与预测相同的形状（增加M维度）
        gt_trajs = gt_trajs.unsqueeze(2).expand(-1, -1, M, -1, -1)  # [B, N, M, T, 2]
        
        # 计算每个预测轨迹与GT的L2距离
        # [B, N, M, T]
        point_wise_dist = torch.sqrt(((pred_trajs - gt_trajs) ** 2).sum(-1) + 1e-6)
        
        # 每个模态的平均位移误差 (ADE) - [B, N, M]
        ade_per_mode = point_wise_dist.mean(dim=3)
        
        # 每个模态的最终位移误差 (FDE) - [B, N, M]
        fde_per_mode = point_wise_dist[:, :, :, -1]
        
        # 找到ADE最小的模态 - [B, N]
        min_ade_idx = torch.argmin(ade_per_mode, dim=2)
        min_fde_idx = torch.argmin(fde_per_mode, dim=2)
        
        # 获取最佳模态的ADE和FDE
        batch_idx = torch.arange(B).view(-1, 1).expand(-1, N).reshape(-1)
        agent_idx = torch.arange(N).view(1, -1).expand(B, -1).reshape(-1)
        
        best_ade_mode_idx = min_ade_idx.reshape(-1)
        best_fde_mode_idx = min_fde_idx.reshape(-1)
        
        min_ade = ade_per_mode[batch_idx, agent_idx, best_ade_mode_idx].reshape(B, N)
        min_fde = fde_per_mode[batch_idx, agent_idx, best_fde_mode_idx].reshape(B, N)
        
        # 计算负对数似然 (NLL) 损失
        best_mode_conf = pred_conf[batch_idx, agent_idx, best_ade_mode_idx].reshape(B, N)
        nll = -torch.log(best_mode_conf + 1e-10)
        
        # 如果有掩码，应用它
        if gt_mask is not None:
            min_ade = min_ade * gt_mask
            min_fde = min_fde * gt_mask
            nll = nll * gt_mask
            
            # 计算有效agent的数量
            valid_count = gt_mask.sum().clamp(min=1.0)
        else:
            valid_count = B * N
        
        # 根据reduction方法聚合损失
        if self.reduction == 'mean':
            min_ade = min_ade.sum() / valid_count
            min_fde = min_fde.sum() / valid_count
            nll = nll.sum() / valid_count
        elif self.reduction == 'sum':
            min_ade = min_ade.sum()
            min_fde = min_fde.sum()
            nll = nll.sum()
            
        # 多样性损失 - 鼓励不同模态之间的差异
        diversity_dist = 0
        for i in range(M):
            for j in range(i+1, M):
                mode_dist = torch.sqrt(((pred_trajs[:,:,i] - pred_trajs[:,:,j]) ** 2).sum(-1) + 1e-6)
                diversity_dist += mode_dist.mean(dim=2)  # 平均时间步的距离
                
        diversity_dist = diversity_dist / (M * (M-1) / 2)  # 平均所有模态对的距离
        
        if gt_mask is not None:
            diversity_dist = (diversity_dist * gt_mask).sum() / valid_count
        else:
            diversity_dist = diversity_dist.mean()
            
        # 总损失 = ADE + FDE + NLL - 多样性损失（多样性损失是负的，因为我们希望最大化多样性）
        total_loss = min_ade + min_fde + 0.5 * nll - 0.1 * diversity_dist
        
        loss_dict = {
            'total_loss': total_loss,
            'min_ade': min_ade,
            'min_fde': min_fde,
            'nll': nll,
            'diversity': diversity_dist
        }
        
        return loss_dict

if __name__ == "__main__":
    # 示例参数
    B, T, N, F = 8, 20, 10, 6  # 8批次，20帧，10个车，每车6维特征
    pred_len = 30
    d_model = 128
    agent_type_num = 2
    
    # 创建模型
    model = TrajPlanningModel(
        in_features=F,
        d_model=d_model,
        agent_type_num=agent_type_num,
        use_agent_type=True,
        num_attn_layers=2,
        pred_len=pred_len,
        output_dim=2,
        num_modes=3
    )
    
    # 创建输入 - 生成合法的随机输入包括agent_type
    x = torch.randn(B, T, N, F-1)  # 先创建F-1维特征
    # 为agent_type创建随机整数 (范围是0到agent_type_num-1)
    agent_types = torch.randint(0, agent_type_num, (B, T, N, 1))
    # 拼接特征和类型
    x = torch.cat([x, agent_types.float()], dim=-1)
    
    # 前向传播
    trajectories, confidences = model(x)
    
    print("Input shape:", x.shape)
    print("Output trajectories shape:", trajectories.shape)
    print("Output confidences shape:", confidences.shape)
    
    # 创建损失函数
    criterion = TrajPlanningLoss()
    
    # 创建假GT数据
    gt_trajs = torch.randn(B, N, pred_len, 2)
    gt_mask = torch.ones(B, N)  # 所有agent都有效
    
    # 计算损失
    loss_dict = criterion(trajectories, confidences, gt_trajs, gt_mask)
    
    print("\nLoss values:")
    for k, v in loss_dict.items():
        print(f"{k}: {v.item():.4f}")