import torch
from model import WorldModel
from torch import nn, optim
from model import WorldModelWithDecoder
import math
def freeze_parameters(model, component_name, component):
    """
    Freezes parameters of a specific model component
    
    Args:
        model: The model containing the component
        component_name: Name of the component for logging
        component: The actual component whose parameters should be frozen
        
    Returns:
        int: Number of frozen parameters
    """
    # Freeze all parameters in the component
    frozen_count = 0
    for name, param in component.named_parameters():
        param.requires_grad = False
        frozen_count += 1
        
    print(f"{component_name} parameters frozen, will not be trained:")
    for name, param in component.named_parameters():
        print(f"  - {name}: {param.shape}, requires_grad = {param.requires_grad}")
    print(f"Total frozen parameters: {frozen_count}")
    
    return frozen_count

def collect_trainable_parameters(model, component_dict):
    """
    Collects trainable parameters from specified components
    
    Args:
        model: The model containing the components
        component_dict: Dictionary mapping component names to actual components
        
    Returns:
        list: Trainable parameters
    """
    trainable_params = []
    trainable_count = 0
    
    for component_name, component in component_dict.items():
        for name, param in component.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                trainable_count += 1
                print(f"Will train {component_name}.{name}: {param.shape}")
    
    print(f"Total trainable parameters: {trainable_count}")
    return trainable_params

def freeze_encoder_parameters(model):
    """
    Freezes encoder parameters in WorldModel, only trains other parts
    
    Args:
        model (WorldModel): Model with parameters to partially freeze
        
    Returns:
        list: Trainable parameters (excluding frozen encoder parameters)
    """
    freeze_parameters(model, "Encoder", model.encoder)
    
    # Collect trainable parameters from remaining components
    component_dict = {
        'action_embedding': model.action_embedding,
        'policy_network': model.policy_network
    }
    
    return collect_trainable_parameters(model, component_dict)


def freeze_world_model_parameters(combined_model):
    """
    Freezes all WorldModel parameters, only trains decoder parameters
    
    Args:
        combined_model (WorldModelWithDecoder): Model with parameters to partially freeze
        
    Returns:
        list: Trainable parameters (decoder parameters only)
    """
    freeze_parameters(combined_model, "WorldModel", combined_model.world_model)
    
    # Collect trainable parameters from decoder
    component_dict = {'decoder': combined_model.decoder}
    
    return collect_trainable_parameters(combined_model, component_dict)

class BaseTrainingFramework:
    """Base class for training frameworks with common functionality"""
    
    def __init__(self, learning_rate=5e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Gradient clipping value
        self.grad_clip_value = 1.0
        
        # Common loss functions
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def _split_trajectories(self, trajectories):
        """Split trajectories into historical and next trajectory"""
        historical_trajectories = trajectories[:, :-1]  # (B, H, N, F)
        next_trajectory = trajectories[:, -1:]  # (B, 1, N, F)
        return historical_trajectories, next_trajectory
    
    def _prepare_batch(self, trajectories, actions):
        """Move batch data to device"""
        return trajectories.to(self.device), actions.to(self.device)
        
    def validate(self, trajectories, actions):
        """Validation step - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement validate")
    
    def train_step(self, trajectories, actions):
        """Training step - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement train_step")

# 1. 修复TrainingFramework中的损失函数
class TrainingFramework(BaseTrainingFramework):
    """Training framework for WorldModel"""
    
    def __init__(self, vehicle_feature_dim, gnn_out_dim, num_heads, 
                 attention_out_dim, latent_dim, lstm_hidden_dim, action_dim, horizon_length, 
                 learning_rate=5e-4, freeze_encoder=False, 
                 lambda_cos=0.5, lambda_reg=1e-5):  # 添加损失权重参数
        super().__init__(learning_rate=learning_rate)
        
        # 保存损失权重
        self.lambda_cos = lambda_cos
        self.lambda_reg = lambda_reg
        
        # 模型初始化
        self.world_model = WorldModel(
            vehicle_feature_dim=vehicle_feature_dim, 
            gnn_out_dim=gnn_out_dim, 
            num_heads=num_heads,
            attention_out_dim=attention_out_dim, 
            latent_dim=latent_dim,
            action_dim=action_dim,
            lstm_hidden_dim=lstm_hidden_dim
        ).to(self.device)
        
        # 初始化模型权重 (添加这一部分)
        self._initialize_weights()
        
        # Freeze encoder parameters if required
        if freeze_encoder:
            trainable_params = freeze_encoder_parameters(self.world_model)
            print("Encoder parameters frozen, training only other parts")
        else:
            trainable_params = self.world_model.parameters()
            print("Training all model parameters")
        
        # 使用带学习率调度的优化器
        self.optimizer = optim.AdamW(
            trainable_params, 
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # 添加学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=5, verbose=True
        )
    
    def _initialize_weights(self):
        """初始化模型的权重"""
        for name, module in self.world_model.named_modules():
            # LSTM特殊初始化
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            
            # 线性层初始化
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.fill_(0)
    
    def _compute_loss(self, predicted_next_state, next_trajectory_encoding):
        """Compute combined loss with properly defined weights"""
        # MSE loss
        mse_loss = self.mse_loss(predicted_next_state, next_trajectory_encoding) 
        
        # Cosine similarity loss
        batch_size = predicted_next_state.size(0)
        target = torch.ones(batch_size, device=self.device)
        cos_loss = self.cosine_loss(
            predicted_next_state.view(batch_size, -1),
            next_trajectory_encoding.view(batch_size, -1),
            target
        )
        
        # 计算正则化损失 (修改为更轻的正则化)
        l2_reg = torch.tensor(0.0, device=self.device)
        for param in self.world_model.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, p=2)
        
        # Combined loss
        total_loss = mse_loss + self.lambda_cos * cos_loss + self.lambda_reg * l2_reg
        
        # 记录每个损失分量用于调试
        loss_components = {
            'mse_loss': mse_loss.item(),
            'cos_loss': cos_loss.item(),
            'reg_loss': (self.lambda_reg * l2_reg).item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def train_step(self, trajectories, actions):
        """
        Training step with improved logging
        
        Args:
            trajectories: (B, H+1, N, F)
            actions: (B, H, action_dim)
            
        Returns:
            dict: Loss values and components
        """
        trajectories, actions = self._prepare_batch(trajectories, actions)
        
        # Split trajectories
        historical_trajectories, next_trajectory = self._split_trajectories(trajectories)
        
        # Predict next state
        predicted_next_state = self.world_model(historical_trajectories, actions)
        
        # Encode next trajectory ground truth
        next_trajectory_encoding = self.world_model.encoder(next_trajectory)
        
        # Compute loss
        loss, loss_components = self._compute_loss(predicted_next_state, next_trajectory_encoding)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # 增加梯度裁剪阈值
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.grad_clip_value)
        
        # 添加梯度检查，帮助调试训练问题
        grad_norm = 0.0
        for p in self.world_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        loss_components['grad_norm'] = grad_norm
        
        self.optimizer.step()
        
        return loss_components
    
    def validate(self, trajectories, actions):
        """
        Validation step with improved consistency
        
        Args:
            trajectories: (B, H+1, N, F)
            actions: (B, H, action_dim)
            
        Returns:
            dict: Loss values and components
        """
        with torch.no_grad():
            trajectories, actions = self._prepare_batch(trajectories, actions)
            
            # Split trajectories
            historical_trajectories, next_trajectory = self._split_trajectories(trajectories)
            
            # Predict next state
            predicted_next_state = self.world_model(historical_trajectories, actions)
            
            # Encode next trajectory ground truth
            next_trajectory_encoding = self.world_model.encoder(next_trajectory)
            
            # Compute loss (使用与训练相同的损失计算)
            loss, loss_components = self._compute_loss(predicted_next_state, next_trajectory_encoding)
            
            return loss_components
    
    def update_scheduler(self, val_loss):
        """更新学习率调度器"""
        self.scheduler.step(val_loss)


# 2. 修复DecoderTrainingFramework
class DecoderTrainingFramework(BaseTrainingFramework):
    """Training framework for WorldModelWithDecoder"""
    
    def __init__(self, vehicle_feature_dim, gnn_out_dim, num_heads, 
                 attention_out_dim, latent_dim, lstm_hidden_dim, action_dim, horizon_length,
                 num_vehicles, learning_rate=1e-4, freeze_world_model=True,
                 warmup_steps=1000, weight_decay=1e-5):  # 增加预热步数，减少权重衰减
        super().__init__(learning_rate=learning_rate)
        
        # Create world model with decoder
        self.combined_model = WorldModelWithDecoder(
            vehicle_feature_dim=vehicle_feature_dim, 
            gnn_out_dim=gnn_out_dim, 
            num_heads=num_heads,
            attention_out_dim=attention_out_dim, 
            latent_dim=latent_dim,
            action_dim=action_dim,
            num_vehicles=num_vehicles,
            lstm_hidden_dim=lstm_hidden_dim
        ).to(self.device)
        
        # 初始化权重
        self._initialize_weights()
        
        # Determine which parameters to optimize based on freeze_world_model flag
        if freeze_world_model:
            # Freeze WorldModel parameters, only train decoder
            trainable_params = freeze_world_model_parameters(self.combined_model)
        else:
            # Train all parameters
            trainable_params = self.combined_model.parameters()
        
        # Optimizer for trainable parameters
        self.optimizer = optim.AdamW(
            trainable_params, 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # L1 Loss for trajectory prediction
        self.l1_loss = nn.L1Loss(reduction='none')
        
        # Track global steps for warmup
        self.global_step = 0
        self.warmup_steps = warmup_steps
        self.base_lr = learning_rate
        
        # 使用更平缓的学习率衰减策略
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=10, verbose=True
        )
        
        # 增大梯度裁剪值
        self.grad_clip_value = 5.0
    
    def _initialize_weights(self):
        """初始化模型的权重"""
        for name, module in self.combined_model.named_modules():
            # LSTM特殊初始化
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            
            # 线性层初始化
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.fill_(0)
    
    def _get_warmup_lr(self):
        """Calculate learning rate during warmup period"""
        if self.global_step < self.warmup_steps:
            # 使用平滑的余弦预热而非线性预热
            return self.base_lr * 0.5 * (1 + math.cos(math.pi * 
                   (self.warmup_steps - self.global_step) / self.warmup_steps))
        return self.base_lr
    
    def train_step(self, trajectories, actions):
        """
        Training step with detailed monitoring
        
        Args:
            trajectories: (B, H+1, N, F)
            actions: (B, H, action_dim)
            
        Returns:
            dict: Dictionary containing loss value and predictions
        """
        # Increment global step counter
        self.global_step += 1
        
        # Apply warmup learning rate
        if self.global_step <= self.warmup_steps:
            lr = self._get_warmup_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        trajectories, actions = self._prepare_batch(trajectories, actions)
        
        # Split trajectories
        historical_trajectories, next_trajectory = self._split_trajectories(trajectories)
        
        # Predict next trajectory
        predicted_trajectories = self.combined_model(historical_trajectories, actions)
        
        # 只预测x位置和y位置
        predicted_trajectories_reduced = predicted_trajectories[:, :, :, 0:2]
        next_trajectory_reduced = next_trajectory[:, :, :, 0:2]
            
        # Compute L1 loss
        l1_errors = self.l1_loss(predicted_trajectories_reduced, next_trajectory_reduced)
        # 使用蒙特卡洛采样来估计重要性权重
        with torch.no_grad():
            # 获取距离作为误差度量
            distances = torch.sqrt(torch.sum((predicted_trajectories_reduced - next_trajectory_reduced) ** 2, dim=-1))
            # 标准化距离
            max_dist = torch.max(distances)
            min_dist = torch.min(distances)
            if max_dist > min_dist:
                norm_distances = (distances - min_dist) / (max_dist - min_dist)
            else:
                norm_distances = torch.zeros_like(distances)
            
            # 生成重要性权重
            importance_weights = 1.0 / (1.0 + norm_distances)
            
            # 归一化权重
            importance_weights = importance_weights / importance_weights.sum()
        
        # 计算加权L1损失
        l1_loss = (l1_errors * importance_weights.unsqueeze(-1)).sum()
        
        # 同时计算欧几里得距离用于监控
        with torch.no_grad():
            euclidean_loss = (distances * importance_weights).sum().item()
        
        loss = l1_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # 增加梯度监控
        grad_norm = 0.0
        for p in self.combined_model.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Gradient clipping with monitoring
        parameters_to_clip = [p for p in self.combined_model.parameters() if p.requires_grad]
        if len(parameters_to_clip) > 0:
            torch.nn.utils.clip_grad_norm_(parameters_to_clip, self.grad_clip_value)
        
        self.optimizer.step()
        
        # 返回详细的监控信息
        return {
            'loss': loss.item(),
            'euclidean_loss': euclidean_loss,
            'grad_norm': grad_norm,
            'lr': self.optimizer.param_groups[0]['lr'],
            'predictions': predicted_trajectories.detach(),
            'targets': next_trajectory
        }

    def validate(self, trajectories, actions):
        """
        保持验证与训练一致的损失计算
        """
        with torch.no_grad():
            trajectories, actions = self._prepare_batch(trajectories, actions)
            
            # Split trajectories
            historical_trajectories, next_trajectory = self._split_trajectories(trajectories)
            
            # Predict next trajectory
            predicted_trajectories = self.combined_model(historical_trajectories, actions)
            
            # 只预测x位置和y位置
            predicted_trajectories_reduced = predicted_trajectories[:, :, :, 0:2]
            next_trajectory_reduced = next_trajectory[:, :, :, 0:2]
            
            # 计算L1损失 (与训练保持一致)
            l1_errors = self.l1_loss(predicted_trajectories_reduced, next_trajectory_reduced)
            
            # 计算距离作为误差度量 (与训练使用相同的权重计算方法)
            distances = torch.sqrt(torch.sum((predicted_trajectories_reduced - next_trajectory_reduced) ** 2, dim=-1))
            
            # 标准化距离
            max_dist = torch.max(distances)
            min_dist = torch.min(distances)
            if max_dist > min_dist:
                norm_distances = (distances - min_dist) / (max_dist - min_dist)
            else:
                norm_distances = torch.zeros_like(distances)
            
            # 生成重要性权重
            importance_weights = 1.0 / (1.0 + norm_distances)
            
            # 归一化权重
            importance_weights = importance_weights / importance_weights.sum()
            
            # 计算加权L1损失
            l1_loss = (l1_errors * importance_weights.unsqueeze(-1)).sum().item()
            
            # 计算加权欧几里得距离损失
            euclidean_loss = (distances * importance_weights).sum().item()
            
            # 返回损失值和预测结果
            return {
                'loss': l1_loss,
                'euclidean_loss': euclidean_loss,
                'predictions': predicted_trajectories,
                'targets': next_trajectory
            }
    
    def update_scheduler(self, val_loss):
        """更新学习率调度器，采用更平缓的衰减策略"""
        if self.global_step > self.warmup_steps:
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr < old_lr:
                print(f"学习率减小: {old_lr:.6f} -> {new_lr:.6f}")
        
        