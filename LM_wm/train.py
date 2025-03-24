import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from typing import Optional, Dict, Tuple, Any, Union, List
from trainer import BaseTrainingFramework, DecoderTrainingFramework

class NormalizationStats:
    """存储归一化参数的类"""
    def __init__(self, means: torch.Tensor, stds: torch.Tensor):
        self.means = means
        self.stds = stds
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """对数据进行归一化"""
        # Ensure all tensors are on the same device
        device = data.device
        means = self.means.to(device)
        stds = self.stds.to(device)
        return (data - means) / stds
    
    def denormalize(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """对归一化的数据进行反归一化"""
        # Ensure all tensors are on the same device
        device = normalized_data.device
        means = self.means.to(device)
        stds = self.stds.to(device)
        return normalized_data * stds + means

def compute_normalization_stats(dataset: TensorDataset) -> NormalizationStats:
    """计算数据集的归一化参数"""
    trajectories, _ = dataset.tensors
    # 假设轨迹数据的形状为 [B, H, N, F]
    # 按照批次和时间维度计算均值和标准差
    means = trajectories.mean(dim=(0, 1))  # [N, F]
    stds = trajectories.std(dim=(0, 1))    # [N, F]
    
    # 防止除零，将过小的标准差设为1
    stds[stds < 1e-5] = 1.0
    
    return NormalizationStats(means, stds)

def normalize_dataset(dataset: TensorDataset, stats: NormalizationStats) -> TensorDataset:
    """对数据集进行归一化"""
    trajectories, actions = dataset.tensors
    normalized_trajectories = stats.normalize(trajectories)
    return TensorDataset(normalized_trajectories, actions)

def load_trajectory_data(pt_folder: str, normalize: bool = True) -> Tuple[TensorDataset, Optional[NormalizationStats]]:
    """从指定文件夹加载所有轨迹和动作数据，并可选地进行归一化"""
    pt_files = [f for f in os.listdir(pt_folder) if f.endswith('.pt')]
    
    if not pt_files:
        raise ValueError(f"在 {pt_folder} 中未找到 .pt 文件")
    
    trajectories_list = []
    actions_list = []
    
    for pt_file in pt_files:
        file_path = os.path.join(pt_folder, pt_file)
        data = torch.load(file_path)
        
        if isinstance(data.tensor_dataset, TensorDataset):
            file_trajectories, file_actions = data.tensor_dataset.tensors
        else:
            file_trajectories, file_actions = data.tensor_dataset
        
        trajectories_list.append(file_trajectories)
        actions_list.append(file_actions)
    
    dataset = TensorDataset(torch.cat(trajectories_list, dim=0), torch.cat(actions_list, dim=0))
    
    if normalize:
        norm_stats = compute_normalization_stats(dataset)
        dataset = normalize_dataset(dataset, norm_stats)
        return dataset, norm_stats
    
    return dataset, None

def prepare_dataloaders(dataset: TensorDataset, batch_size: int, val_ratio: float) -> Tuple[DataLoader, DataLoader]:
    """准备训练和验证数据加载器"""
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, val_loader

def compute_ade_fde(predictions: torch.Tensor, targets: torch.Tensor, features_indices: List[int] = [0, 1]) -> Tuple[float, float]:
    """
    计算平均位移误差(ADE)和最终位移误差(FDE)
    
    参数:
    - predictions: 预测轨迹，形状 [B, H, N, F]
    - targets: 真实轨迹，形状 [B, H, N, F]
    - features_indices: 用于计算位移误差的特征索引（通常是x和y坐标）
    
    返回:
    - ADE：所有时间步的平均位移误差
    - FDE：最终时间步的位移误差
    """
    # 选择指定的特征维度（通常是x和y坐标）
    pred_pos = predictions[..., features_indices]
    target_pos = targets[..., features_indices]
    cha = pred_pos[0] - target_pos[0]
    
    # 计算每个时间步的欧几里得距离
    distances = torch.sqrt(torch.sum((pred_pos - target_pos) ** 2, dim=-1))  # [B, H, N]
    
    # 计算ADE：所有时间步的平均位移误差
    ade = distances.mean().item()
    
    # 计算FDE：最终时间步的位移误差
    fde = distances[:, -1].mean().item()
    
    return ade, fde

def train_epoch(framework: BaseTrainingFramework, train_loader: DataLoader, 
                norm_stats: Optional[NormalizationStats] = None, verbose: bool = True) -> Dict[str, float]:
    """执行一个训练轮次，返回损失和可能的ADE/FDE指标"""
    # Check which model attribute exists and set to train mode
    if hasattr(framework, 'world_model'):
        framework.world_model.train()
    elif hasattr(framework, 'combined_model'):
        framework.combined_model.train()
    else:
        raise AttributeError("Framework does not have a model attribute")
    
    epoch_metrics = {'loss': 0.0}
    
    # 如果是解码器框架，准备额外指标
    is_decoder = isinstance(framework, DecoderTrainingFramework)
    if is_decoder:
        epoch_metrics.update({'ade': 0.0, 'fde': 0.0})
    
    total_batches = 0
    
    with tqdm(train_loader, desc="[Train]", disable=not verbose) as progress_bar:
        for batch_trajectories, batch_actions in progress_bar:
            # 训练步骤
            batch_metrics = framework.train_step(batch_trajectories, batch_actions)
            
            # 如果返回的是单个损失值而不是字典，则转换为字典
            if not isinstance(batch_metrics, dict):
                batch_loss = batch_metrics
                batch_metrics = {'loss': batch_loss}
            
            # 更新累积损失
            for key, value in batch_metrics.items():
                if key not in ['predictions', 'targets']:  # 只累加数值型指标
                    epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value
            
            # 如果是解码器框架且有需要，计算ADE和FDE
            if is_decoder and norm_stats and 'predictions' in batch_metrics and 'targets' in batch_metrics:
                # 反归一化预测和真实轨迹
                denorm_predictions = norm_stats.denormalize(batch_metrics['predictions'])
                denorm_targets = norm_stats.denormalize(batch_metrics['targets'])
                
                # 计算ADE和FDE (使用x和y坐标)
                ade, fde = compute_ade_fde(denorm_predictions, denorm_targets, features_indices=[0, 1])
                
                # 更新进度条显示
                progress_bar.set_postfix({
                    'loss': f"{batch_metrics['loss']:.4f}",
                    'ade': f"{ade:.4f}",
                    'fde': f"{fde:.4f}"
                })
                
                # 累加ADE和FDE
                epoch_metrics['ade'] += ade
                epoch_metrics['fde'] += fde
            else:
                progress_bar.set_postfix({'loss': f"{batch_metrics['loss']:.4f}"})
            
            total_batches += 1
    
    # 计算平均指标
    for key in epoch_metrics:
        epoch_metrics[key] /= total_batches
    
    return epoch_metrics

def validate_epoch(framework: BaseTrainingFramework, val_loader: DataLoader, 
                   norm_stats: Optional[NormalizationStats] = None, verbose: bool = True) -> Dict[str, float]:
    """执行一个验证轮次，返回损失和可能的ADE/FDE指标"""
    # Check which model attribute exists and set to eval mode
    if hasattr(framework, 'world_model'):
        framework.world_model.eval()
    elif hasattr(framework, 'combined_model'):
        framework.combined_model.eval()
    else:
        raise AttributeError("Framework does not have a model attribute")
    
    epoch_metrics = {'loss': 0.0}
    
    # 如果是解码器框架，准备额外指标
    is_decoder = isinstance(framework, DecoderTrainingFramework)
    if is_decoder:
        epoch_metrics.update({'ade': 0.0, 'fde': 0.0})
    
    total_batches = 0
    
    with torch.no_grad():
        with tqdm(val_loader, desc="[Val]", disable=not verbose) as progress_bar:
            for batch_trajectories, batch_actions in progress_bar:
                # 验证步骤
                batch_metrics = framework.validate(batch_trajectories, batch_actions)
                
                # 如果返回的是单个损失值而不是字典，则转换为字典
                if not isinstance(batch_metrics, dict):
                    batch_loss = batch_metrics
                    batch_metrics = {'loss': batch_loss}
                
                # 更新累积损失
                for key, value in batch_metrics.items():
                    if key not in ['predictions', 'targets']:  # 只累加数值型指标
                        epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value
                
                # 如果是解码器框架且有需要，计算ADE和FDE
                if is_decoder and norm_stats and 'predictions' in batch_metrics and 'targets' in batch_metrics:
                    # 反归一化预测和真实轨迹
                    denorm_predictions = norm_stats.denormalize(batch_metrics['predictions'])
                    denorm_targets = norm_stats.denormalize(batch_metrics['targets'])
                    
                    # 计算ADE和FDE (使用x和y坐标)
                    ade, fde = compute_ade_fde(denorm_predictions, denorm_targets, features_indices=[0, 1])
                    
                    # 更新进度条显示
                    progress_bar.set_postfix({
                        'loss': f"{batch_metrics['loss']:.4f}",
                        'ade': f"{ade:.4f}",
                        'fde': f"{fde:.4f}"
                    })
                    
                    # 累加ADE和FDE
                    epoch_metrics['ade'] += ade
                    epoch_metrics['fde'] += fde
                else:
                    progress_bar.set_postfix({'val_loss': f"{batch_metrics['loss']:.4f}"})
                
                total_batches += 1
    
    # 计算平均指标
    for key in epoch_metrics:
        epoch_metrics[key] /= total_batches
    
    return epoch_metrics

def save_model(framework: BaseTrainingFramework, save_path: str, epoch: int, 
              metrics: Dict[str, float], config: Dict[str, Any], 
              norm_stats: Optional[NormalizationStats] = None) -> None:
    """保存模型、训练状态和归一化参数"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Determine which model to save
    if hasattr(framework, 'world_model'):
        model = framework.world_model
    elif hasattr(framework, 'combined_model'):
        model = framework.combined_model
    else:
        raise AttributeError("Framework does not have a model attribute")
    
    save_dict = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': framework.optimizer.state_dict(),
        'scheduler_state_dict': framework.scheduler.state_dict() if hasattr(framework, 'scheduler') else None,
        'metrics': metrics,
        'config': config
    }
    
    # 如果有归一化参数，也保存它们
    if norm_stats:
        save_dict['normalization'] = {
            'means': norm_stats.means,
            'stds': norm_stats.stds
        }
    
    torch.save(save_dict, save_path)

def train_model(
    framework: BaseTrainingFramework,
    pt_folder: str,
    batch_size: int = 16,
    num_epochs: int = 50,
    val_ratio: float = 0.2,
    patience: int = 10,
    save_dir: Optional[str] = None,
    model_name: str = "model",
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    normalize: bool = True,
    features_for_metrics: List[int] = [0, 1]  # 默认用前两个特征（假设是x,y坐标）计算指标
) -> Tuple[Any, Dict[str, list]]:
    """
    通用模型训练函数，支持数据归一化和额外评价指标
    
    参数:
    - framework: 训练框架实例 (TrainingFramework 或 DecoderTrainingFramework)
    - pt_folder: 包含训练数据的文件夹
    - batch_size: 批次大小
    - num_epochs: 训练轮数
    - val_ratio: 验证集比例
    - patience: 早停耐心值
    - save_dir: 模型保存目录
    - model_name: 保存的模型名称
    - config: 模型配置字典，用于保存
    - verbose: 是否显示详细信息
    - normalize: 是否对数据进行归一化
    - features_for_metrics: 用于计算ADE和FDE的特征索引列表
    
    返回:
    - model: 训练后的模型
    - history: 训练历史
    """
    # 加载并准备数据，可选地进行归一化
    dataset, norm_stats = load_trajectory_data(pt_folder, normalize=normalize)
    train_loader, val_loader = prepare_dataloaders(dataset, batch_size, val_ratio)
    
    # 判断是否为解码器框架
    is_decoder = isinstance(framework, DecoderTrainingFramework)
    
    # 打印训练信息
    if verbose:
        print(f"\n===== 开始训练 {model_name} =====")
        print(f"数据集大小: {len(dataset)} (训练: {len(train_loader.dataset)}, 验证: {len(val_loader.dataset)})")
        print(f"批次大小: {batch_size}, 训练轮数: {num_epochs}, 早停耐心: {patience}")
        if normalize:
            print("数据已归一化")
        if is_decoder:
            print(f"将计算ADE和FDE指标，使用特征索引: {features_for_metrics}")
    
    # 训练循环
    best_val_loss = float('inf')
    best_val_metrics = None
    no_improve_count = 0
    history = {'train_loss': [], 'val_loss': []}
    
    # 如果是解码器，添加额外的指标历史记录
    if is_decoder:
        history.update({
            'train_ade': [], 'train_fde': [],
            'val_ade': [], 'val_fde': []
        })
    
    best_model_path = os.path.join(save_dir, f"best_{model_name}.pt") if save_dir else None
    
    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"\nEpoch {epoch+1}/{num_epochs}, 全局步数: {framework.global_step}")
        
        # 训练和验证
        train_metrics = train_epoch(framework, train_loader, norm_stats if is_decoder else None, verbose)
        val_metrics = validate_epoch(framework, val_loader, norm_stats if is_decoder else None, verbose)
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        
        # 如果是解码器，记录额外指标
        if is_decoder and 'ade' in train_metrics:
            history['train_ade'].append(train_metrics['ade'])
            history['train_fde'].append(train_metrics['fde'])
            history['val_ade'].append(val_metrics['ade'])
            history['val_fde'].append(val_metrics['fde'])
        
        # 更新学习率调度器
        if hasattr(framework, 'scheduler'):
            framework.scheduler.step(val_metrics['loss'])
        
        if verbose:
            print(f"训练损失: {train_metrics['loss']:.6f}, 验证损失: {val_metrics['loss']:.6f}")
            if is_decoder and 'ade' in val_metrics:
                print(f"验证 ADE: {val_metrics['ade']:.6f}, 验证 FDE: {val_metrics['fde']:.6f}")
        
        # 模型保存和早停逻辑
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_metrics = val_metrics.copy()
            no_improve_count = 0
            
            if save_dir and best_model_path:
                save_model(framework, best_model_path, epoch, val_metrics, config or {}, norm_stats)
                if verbose:
                    print(f"新的最佳模型已保存 (验证损失: {best_val_loss:.6f})")
        else:
            no_improve_count += 1
            if verbose:
                print(f"验证损失未改善 ({no_improve_count}/{patience})")
            
            if no_improve_count >= patience:
                if verbose:
                    print(f"\n早停：{patience}个epoch验证损失未改善")
                break
    
    if verbose:
        print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")
        if is_decoder and best_val_metrics and 'ade' in best_val_metrics:
            print(f"最佳验证 ADE: {best_val_metrics['ade']:.6f}, FDE: {best_val_metrics['fde']:.6f}")
    
    # Fix: Return the correct model attribute based on framework type
    if hasattr(framework, 'world_model'):
        return framework.world_model, history
    elif hasattr(framework, 'combined_model'):
        return framework.combined_model, history
    else:
        raise AttributeError("Framework does not have a model attribute")

# 使用示例
if __name__ == "__main__":
    
    from trainer import TrainingFramework, DecoderTrainingFramework
    
    # 先定义基本维度参数
    batch_size = 64          # B: 批次大小
    horizon_length = 20        # H: 时间帧数
    num_vehicles = 10          # N: 车辆数量
    vehicle_feature_dim = 2    # F: 车辆特征维度
    action_dim = 10            # 动作维度
    latent_dim = 512           # 潜在空间维度
    gnn_out_dim = 512          # GNN输出维度
    num_heads = 4              # 注意力头数量
    attention_out_dim = 512     # 注意力输出维度
    lstm_hidden_dim = 512      # LSTM隐藏层维度

    # 基础配置
    base_config = {
        'vehicle_feature_dim': vehicle_feature_dim,
        'gnn_out_dim': gnn_out_dim,
        'num_heads': num_heads,
        'attention_out_dim': attention_out_dim,
        'latent_dim': latent_dim,
        'action_dim': action_dim,
        'horizon_length': horizon_length,
        'lstm_hidden_dim': lstm_hidden_dim,
    }

    # # # 世界模型配置和训练
    # world_model_config = {**base_config}

    # world_framework = TrainingFramework(
    #     learning_rate=5e-4,
    #     freeze_encoder=False,
    #     **world_model_config
    # )

    # world_model, _ = train_model(
    #     framework=world_framework,
    #     pt_folder='get_LM_scene/WM_train_data/train_data_relative',
    #     save_dir='get_LM_scene/model',
    #     model_name='world_model',
    #     config=world_model_config,
    #     verbose=True,
    #     batch_size=batch_size,
    #     normalize=True,  # 添加归一化
    # )

    # # 解码器模型配置和训练
    decoder_config = {
        **base_config,
        'num_vehicles': num_vehicles  
    }

    decoder_framework = DecoderTrainingFramework(
        learning_rate=1e-4,
        **decoder_config, 
        freeze_world_model=False,
        warmup_steps= 2000
    )

    # 加载预训练的世界模型
    # checkpoint = torch.load('get_LM_scene/WM_model/best_world_model.pt')
    # decoder_framework.combined_model.world_model.load_state_dict(checkpoint['model_state_dict'])
    
    # 如果世界模型保存了归一化参数，也可以重用
    # norm_stats = None
    # if 'normalization' in checkpoint:
    #     norm_stats = NormalizationStats(
    #         checkpoint['normalization']['means'],
    #         checkpoint['normalization']['stds']
    #     )

    # 训练解码器模型
    decoder_model, history = train_model(
        framework=decoder_framework,
        num_epochs=50,
        pt_folder='get_LM_scene/WM_train_data/train_data_xy',
        save_dir='get_LM_scene/WM_model_xy_1',
        model_name='decoder_model',
        config=decoder_config,
        verbose=True,
        batch_size=batch_size,
        normalize=True,  # 添加归一化
        features_for_metrics=[0, 1]  # 假设前两个特征是x,y坐标
    )