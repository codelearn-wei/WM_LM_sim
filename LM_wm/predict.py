import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

# 从您的训练文件中导入必要的类
from trainer import DecoderTrainingFramework
from train import NormalizationStats, compute_ade_fde

class DecoderInference:
    """用于解码器模型推理的类"""
    
    def __init__(self, model_path: str, config: Dict):
        """
        初始化推理类
        
        参数:
        - model_path: 训练好的模型路径
        - config: 模型配置字典
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载模型和配置
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.config = config
        
        # 初始化模型框架
        self.framework = DecoderTrainingFramework(
            learning_rate=1e-4,  # 推理时不会用到学习率，但初始化需要
            **self.config
        )
        
        # 加载模型权重
        self.framework.combined_model.load_state_dict(self.checkpoint['model_state_dict'])
        self.framework.combined_model.to(self.device)
        self.framework.combined_model.eval()
        
        # 加载归一化参数
        if 'normalization' in self.checkpoint:
            self.norm_stats = NormalizationStats(
                self.checkpoint['normalization']['means'],
                self.checkpoint['normalization']['stds']
            )
        else:
            self.norm_stats = None
            print("警告: 模型没有保存归一化参数，可能会影响预测准确性")

    def load_test_data(self, test_folder: str, batch_size: int = 16) -> DataLoader:
        """
        加载测试数据
        
        参数:
        - test_folder: 包含测试数据的文件夹
        - batch_size: 批次大小
        
        返回:
        - 测试数据加载器
        """
        pt_files = [f for f in os.listdir(test_folder) if f.endswith('.pt')]
        
        if not pt_files:
            raise ValueError(f"在 {test_folder} 中未找到 .pt 文件")
        
        trajectories_list = []
        actions_list = []
        
        for pt_file in tqdm(pt_files, desc="加载测试数据"):
            file_path = os.path.join(test_folder, pt_file)
            data = torch.load(file_path)
            
            if isinstance(data.tensor_dataset, TensorDataset):
                file_trajectories, file_actions = data.tensor_dataset.tensors
            else:
                file_trajectories, file_actions = data.tensor_dataset
            
            trajectories_list.append(file_trajectories)
            actions_list.append(file_actions)
        
        # 合并所有数据
        test_trajectories = torch.cat(trajectories_list, dim=0)
        test_actions = torch.cat(actions_list, dim=0)
        
        # 如果有归一化参数，对测试数据进行归一化
        if self.norm_stats:
            test_trajectories = self.norm_stats.normalize(test_trajectories)
        
        test_dataset = TensorDataset(test_trajectories, test_actions)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        print(f"测试数据大小: {len(test_dataset)}")
        return test_loader

    def predict(self, test_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        使用模型进行预测
        
        参数:
        - test_loader: 测试数据加载器
        
        返回:
        - 预测轨迹
        - 真实轨迹
        - 评估指标
        """
        self.framework.combined_model.eval()
        
        all_predictions = []
        all_targets = []
        
        ade_sum = 0.0
        fde_sum = 0.0
        loss_sum = 0.0
        total_batches = 0
        
        with torch.no_grad():
            with tqdm(test_loader, desc="预测中") as progress_bar:
                for batch_trajectories, batch_actions in progress_bar:
                    batch_trajectories = batch_trajectories.to(self.device)
                    batch_actions = batch_actions.to(self.device)
                    
                    # 使用框架的验证方法获取预测
                    batch_metrics = self.framework.validate(batch_trajectories, batch_actions)
                    
                    # 获取预测和真实轨迹
                    predictions = batch_metrics['predictions'].cpu()
                    targets = batch_metrics['targets'].cpu()
                    
                    # 如果有归一化参数，对预测和目标进行反归一化
                    if self.norm_stats:
                        predictions = self.norm_stats.denormalize(predictions)
                        targets = self.norm_stats.denormalize(targets)
                    
                    # 计算ADE和FDE
                    ade, fde = compute_ade_fde(predictions, targets, features_indices=[0, 1])
                    
                    # 更新累积指标
                    ade_sum += ade
                    fde_sum += fde
                    loss_sum += batch_metrics['loss']
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f"{batch_metrics['loss']:.4f}",
                        'ade': f"{ade:.4f}",
                        'fde': f"{fde:.4f}"
                    })
                    
                    all_predictions.append(predictions)
                    all_targets.append(targets)
                    total_batches += 1
        
        # 合并所有批次的预测和目标
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 计算平均指标
        metrics = {
            'loss': loss_sum / total_batches,
            'ade': ade_sum / total_batches,
            'fde': fde_sum / total_batches
        }
        
        print(f"测试结果: 损失={metrics['loss']:.6f}, ADE={metrics['ade']:.6f}, FDE={metrics['fde']:.6f}")
        
        return all_predictions, all_targets, metrics

    def visualize_predictions(self, predictions: torch.Tensor, targets: torch.Tensor, 
                             num_samples: int = 5, save_dir: Optional[str] = None) -> None:
        """
        可视化预测轨迹
        
        参数:
        - predictions: 预测轨迹 [B, H, N, F]
        - targets: 真实轨迹 [B, H, N, F]
        - num_samples: 要可视化的样本数量
        - save_dir: 保存可视化结果的目录
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 随机选择样本进行可视化
        indices = np.random.choice(predictions.shape[0], min(num_samples, predictions.shape[0]), replace=False)
        
        for i, idx in enumerate(indices):
            pred = predictions[idx]  # [H, N, F]
            target = targets[idx]    # [H, N, F]
            
            # 创建图
            plt.figure(figsize=(12, 8))
            
            # 遍历每个对象
            for n in range(min(5, pred.shape[1])):  # 最多显示5个对象
                # 获取x和y坐标 (假设前两个特征是x,y)
                pred_x = pred[:, n, 0].numpy()
                pred_y = pred[:, n, 1].numpy()
                target_x = target[:, n, 0].numpy()
                target_y = target[:, n, 1].numpy()
                
                # 绘制轨迹
                plt.plot(pred_x, pred_y, 'r-', alpha=0.7, linewidth=2, label=f'预测 {n+1}' if n==0 else "")
                plt.plot(target_x, target_y, 'b-', alpha=0.7, linewidth=2, label=f'真实 {n+1}' if n==0 else "")
                
                # 标记起点和终点
                plt.scatter(pred_x[0], pred_y[0], c='r', marker='o', s=50, alpha=0.7)
                plt.scatter(pred_x[-1], pred_y[-1], c='r', marker='x', s=50, alpha=0.7)
                plt.scatter(target_x[0], target_y[0], c='b', marker='o', s=50, alpha=0.7)
                plt.scatter(target_x[-1], target_y[-1], c='b', marker='x', s=50, alpha=0.7)
            
            plt.title(f'样本 {idx}: 轨迹预测vs真实')
            plt.xlabel('X 坐标')
            plt.ylabel('Y 坐标')
            plt.legend()
            plt.grid(True)
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'trajectory_sample_{idx}.png'))
                plt.close()
            else:
                plt.show()

def main():
    # 模型配置（需要与训练时使用的配置一致）
    base_config = {
        'vehicle_feature_dim': 7,
        'gnn_out_dim': 256,
        'num_heads': 4,
        'attention_out_dim': 256,
        'latent_dim': 256,
        'action_dim': 10,
        'horizon_length': 20,
        'lstm_hidden_dim': 256,
        'num_vehicles': 10
    }
    
    # 初始化推理类
    inference = DecoderInference(
        model_path='get_LM_scene/WM_model/best_decoder_model.pt',
        config=base_config
    )
    
    # 加载测试数据
    test_loader = inference.load_test_data(
        test_folder='get_LM_scene/WM_train_data/train_data_relative',
        batch_size=1
    )
    
    # 进行预测
    predictions, targets, metrics = inference.predict(test_loader)
    
    # 可视化结果
    inference.visualize_predictions(
        predictions=predictions,
        targets=targets,
        num_samples=10,
        save_dir='get_LM_scene/results/visualizations'
    )
    
    # 保存指标结果
    results_dir = 'get_LM_scene/results'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(f"测试损失: {metrics['loss']:.6f}\n")
        f.write(f"平均位移误差 (ADE): {metrics['ade']:.6f}\n")
        f.write(f"最终位移误差 (FDE): {metrics['fde']:.6f}\n")

if __name__ == "__main__":
    main()