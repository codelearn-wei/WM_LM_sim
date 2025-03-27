#!/usr/bin/env python3
"""
预测脚本 - 使用训练好的BEV预测模型生成预测结果
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm

# 相对导入，确保能够正确导入模块
try:
    from models.bev_encoder import BEVPredictionModel
    from datasets.training_dataset import LMTrainingDataset
    from configs.config import Config
    from utils.visualization import visualize_predictions, visualize_weighted_regions
    from utils.logger import setup_logger
    from utils.image_utils import maintain_aspect_ratio_resize
except ImportError:
    # 当作为模块导入时，使用绝对导入
    from LM_wm.models.bev_encoder import BEVPredictionModel
    from LM_wm.datasets.training_dataset import LMTrainingDataset
    from LM_wm.configs.config import Config
    from LM_wm.utils.visualization import visualize_predictions, visualize_weighted_regions
    from LM_wm.utils.logger import setup_logger
    from LM_wm.utils.image_utils import maintain_aspect_ratio_resize

def setup_device():
    """设置设备（CPU或GPU）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU进行预测")
    return device

def load_model(checkpoint_path, config):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 模型检查点路径
        config: 配置对象
        
    Returns:
        加载好的模型
    """
    # 设置设备
    config.device = setup_device()
    
    # 创建模型
    model = BEVPredictionModel(
        action_dim=config.action_dim,
        history_steps=config.history_steps,
        hidden_dim=config.hidden_dim,
        device=config.device,
        mode='image',  # 使用图像模式
        road_weight=config.initial_road_weight,
        vehicle_weight=config.initial_vehicle_weight,
        boundary_weight=config.initial_boundary_weight,
        other_losses_weight=config.initial_other_losses_weight,
    ).to(config.device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    model.eval()
    
    print(f"成功加载模型检查点: {checkpoint_path}")
    print(f"模型训练到第 {checkpoint['epoch'] + 1} 个epoch，损失值: {checkpoint['loss']:.6f}")
    
    return model

def predict_single_sample(model, sample, config, save_dir=None):
    """
    对单个样本进行预测并可视化
    
    Args:
        model: 加载好的模型
        sample: 包含BEV帧序列和动作的样本
        config: 配置对象
        save_dir: 保存结果的目录（可选）
        
    Returns:
        预测图像
    """
    # 提取样本数据
    bev_frames = sample['bev_frames'].unsqueeze(0).to(config.device)  # 添加批次维度
    actions = sample['actions'].unsqueeze(0).to(config.device)        # 添加批次维度
    next_frame = sample['next_frame'].unsqueeze(0).to(config.device)  # 添加批次维度
    
    # 进行预测
    with torch.no_grad():
        _, _, pred_image, _ = model(bev_frames, actions, next_frame)
    
    # 如果指定了保存目录，则保存可视化结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存预测结果可视化
        viz_pred = visualize_predictions(pred_image, next_frame)
        viz_pred.savefig(os.path.join(save_dir, "prediction_result.png"))
        plt.close(viz_pred)
        
        # 保存权重区域可视化
        viz_weights = visualize_weighted_regions(next_frame[0])
        viz_weights.savefig(os.path.join(save_dir, "weight_regions.png"))
        plt.close(viz_weights)
        
        # 保存注意力图（如果可用）
        attention_maps = model.get_attention_maps()
        if attention_maps is not None:
            # TODO: 可以在这里添加注意力图可视化代码
            pass
        
        # 保存原始图像和预测图像为PNG文件
        pred_np = pred_image[0].permute(1, 2, 0).cpu().numpy()
        pred_np = np.clip(pred_np * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, "predicted.png"), cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR))
        
        target_np = next_frame[0].permute(1, 2, 0).cpu().numpy()
        target_np = np.clip(target_np * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, "target.png"), cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR))
    
    return pred_image[0]  # 返回预测图像（去除批次维度）

def predict_sequence(image_sequence, actions_sequence, model, config):
    """
    预测一个图像和动作序列
    
    Args:
        image_sequence: 图像序列 [history_steps, C, H, W]
        actions_sequence: 动作序列 [history_steps, action_dim]
        model: 加载好的模型
        config: 配置对象
        
    Returns:
        预测图像 [C, H, W]
    """
    # 确保输入形状正确
    bev_frames = image_sequence.unsqueeze(0).to(config.device)  # [1, history_steps, C, H, W]
    actions = actions_sequence.unsqueeze(0).to(config.device)   # [1, history_steps, action_dim]
    
    # 创建一个空的占位符作为next_frame（预测模式下不需要实际的next_frame）
    dummy_next_frame = torch.zeros((1, 3, config.image_size[0], config.image_size[1])).to(config.device)
    
    # 进行预测
    with torch.no_grad():
        _, _, pred_image, _ = model(bev_frames, actions, dummy_next_frame)
    
    return pred_image[0]  # 返回预测图像 [C, H, W]

def load_test_dataset(config, data_dir=None):
    """
    加载测试数据集
    
    Args:
        config: 配置对象
        data_dir: 数据目录（可选，默认使用配置中的数据目录）
        
    Returns:
        测试数据集
    """
    if data_dir:
        config.data_dir = data_dir
    
    # 创建数据集
    test_dataset = LMTrainingDataset(
        data_dir=config.data_dir,
        history_steps=config.history_steps,
        focus_on_road=config.focus_on_road
    )
    
    print(f"加载测试数据集，共 {len(test_dataset)} 个样本")
    return test_dataset

def batch_predict(model, test_loader, config, output_dir):
    """
    批量预测并保存结果
    
    Args:
        model: 加载好的模型
        test_loader: 测试数据加载器
        config: 配置对象
        output_dir: 输出目录
        
    Returns:
        None
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置进度条
    pbar = tqdm(test_loader, desc="批量预测")
    
    for batch_idx, batch in enumerate(pbar):
        # 提取批次数据
        bev_frames = batch['bev_frames'].to(config.device)
        actions = batch['actions'].to(config.device)
        next_frame = batch['next_frame'].to(config.device)
        
        # 进行预测
        with torch.no_grad():
            _, _, pred_images, _ = model(bev_frames, actions, next_frame)
        
        # 保存批次中的每个预测结果
        for i in range(pred_images.size(0)):
            # 创建当前样本的输出目录
            sample_dir = os.path.join(output_dir, f"sample_{batch_idx * test_loader.batch_size + i}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # 提取单个样本
            pred_image = pred_images[i:i+1]
            target_image = next_frame[i:i+1]
            
            # 保存预测结果可视化
            viz_pred = visualize_predictions(pred_image, target_image)
            viz_pred.savefig(os.path.join(sample_dir, "prediction.png"))
            plt.close(viz_pred)
            
            # 保存预测图像为PNG
            pred_np = pred_image[0].permute(1, 2, 0).cpu().numpy()
            pred_np = np.clip(pred_np * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(sample_dir, "predicted.png"), cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR))
            
            # 保存目标图像为PNG
            target_np = target_image[0].permute(1, 2, 0).cpu().numpy()
            target_np = np.clip(target_np * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(sample_dir, "target.png"), cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR))

def predict_main_func(checkpoint_path=None, data_dir=None, output_dir="LM_wm/predictions", batch_size=4, sample_idx=None):
    """
    预测功能的主入口函数，专为在main函数中调用设计
    
    Args:
        checkpoint_path: 模型检查点路径，默认使用最佳模型
        data_dir: 测试数据目录，默认使用配置中的数据目录
        output_dir: 输出目录
        batch_size: 批次大小
        sample_idx: 指定预测单个样本的索引，如果为None则进行批量预测
        
    Returns:
        None
    """
    # 打印标题
    print("\n" + "="*50)
    print("BEV预测模型推理系统")
    print("="*50 + "\n")
    
    # 加载配置
    config = Config()
    config.train_mode = 'image'  # 使用图像模式
    
    # 使用默认的最佳模型路径（如果未指定）
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.save_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到模型检查点: {checkpoint_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(output_dir)
    logger.info(f"开始预测，使用模型检查点: {checkpoint_path}")
    
    # 加载模型
    model = load_model(checkpoint_path, config)
    
    # 加载测试数据集
    test_dataset = load_test_dataset(config, data_dir)
    
    # 如果指定了样本索引，则只预测该样本
    if sample_idx is not None:
        if sample_idx >= len(test_dataset):
            raise ValueError(f"样本索引 {sample_idx} 超出数据集范围 (0-{len(test_dataset)-1})")
        
        # 获取指定样本
        sample = test_dataset[sample_idx]
        
        # 创建样本输出目录
        sample_dir = os.path.join(output_dir, f"sample_{sample_idx}")
        
        # 预测单个样本
        logger.info(f"预测样本 {sample_idx}")
        pred_image = predict_single_sample(model, sample, config, sample_dir)
        
        logger.info(f"预测完成，结果保存在: {sample_dir}")
        print(f"\n单样本预测完成! 结果已保存至: {sample_dir}")
    else:
        # 创建测试数据加载器
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # 批量预测
        logger.info(f"开始批量预测 {len(test_dataset)} 个样本")
        batch_predict(model, test_loader, config, output_dir)
        
        logger.info(f"批量预测完成，结果保存在: {output_dir}")
        print(f"\n批量预测完成! 共处理 {len(test_dataset)} 个样本")
        print(f"结果已保存至: {output_dir}")
    
    print("\n" + "="*50)
    return

def predict_model(checkpoint_path=None, data_dir=None, output_dir="LM_wm/predictions", batch_size=4, sample_idx=None):
    """
    主预测函数 - 向后兼容接口
    
    Args:
        checkpoint_path: 模型检查点路径，默认使用最佳模型
        data_dir: 测试数据目录，默认使用配置中的数据目录
        output_dir: 输出目录
        batch_size: 批次大小
        sample_idx: 指定预测单个样本的索引，如果为None则进行批量预测
        
    Returns:
        None
    """
    return predict_main_func(checkpoint_path, data_dir, output_dir, batch_size, sample_idx)

def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='使用BEV预测模型进行预测')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='模型检查点路径，默认使用最佳模型')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='测试数据目录，默认使用配置中的数据目录')
    parser.add_argument('--output_dir', type=str, default="LM_wm/predictions",
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='指定预测单个样本的索引，如果不指定则进行批量预测')
    
    args = parser.parse_args()
    
    # 调用主函数
    predict_main_func(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        sample_idx=args.sample_idx
    )

if __name__ == "__main__":
    main() 