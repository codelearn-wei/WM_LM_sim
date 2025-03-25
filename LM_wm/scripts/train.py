import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datasets.training_dataset import LMTrainingDataset
from models.bev_encoder import BEVPredictionModel
from tqdm import tqdm
import time
from configs.config import Config
from utils.visualization import visualize_predictions
from utils.logger import setup_logger
import matplotlib.pyplot as plt

def setup_device():
    """
    设置训练设备
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # 设置CUDA优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU训练")
    return device

def create_dataloaders(config):
    """
    创建训练和验证数据加载器
    """
    # 创建完整数据集
    full_dataset = LMTrainingDataset(
        data_dir=config.data_dir,
        history_steps=config.history_steps
    )
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    return train_loader, val_loader

def setup_model(config):
    """
    设置模型
    """
    model = BEVPredictionModel(
        action_dim=config.action_dim,
        history_steps=config.history_steps,
        hidden_dim=config.hidden_dim,
        device=config.device,
        mode=config.train_mode
    ).to(config.device)
    
    # 如果使用多GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU训练")
        model = torch.nn.DataParallel(model)
    
    return model

def setup_optimizer(model, config):
    """
    设置优化器和学习率调度器
    """
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    return optimizer, scheduler

def save_checkpoint(model, optimizer, epoch, loss, config, is_best=False):
    """
    保存检查点
    """
    # 如果是DataParallel模型，保存module
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if is_best:
        torch.save(checkpoint, os.path.join(config.save_dir, 'best_model.pth'))
    else:
        torch.save(checkpoint, os.path.join(config.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

def validate(model, val_loader, config):
    """
    验证模型
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中"):
            bev_frames = batch['bev_frames'].to(config.device)
            actions = batch['actions'].to(config.device)
            next_frame = batch['next_frame'].to(config.device)
            
            if config.train_mode == 'feature':
                pred_features, target_features = model(bev_frames, actions, next_frame)
                loss = model.compute_loss(pred_features, target_features)
            else:
                pred_features, target_features, pred_image, next_frame = model(bev_frames, actions, next_frame)
                loss = model.compute_loss(pred_features, target_features, pred_image, next_frame)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train_feature_mode(model, train_loader, val_loader, optimizer, scheduler, config, logger):
    """特征预测模式训练"""
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        
        # 训练阶段
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')):
            optimizer.zero_grad()
            
            # 获取数据
            bev_frames = batch['bev_frames'].to(config.device)
            actions = batch['actions'].to(config.device)
            next_frame = batch['next_frame'].to(config.device)
            
            # 前向传播
            pred_features, target_features = model(bev_frames, actions, next_frame)
            
            # 计算损失
            loss = model.compute_loss(pred_features, target_features)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            
            train_loss += loss.item()
            
            # 记录训练损失
            if batch_idx % config.log_interval == 0:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # 验证阶段
        val_loss = validate(model, val_loader, config)
        logger.info(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, config, is_best=True)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= config.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # 更新学习率
        scheduler.step()
        
        # 定期保存检查点
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, config)

def train_image_mode(model, train_loader, val_loader, optimizer, scheduler, config, logger):
    """图像生成模式训练"""
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        
        # 训练阶段
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')):
            optimizer.zero_grad()
            
            # 获取数据
            bev_frames = batch['bev_frames'].to(config.device)
            actions = batch['actions'].to(config.device)
            next_frame = batch['next_frame'].to(config.device)
            
            # 前向传播
            pred_features, target_features, pred_image, next_frame = model(bev_frames, actions, next_frame)
            
            # 计算损失
            loss = model.compute_loss(pred_features, target_features, pred_image, next_frame)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            
            train_loss += loss.item()
            
            # 记录训练损失
            if batch_idx % config.log_interval == 0:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                
                # 可视化预测结果
                if batch_idx % config.vis_interval == 0:
                    vis_images = visualize_predictions(pred_image[:config.num_vis_samples], 
                                                    next_frame[:config.num_vis_samples])
                    # 保存可视化结果
                    vis_path = os.path.join(config.log_dir, f'vis_epoch_{epoch+1}_batch_{batch_idx}.png')
                    vis_images.savefig(vis_path)
                    plt.close(vis_images)
        
        # 验证阶段
        val_loss = validate(model, val_loader, config)
        logger.info(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, config, is_best=True)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= config.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # 更新学习率
        scheduler.step()
        
        # 定期保存检查点
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, config)

def train_model(mode='feature'):
    """训练模型的主函数"""
    # 加载配置
    config = Config()
    config.train_mode = mode
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(config.log_dir)
    logger.info(f"Starting training in {config.train_mode} mode")
    
    # 设置设备
    config.device = setup_device()
    
    # 创建数据集和数据加载器
    train_loader, val_loader = create_dataloaders(config)
    
    # 创建模型
    model = setup_model(config)
    
    # 创建优化器和学习率调度器
    optimizer, scheduler = setup_optimizer(model, config)
    
    # 根据模式选择训练函数
    if config.train_mode == 'feature':
        train_feature_mode(model, train_loader, val_loader, optimizer, scheduler, config, logger)
    else:
        train_image_mode(model, train_loader, val_loader, optimizer, scheduler, config, logger)
    
    logger.info("Training completed!")

def main():
    print("开始训练模型...")
    train_model()
    print("训练完成！")

if __name__ == "__main__":
    main() 