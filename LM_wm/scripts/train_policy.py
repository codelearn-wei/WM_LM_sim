import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import  random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from LM_wm.datasets.training_dataset import LMTrainingDataset
from LM_wm.models.wm_policy import WM_Policy
from LM_wm.models.feature_extractor import DINOFeatureExtractor
from LM_wm.configs.policy_config import PolicyConfig
from LM_wm.utils.logger import setup_logger

def setup_device():
    """设置训练设备"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    return device

def create_dataloaders(config):
    """
    创建训练和验证数据加载器
    """
    # 创建完整数据集
    full_dataset = LMTrainingDataset(
        data_dir=config.data_dir,
        history_steps=config.history_steps,
        focus_on_road=config.focus_on_road if hasattr(config, 'focus_on_road') else True
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

def save_checkpoint(model, optimizer, epoch, loss, config, is_best=False):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if is_best:
        torch.save(checkpoint, os.path.join(config.save_dir, 'best_model.pth'))
    else:
        torch.save(checkpoint, os.path.join(config.save_dir, f'checkpoint_epoch_{epoch}.pth'))

def validate(model, val_loader, config, feature_extractor):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # 获取数据
            bev_frames = batch['bev_frames'].to(config.device)
            actions = batch['actions'].to(config.device)
            next_frame = batch['next_frame'].to(config.device)
            
            # 提取特征
            history_features = []
            for t in range(config.history_steps):
                frame = bev_frames[:, t]
                features = feature_extractor.extract_features(frame)
                history_features.append(features)
            history_features = torch.stack(history_features, dim=1)  # [B, T, dino_dim]
            
            target_features = feature_extractor.extract_features(next_frame)  # [B, dino_dim]
            
            # 前向传播
            pred_features, target_features = model(history_features, actions, target_features)
            
            # 计算损失
            loss = model.compute_loss(pred_features, target_features)
            
            total_loss += loss.item()
            
            # 打印验证进度
            if (batch_idx + 1) % config.log_interval == 0:
                print(f"Validation Batch {batch_idx+1}/{len(val_loader)}, Loss: {loss.item():.6f}")
    
    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train_policy(config):
    """训练策略网络的主函数"""
    # 设置随机种子
    torch.manual_seed(config.seed)
    
    # 设置日志
    logger = setup_logger(config.log_dir)
    logger.info("Starting policy training...")
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(config)
    
    # 创建特征提取器（仅用于推理）
    feature_extractor = DINOFeatureExtractor(config.device)
    
    # 创建模型
    model = WM_Policy(
        action_dim=config.action_dim,
        history_steps=config.history_steps,
        hidden_dim=config.hidden_dim,
        device=config.device,
        mode='feature'
    ).to(config.device)
    
    # 创建优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        
        # 创建进度条
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}', 
                   leave=True, position=0, ncols=100)
        
        # 训练阶段
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # 获取数据
            bev_frames = batch['bev_frames'].to(config.device)
            actions = batch['actions'].to(config.device)
            next_frame = batch['next_frame'].to(config.device)
            
            # 提取特征
            with torch.no_grad():
                history_features = []
                for t in range(config.history_steps):
                    frame = bev_frames[:, t]
                    features = feature_extractor.extract_features(frame)
                    history_features.append(features)
                history_features = torch.stack(history_features, dim=1)  # [B, T, dino_dim]
                
                target_features = feature_extractor.extract_features(next_frame)  # [B, dino_dim]
            
            # 前向传播
            # try:
            pred_features, target_features = model(history_features, actions, target_features)
            
            # 计算损失
            loss = model.compute_loss(pred_features, target_features)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            # 更新参数
            optimizer.step()
            
            # 更新损失
            train_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
            # except Exception as e:
            #     logger.error(f"前向传播或反向传播时发生错误: {e}")
            #     continue
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{config.num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}")
        
        # 计算平均损失
        if len(train_loader) > 0:
            train_loss /= len(train_loader)
            logger.info(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.6f}")
        
        # 验证阶段
        if (epoch + 1) % config.val_interval == 0:
            try:
                val_loss = validate(model, val_loader, config, feature_extractor)
                logger.info(f"Epoch {epoch+1}/{config.num_epochs}, Val Loss: {val_loss:.6f}")
                
                # 更新学习率
                scheduler.step(val_loss)
                
                # 检查是否需要保存模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(model, optimizer, epoch, train_loss, config, is_best=True)
                    logger.info(f"New best model saved! Val Loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
                
                # 检查是否需要早停
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            except Exception as e:
                logger.error(f"验证阶段发生错误: {e}")
        
        # 定期保存检查点
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, config)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    config = PolicyConfig()
    train_policy(config) 