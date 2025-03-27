import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from datasets.training_dataset import LMTrainingDataset
from models.bev_encoder import BEVPredictionModel
from configs.config import Config
from utils.visualization import visualize_predictions, visualize_weighted_regions, visualize_weight_schedule
from utils.logger import setup_logger
import matplotlib.pyplot as plt
import argparse

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

def setup_model(config):
    """
    设置模型
    """
    # 确定当前应该使用的初始权重
    if config.use_progressive_weights and config.weight_schedule and 0 in config.weight_schedule:
        # 如果有渐进式权重配置，使用第0个epoch的权重作为初始值
        road_weight = config.weight_schedule[0].get("road", config.initial_road_weight)
        vehicle_weight = config.weight_schedule[0].get("vehicle", config.initial_vehicle_weight)
        boundary_weight = config.weight_schedule[0].get("boundary", config.initial_boundary_weight)
        other_losses_weight = config.weight_schedule[0].get("other_losses", config.initial_other_losses_weight)
    else:
        # 否则使用配置的初始值
        road_weight = config.initial_road_weight
        vehicle_weight = config.initial_vehicle_weight
        boundary_weight = config.initial_boundary_weight
        other_losses_weight = config.initial_other_losses_weight
    
    model = BEVPredictionModel(
        action_dim=config.action_dim,
        history_steps=config.history_steps,
        hidden_dim=config.hidden_dim,
        device=config.device,
        mode=config.train_mode,
        road_weight=road_weight,
        vehicle_weight=vehicle_weight,
        boundary_weight=boundary_weight,
        other_losses_weight=other_losses_weight,
        weight_schedule=config.weight_schedule if config.use_progressive_weights else None,
        current_epoch=0
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
    
    # 创建可视化目录
    viz_dir = os.path.join(config.log_dir, 'val_visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="验证中")):
            bev_frames = batch['bev_frames'].to(config.device)
            actions = batch['actions'].to(config.device)
            next_frame = batch['next_frame'].to(config.device)
            
            # 获取道路掩码（如果存在）
            road_mask = batch.get('road_mask')
            if road_mask is not None:
                road_mask = road_mask.to(config.device)
            
            if config.train_mode == 'feature':
                pred_features, target_features = model(bev_frames, actions, next_frame)
                loss = model.compute_loss(pred_features, target_features)
            else:
                pred_features, target_features, pred_image, next_frame = model(bev_frames, actions, next_frame)
                loss = model.compute_loss(pred_features, target_features, pred_image, next_frame, road_mask)
                
                # 保存一些验证结果可视化
                if batch_idx == 0:
                    vis_images = visualize_predictions(
                        pred_image[:min(4, pred_image.size(0))], 
                        next_frame[:min(4, next_frame.size(0))]
                    )
                    vis_path = os.path.join(viz_dir, f'val_predictions.png')
                    vis_images.savefig(vis_path)
                    plt.close(vis_images)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train_feature_mode(model, train_loader, val_loader, optimizer, scheduler, config, logger):
    """特征预测模式训练"""
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # 更新epoch用于渐进式权重调整
        if config.use_progressive_weights:
            model.update_epoch(epoch)
            
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
            
            # 前向传播
            pred_features, target_features = model(bev_frames, actions, next_frame)
            
            # 计算损失
            loss = model.compute_loss(pred_features, target_features)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            
            train_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
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
    
    # 创建可视化目录
    viz_dir = os.path.join(config.log_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 保存一个样本的加权区域图，用于理解损失函数的权重
    if len(train_loader) > 0:
        sample_batch = next(iter(train_loader))
        sample_image = sample_batch['next_frame'][0].unsqueeze(0)
        weight_viz_path = os.path.join(viz_dir, 'weight_regions_initial.png')
        visualize_weighted_regions(sample_image, save_path=weight_viz_path)
        logger.info(f"初始权重区域可视化已保存到 {weight_viz_path}")
    
    # 记录训练过程中的权重变化
    weight_history = {
        'epoch': [],
        'vehicle': [],
        'road': [],
        'boundary': [],
        'other_losses': []
    }
    
    # 安全获取模型属性的辅助函数
    def get_model_attr(attr_name, default_value=None):
        """安全获取模型属性，处理DataParallel模型"""
        try:
            if isinstance(model, torch.nn.DataParallel):
                return getattr(model.module, attr_name, default_value)
            return getattr(model, attr_name, default_value)
        except (AttributeError, KeyError) as e:
            logger.warning(f"无法获取模型属性 {attr_name}: {e}")
            return default_value
    
    for epoch in range(config.num_epochs):
        # 更新epoch用于渐进式权重调整
        if config.use_progressive_weights:
            # 安全更新epoch
            try:
                if isinstance(model, torch.nn.DataParallel):
                    model.module.update_epoch(epoch)
                else:
                    model.update_epoch(epoch)
                
                # 记录当前权重
                weight_history['epoch'].append(epoch)
                
                # 安全获取权重值
                image_loss_fn = get_model_attr('image_loss_fn')
                if image_loss_fn:
                    weight_history['vehicle'].append(getattr(image_loss_fn, 'vehicle_weight', config.weights['initial']['vehicle']))
                    weight_history['road'].append(getattr(image_loss_fn, 'road_weight', config.weights['initial']['road']))
                    weight_history['boundary'].append(getattr(image_loss_fn, 'boundary_weight', config.weights['initial']['boundary']))
                else:
                    # 使用配置中的权重作为备用
                    weight_history['vehicle'].append(config.weights['initial']['vehicle'])
                    weight_history['road'].append(config.weights['initial']['road'])
                    weight_history['boundary'].append(config.weights['initial']['boundary'])
                
                weight_history['other_losses'].append(get_model_attr('other_losses_weight', config.weights['initial']['other_losses']))
                
                # 每10个epoch或在权重变化点，保存加权区域可视化
                if epoch in config.weight_schedule or epoch % 10 == 0:
                    if len(train_loader) > 0:
                        sample_batch = next(iter(train_loader))
                        sample_image = sample_batch['next_frame'][0].unsqueeze(0)
                        weight_viz_path = os.path.join(viz_dir, f'weight_regions_epoch_{epoch}.png')
                        visualize_weighted_regions(sample_image, save_path=weight_viz_path)
                        logger.info(f"Epoch {epoch}权重区域可视化已保存到 {weight_viz_path}")
            except Exception as e:
                logger.error(f"更新epoch时发生错误: {e}")
            
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
            
            # 获取道路掩码（如果存在）
            road_mask = None
            if 'road_mask' in batch and config.focus_on_road:
                road_mask = batch['road_mask'].to(config.device)
            
            # 前向传播
            try:
                pred_features, target_features, pred_image, _ = model(bev_frames, actions, next_frame)
                
                # 计算损失
                loss = model.compute_loss(pred_features, target_features, pred_image, next_frame, road_mask)
                
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
            except Exception as e:
                logger.error(f"前向传播或反向传播时发生错误: {e}")
                continue
        
        # 计算平均损失
        if len(train_loader) > 0:
            train_loss /= len(train_loader)
            logger.info(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.6f}")
        
        # 每隔val_interval个epoch进行验证
        if (epoch + 1) % config.val_interval == 0:
            try:
                val_loss = validate(model, val_loader, config)
                logger.info(f"Epoch {epoch+1}/{config.num_epochs}, Val Loss: {val_loss:.6f}")
                
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
        
        # 更新学习率
        scheduler.step()
        
        # 定期保存检查点
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, config)
    
    # 训练结束后，保存权重历史记录
    if config.use_progressive_weights and len(weight_history['epoch']) > 0:
        try:
            # 绘制权重历史曲线
            plt.figure(figsize=(10, 6))
            plt.plot(weight_history['epoch'], weight_history['vehicle'], 'b-', label='车辆权重')
            plt.plot(weight_history['epoch'], weight_history['road'], 'g-', label='道路权重')
            plt.plot(weight_history['epoch'], weight_history['boundary'], 'r-', label='边界权重')
            plt.plot(weight_history['epoch'], weight_history['other_losses'], 'm-', label='其他损失权重')
            plt.title('训练过程中的权重变化')
            plt.xlabel('Epoch')
            plt.ylabel('权重值')
            plt.grid(True)
            plt.legend()
            
            weight_history_path = os.path.join(viz_dir, 'weight_history.png')
            plt.savefig(weight_history_path)
            plt.close()
            logger.info(f"权重历史记录已保存到 {weight_history_path}")
        except Exception as e:
            logger.error(f"保存权重历史记录时发生错误: {e}")

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
    viz_dir = os.path.join(config.log_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(config.log_dir)
    logger.info(f"Starting training in {config.train_mode} mode")
    
    # 如果使用渐进式权重调整，生成并保存权重调度可视化
    if config.use_progressive_weights:
        logger.info("使用渐进式权重调整策略")
        
        try:
            # 记录权重配置信息
            logger.info(f"权重调整范围: Epoch {config.weight_epochs['start']} 到 {config.weight_epochs['end']}")
            
            # 记录初始权重和最终权重
            logger.info("初始权重配置:")
            for key, value in config.weights['initial'].items():
                logger.info(f"  {key}: {value}")
                
            logger.info("最终权重配置:")
            for key, value in config.weights['final'].items():
                logger.info(f"  {key}: {value}")
            
            # 生成并保存权重调度可视化图表
            try:
                weight_schedule_path = os.path.join(viz_dir, 'weight_schedule.png')
                visualize_weight_schedule(config, save_path=weight_schedule_path)
                logger.info(f"权重调度可视化已保存到 {weight_schedule_path}")
            except Exception as e:
                logger.error(f"生成权重调度可视化时发生错误: {e}")
                logger.info("继续训练过程...")
        except Exception as e:
            logger.error(f"配置权重调度时发生错误: {e}")
            logger.info("将使用默认权重配置继续训练...")
    
    try:
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
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("训练中断")

def main():
    """主函数，处理命令行参数，支持训练和预测模式"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='BEV预测模型训练与预测工具')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train',
                        help='运行模式: train (训练模型) 或 predict (使用训练好的模型进行预测)')
    
    # 训练模式特有参数
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument('--train_mode', type=str, choices=['feature', 'image'], default='image',
                            help='训练模式: feature (特征预测) 或 image (图像生成)')
    
    # 预测模式特有参数
    predict_group = parser.add_argument_group('预测参数')
    predict_group.add_argument('--checkpoint', type=str, default=None, 
                            help='模型检查点路径，默认使用最佳模型')
    predict_group.add_argument('--data_dir', type=str, default=None,
                            help='测试数据目录，默认使用配置中的数据目录')
    predict_group.add_argument('--output_dir', type=str, default="LM_wm/predictions",
                            help='预测结果输出目录')
    predict_group.add_argument('--batch_size', type=int, default=4,
                            help='预测批次大小')
    predict_group.add_argument('--sample_idx', type=int, default=None,
                            help='指定预测单个样本的索引，如果不指定则进行批量预测')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"开始训练模型，使用模式: {args.train_mode}")
        train_model(mode=args.train_mode)
        print("训练完成！")
    elif args.mode == 'predict':
        # 导入预测功能
        try:
            # 尝试使用绝对导入
            from LM_wm.scripts.predict import predict_main_func
        except (ImportError, ModuleNotFoundError):
            try:
                # 尝试使用相对导入
                from .predict import predict_main_func
            except (ImportError, ValueError):
                # 直接导入本地模块
                try:
                    from predict import predict_main_func
                except ImportError:
                    print("错误：无法导入预测模块。请确保predict.py在正确的路径中。")
                    return
            
        # 调用预测主函数
        predict_main_func(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            sample_idx=args.sample_idx
        )

if __name__ == "__main__":
    main() 