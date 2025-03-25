import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from datasets.training_dataset import LMTrainingDataset
from models.bev_encoder import BEVPredictionModel
from tqdm import tqdm
import time
from LM_wm.configs.config import (
    TRAINING_DATA_DIR, VALIDATION_DATA_DIR, MODEL_DIR, BATCH_SIZE,
    NUM_EPOCHS, LEARNING_RATE, HISTORY_STEPS, HIDDEN_DIM, ACTION_DIM,
    NUM_WORKERS, CUDA_DEVICE, PIN_MEMORY, USE_AMP, USE_CUDNN_BENCHMARK,
    VALIDATION_FREQ, EARLY_STOPPING_PATIENCE, GRADIENT_ACCUMULATION_STEPS,
    PREFETCH_FACTOR
)

def setup_device():
    """
    设置训练设备
    """
    device = torch.device(CUDA_DEVICE)
    if device.type == 'cuda':
        # 设置CUDA优化
        torch.backends.cudnn.benchmark = USE_CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = False
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU训练")
    return device

def create_dataloaders(device):
    """
    创建训练和验证数据加载器
    """
    # 创建完整数据集
    full_dataset = LMTrainingDataset(
        data_dir=TRAINING_DATA_DIR,
        history_steps=HISTORY_STEPS
    )
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY if device.type == 'cuda' else False,
        prefetch_factor=PREFETCH_FACTOR
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY if device.type == 'cuda' else False,
        prefetch_factor=PREFETCH_FACTOR
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    return train_loader, val_loader

def setup_model(device):
    """
    设置模型
    """
    model = BEVPredictionModel(
        action_dim=ACTION_DIM,
        history_steps=HISTORY_STEPS,
        hidden_dim=HIDDEN_DIM,
        device=device
    ).to(device)
    
    # 如果使用多GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU训练")
        model = torch.nn.DataParallel(model)
    
    return model

def setup_optimizer(model):
    """
    设置优化器和学习率调度器
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    return optimizer, scheduler

def save_checkpoint(model, optimizer, epoch, loss, is_best=False):
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
        torch.save(checkpoint, Path(MODEL_DIR) / 'best_model.pth')
    else:
        torch.save(checkpoint, Path(MODEL_DIR) / f'checkpoint_epoch_{epoch+1}.pth')

def validate(model, val_loader, device, scaler=None):
    """
    验证模型
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中"):
            bev_frames = batch['bev_frames'].to(device)
            actions = batch['actions'].to(device)
            next_frame = batch['next_frame'].to(device)
            
            if USE_AMP and scaler is not None:
                with torch.cuda.amp.autocast():
                    pred_encoding, target_encoding = model(bev_frames, actions, next_frame)
                    loss = model.compute_loss(pred_encoding, target_encoding)
            else:
                pred_encoding, target_encoding = model(bev_frames, actions, next_frame)
                loss = model.compute_loss(pred_encoding, target_encoding)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train_model():
    """
    训练模型
    """
    # 设置设备
    device = setup_device()
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(device)
    
    # 创建模型
    model = setup_model(device)
    
    # 设置优化器和调度器
    optimizer, scheduler = setup_optimizer(model)
    
    # 设置自动混合精度训练
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        # 使用tqdm创建进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备上
            bev_frames = batch['bev_frames'].to(device)
            actions = batch['actions'].to(device)
            next_frame = batch['next_frame'].to(device)
            
            # 使用自动混合精度训练
            if USE_AMP and scaler is not None:
                with torch.cuda.amp.autocast():
                    pred_encoding, target_encoding = model(bev_frames, actions, next_frame)
                    loss = model.compute_loss(pred_encoding, target_encoding)
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                pred_encoding, target_encoding = model(bev_frames, actions, next_frame)
                loss = model.compute_loss(pred_encoding, target_encoding)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}'})
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Average Loss: {avg_loss:.4f}")
        
        # 验证
        if (epoch + 1) % VALIDATION_FREQ == 0:
            val_loss = validate(model, val_loader, device, scaler)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, best_val_loss, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss)
    
    # 打印训练时间
    training_time = time.time() - start_time
    print(f"\n训练完成！总训练时间: {training_time/3600:.2f}小时")

def main():
    print("开始训练模型...")
    train_model()
    print("训练完成！")

if __name__ == "__main__":
    main() 