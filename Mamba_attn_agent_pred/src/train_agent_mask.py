import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.mamba_agent_model import MambaAgentModel
from models.loss.losses_mask import compute_total_loss
from datasets.trajectory_dataset_mask import TrajectoryDatasetMask
from utils.logger import setup_logger
from utils.metrics import compute_metrics

def evaluate_model(model, data_loader, config, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: MambaAgentModel instance
        data_loader: Data loader for evaluation
        config: Configuration dictionary
        device: Device to run evaluation on
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0
    steps = 0
    
    # Initialize metrics
    ego_metrics = {
        'ade': 0, 'fde': 0,
        'ade_x': 0, 'ade_y': 0,
        'fde_x': 0, 'fde_y': 0
    }
    agent_metrics = {
        'ade': 0, 'fde': 0,
        'ade_x': 0, 'ade_y': 0,
        'fde_x': 0, 'fde_y': 0
    }
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            inputs = {
                'agent_history': batch['agent_history'].to(device),
                'agent_history_mask': batch['agent_history_mask'].to(device),
                'agent_future_mask': batch['agent_future_mask'].to(device)
            }
            targets = {
                'agent_future': batch['agent_future'].to(device),
                'agent_future_mask': batch['agent_future_mask'].to(device)
            }
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate losses
            loss = compute_total_loss(outputs, targets, config)
            total_loss += loss.item()
            
            # Denormalize trajectories before computing metrics
            # Get normalization parameters from dataset
            # ego_mean = data_loader.dataset.future_ego_mean.to(device)
            # ego_std = data_loader.dataset.future_ego_std.to(device)
            agent_mean = data_loader.dataset.future_agent_mean.to(device)
            agent_std = data_loader.dataset.future_agent_std.to(device)
            # Denormalize trajectories
            # denorm_ego_pred = outputs['ego_trajectories'] * ego_std + ego_mean
            # denorm_ego_gt = batch['ego_future'] * ego_std + ego_mean
            denorm_agent_pred = outputs['agent_trajectories'] * agent_std + agent_mean
            denorm_agent_gt = targets['agent_future'] * agent_std + agent_mean
            
            # # Compute metrics for ego vehicle (first agent)
            # ego_metrics_batch = compute_metrics(
            #     denorm_ego_pred,  # [batch_size, 1, pred_horizon, 3]
            #     denorm_ego_gt,    # [batch_size, 1, pred_horizon, 3]
            #     mode='min'
            # )
            
            # Compute metrics for other agents
            agent_metrics_batch = compute_metrics(
                denorm_agent_pred,  # [batch_size, num_agents, pred_horizon, 3]
                denorm_agent_gt.permute(0 , 2 , 1 , 3),    # [batch_size, num_agents, pred_horizon, 3]
                mode='min'
            )
            
            # Update metrics
            for k in ego_metrics.keys():
                # ego_metrics[k] += ego_metrics_batch[k]
                agent_metrics[k] += agent_metrics_batch[k]
            
            steps += 1
    
    # Average metrics
    for k in ego_metrics.keys():
        ego_metrics[k] /= steps
        agent_metrics[k] /= steps
    
    return {
        'loss': total_loss / steps,
        'ego_metrics': ego_metrics,
        'agent_metrics': agent_metrics
    }

def train_model(model, train_loader, val_loader, config, device):
    """
    Train the trajectory prediction model.
    
    Args:
        model: MambaAgentModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
        device: Device to run training on
    """
    logger = setup_logger()
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(config['checkpoint_dir'], f'agent_exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # Optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Initialize metrics tracking
    metrics = {
        'train_loss': [],
        'train_ego_ade': [],
        'train_ego_fde': [],
        'train_agent_ade': [],
        'train_agent_fde': [],
        'val_loss': [],
        'val_ego_ade': [],
        'val_ego_fde': [],
        'val_agent_ade': [],
        'val_agent_fde': [],
        'learning_rates': []
    }
    
    # Early stopping parameters
    patience = config.get('early_stopping_patience', 10)
    min_delta = config.get('early_stopping_min_delta', 1e-4)
    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_ego_ade = 0
        train_ego_fde = 0
        train_agent_ade = 0
        train_agent_fde = 0
        train_steps = 0
        
        # Create progress bar for training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
        for batch in pbar:
            # Move data to device
            inputs = {
                'agent_history': batch['agent_history'].to(device),
                'agent_history_mask': batch['agent_history_mask'].to(device),
                'agent_future_mask': batch['agent_future_mask'].to(device)
            }
            targets = {
                'agent_future': batch['agent_future'].to(device),
                'agent_future_mask': batch['agent_future_mask'].to(device)
            }
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate losses
            loss = compute_total_loss(outputs, targets, config)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            
            # Update loss metrics
            train_loss += loss.item()
            
            # Compute training metrics
            with torch.no_grad():
                # Denormalize trajectories
                agent_mean = train_loader.dataset.future_agent_mean.to(device)
                agent_std = train_loader.dataset.future_agent_std.to(device)
                
                
                # denorm_ego_pred = outputs['ego_trajectories'] * ego_std + ego_mean
                # denorm_ego_gt = batch['ego_future'] * ego_std + ego_mean
                denorm_agent_pred = outputs['agent_trajectories'] * agent_std + agent_mean
                denorm_agent_gt = targets['agent_future'] * agent_std + agent_mean
            
                # # Compute metrics for ego vehicle (first agent)
                # ego_metrics = compute_metrics(
                #     denorm_ego_pred,  # [batch_size, 1, pred_horizon, 3]
                #     denorm_ego_gt,    # [batch_size, 1, pred_horizon, 3]
                #     mode='min'
                # )
                
                # Compute metrics for other agents
                agent_metrics = compute_metrics(
                    denorm_agent_pred,  # [batch_size, num_agents, pred_horizon, 3]
                    denorm_agent_gt.permute(0 , 2 , 1 , 3),    # [batch_size, num_agents, pred_horizon, 3]
                    mode='min'
                )
                    
                # train_ego_ade += ego_metrics['ade']
                # train_ego_fde += ego_metrics['fde']
                train_agent_ade += agent_metrics['ade']
                train_agent_fde += agent_metrics['fde']
            
            train_steps += 1
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'agent_ade': f'{agent_metrics["ade"]:.4f}',
                'agent_fde': f'{agent_metrics["fde"]:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate average training metrics
        avg_train_loss = train_loss / train_steps
        avg_train_ego_ade = train_ego_ade / train_steps
        avg_train_ego_fde = train_ego_fde / train_steps
        avg_train_agent_ade = train_agent_ade / train_steps
        avg_train_agent_fde = train_agent_fde / train_steps
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, config, device)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log progress
        logger.info(f'\nEpoch {epoch+1}/{config["num_epochs"]}:')
        logger.info(f'Train Loss: {avg_train_loss:.4f}')
        logger.info(f'Train Ego ADE: {avg_train_ego_ade:.4f}')
        logger.info(f'Train Ego FDE: {avg_train_ego_fde:.4f}')
        logger.info(f'Train Agent ADE: {avg_train_agent_ade:.4f}')
        logger.info(f'Train Agent FDE: {avg_train_agent_fde:.4f}')
        logger.info(f'Val Loss: {val_metrics["loss"]:.4f}')
        
        # Log ADE and FDE metrics
        logger.info('\nEgo Vehicle Metrics:')
        logger.info(f'ADE: {val_metrics["ego_metrics"]["ade"]:.4f}')
        logger.info(f'FDE: {val_metrics["ego_metrics"]["fde"]:.4f}')
        logger.info(f'ADE_x: {val_metrics["ego_metrics"]["ade_x"]:.4f}')
        logger.info(f'ADE_y: {val_metrics["ego_metrics"]["ade_y"]:.4f}')
        logger.info(f'FDE_x: {val_metrics["ego_metrics"]["fde_x"]:.4f}')
        logger.info(f'FDE_y: {val_metrics["ego_metrics"]["fde_y"]:.4f}')
        
        logger.info('\nAgent Vehicle Metrics:')
        logger.info(f'ADE: {val_metrics["agent_metrics"]["ade"]:.4f}')
        logger.info(f'FDE: {val_metrics["agent_metrics"]["fde"]:.4f}')
        logger.info(f'ADE_x: {val_metrics["agent_metrics"]["ade_x"]:.4f}')
        logger.info(f'ADE_y: {val_metrics["agent_metrics"]["ade_y"]:.4f}')
        logger.info(f'FDE_x: {val_metrics["agent_metrics"]["fde_x"]:.4f}')
        logger.info(f'FDE_y: {val_metrics["agent_metrics"]["fde_y"]:.4f}')
        
        logger.info(f'\nLearning Rate: {current_lr:.6f}')
        
        # Save metrics
        metrics['train_loss'].append(avg_train_loss)
        metrics['train_ego_ade'].append(avg_train_ego_ade)
        metrics['train_ego_fde'].append(avg_train_ego_fde)
        metrics['train_agent_ade'].append(avg_train_agent_ade)
        metrics['train_agent_fde'].append(avg_train_agent_fde)
        metrics['val_loss'].append(val_metrics['loss'])
        metrics['val_ego_ade'].append(val_metrics['ego_metrics']['ade'])
        metrics['val_ego_fde'].append(val_metrics['ego_metrics']['fde'])
        metrics['val_agent_ade'].append(val_metrics['agent_metrics']['ade'])
        metrics['val_agent_fde'].append(val_metrics['agent_metrics']['fde'])
        metrics['learning_rates'].append(current_lr)
        
        # Early stopping check
        if val_metrics['loss'] < best_val_loss - min_delta:
            best_val_loss = val_metrics['loss']
            no_improve_epochs = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'metrics': metrics
            }, os.path.join(exp_dir, 'best_model.pth'))
            logger.info('Saved best model')
        else:
            no_improve_epochs += 1
            logger.info(f'No improvement for {no_improve_epochs} epochs')
            
            if no_improve_epochs >= patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics
        }, os.path.join(exp_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save metrics to file
        np.save(os.path.join(exp_dir, 'metrics.npy'), metrics)

def main():
    # Configuration
    config = {
        'input_dim': 7,
        'output_dim': 3,
        'hidden_dim': 512,
        'prediction_horizon': 30,
        'num_agents': 6,
        'history_len': 12,
        'num_attn_layers': 3,
        'learning_rate': 5e-4,
        'num_epochs': 100,
        'batch_size': 64,
        'grad_clip': 0.5,
        'early_stopping_patience': 15,
        'early_stopping_min_delta': 1e-4,
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints_mask',
        'warmup_epochs': 5,
        'weight_decay': 1e-4
    }
    
    # Create directories
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets and data loaders
    train_dataset = TrajectoryDatasetMask(r'src\datasets\data\agent_mask_train_data\train_trajectories_with_mask.pkl')
    val_dataset = TrajectoryDatasetMask(r'src\datasets\data\agent_mask_train_data\val_trajectories_with_mask.pkl')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = MambaAgentModel(config).to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, config, device)

if __name__ == '__main__':
    main() 