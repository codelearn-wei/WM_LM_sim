import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.mamba_wm_model import MambaWorldModel
from models.loss.losses import compute_total_loss
from datasets.trajectory_dataset import TrajectoryDataset
from utils.logger import setup_logger
from utils.metrics import compute_metrics
from utils.tensorboard_logger import TensorBoardLogger

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(config['save_dir'], f'exp_{timestamp}')
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate']
        )
        
        # Initialize TensorBoard logger
        self.tb_logger = TensorBoardLogger(log_dir=os.path.join(self.exp_dir, 'tensorboard'))
        
        # Initialize logger
        self.logger = setup_logger(
            name='trainer',
            log_file=os.path.join(self.exp_dir, 'training.log')
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.last_best_save_epoch = 0
        
        # Log configuration
        self.logger.info("Training Configuration:")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        train_steps = 0
        
        # Initialize metrics
        train_metrics = {
            'ego_ade': 0, 'ego_fde': 0,
            'agent_ade': 0, 'agent_fde': 0
        }
        
        with tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate losses
                batch_loss, batch_ego_loss, batch_agent_loss = compute_total_loss(
                    outputs, outputs, batch, self.config
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += batch_loss.item()
                train_steps += 1
                
                # Compute training metrics
                with torch.no_grad():
                    # Denormalize trajectories
                    ego_mean = self.train_loader.dataset.future_ego_mean.to(self.device)
                    ego_std = self.train_loader.dataset.future_ego_std.to(self.device)
                    agent_mean = self.train_loader.dataset.future_agent_mean.to(self.device)
                    agent_std = self.train_loader.dataset.future_agent_std.to(self.device)
                    
                    denorm_ego_pred = outputs['ego_trajectories'] * ego_std + ego_mean
                    denorm_ego_gt = batch['ego_future'] * ego_std + ego_mean
                    
                    denorm_agent_pred = outputs['agent_trajectories'] * agent_std + agent_mean
                    denorm_agent_gt = batch['agent_future'] * agent_std + agent_mean
                    
                    # Compute metrics
                    ego_metrics = compute_metrics(
                        denorm_ego_pred,
                        denorm_ego_gt,
                        mode='min'
                    )
                    agent_metrics = compute_metrics(
                        denorm_agent_pred,
                        denorm_agent_gt.permute(0, 2, 1, 3),
                        mode='min'
                    )
                    
                    train_metrics['ego_ade'] += ego_metrics['ade']
                    train_metrics['ego_fde'] += ego_metrics['fde']
                    train_metrics['agent_ade'] += agent_metrics['ade']
                    train_metrics['agent_fde'] += agent_metrics['fde']
                
                # Log batch metrics to TensorBoard
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.tb_logger.log_metrics({
                    'loss': batch_loss.item(),
                    'ego_ade': ego_metrics['ade'],
                    'ego_fde': ego_metrics['fde'],
                    'agent_ade': agent_metrics['ade'],
                    'agent_fde': agent_metrics['fde']
                }, global_step, prefix='train/batch/')
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{batch_loss.item():.4f}',
                    'ego_ade': f'{ego_metrics["ade"]:.4f}',
                    'agent_ade': f'{agent_metrics["ade"]:.4f}'
                })
        
        # Average metrics for the epoch
        avg_loss = total_loss / train_steps
        for k in train_metrics:
            train_metrics[k] /= train_steps
        
        # Log epoch metrics to TensorBoard
        self.tb_logger.log_metrics({
            'loss': avg_loss,
            **train_metrics
        }, self.current_epoch, prefix='train/epoch/')
        
        return avg_loss, train_metrics
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        val_steps = 0
        
        # Initialize metrics
        val_metrics = {
            'ego_ade': 0, 'ego_fde': 0,
            'agent_ade': 0, 'agent_fde': 0
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validating')):
                # Move data to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate losses
                batch_loss, batch_ego_loss, batch_agent_loss = compute_total_loss(
                    outputs, outputs, batch, self.config
                )
                
                # Update metrics
                total_loss += batch_loss.item()
                val_steps += 1
                
                # Compute validation metrics
                ego_mean = self.val_loader.dataset.future_ego_mean.to(self.device)
                ego_std = self.val_loader.dataset.future_ego_std.to(self.device)
                agent_mean = self.val_loader.dataset.future_agent_mean.to(self.device)
                agent_std = self.val_loader.dataset.future_agent_std.to(self.device)
                
                denorm_ego_pred = outputs['ego_trajectories'] * ego_std + ego_mean
                denorm_ego_gt = batch['ego_future'] * ego_std + ego_mean
                
                denorm_agent_pred = outputs['agent_trajectories'] * agent_std + agent_mean
                denorm_agent_gt = batch['agent_future'] * agent_std + agent_mean
                
                # Compute metrics
                ego_metrics = compute_metrics(
                    denorm_ego_pred,
                    denorm_ego_gt,
                    mode='min'
                )
                agent_metrics = compute_metrics(
                    denorm_agent_pred,
                    denorm_agent_gt.permute(0, 2, 1, 3),
                    mode='min'
                )
                
                val_metrics['ego_ade'] += ego_metrics['ade']
                val_metrics['ego_fde'] += ego_metrics['fde']
                val_metrics['agent_ade'] += agent_metrics['ade']
                val_metrics['agent_fde'] += agent_metrics['fde']
                
                # Log batch metrics to TensorBoard
                global_step = self.current_epoch * len(self.val_loader) + batch_idx
                self.tb_logger.log_metrics({
                    'loss': batch_loss.item(),
                    'ego_ade': ego_metrics['ade'],
                    'ego_fde': ego_metrics['fde'],
                    'agent_ade': agent_metrics['ade'],
                    'agent_fde': agent_metrics['fde']
                }, global_step, prefix='val/batch/')
        
        # Average metrics
        avg_loss = total_loss / val_steps
        for k in val_metrics:
            val_metrics[k] /= val_steps
        
        # Log epoch metrics to TensorBoard
        self.tb_logger.log_metrics({
            'loss': avg_loss,
            **val_metrics
        }, self.current_epoch, prefix='val/epoch/')
        
        return avg_loss, val_metrics
    
    def train(self):
        """Train the model."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Train Ego ADE: {train_metrics['ego_ade']:.4f}, "
                f"Val Ego ADE: {val_metrics['ego_ade']:.4f}"
            )
            
            # Check for improvement
            if val_loss < self.best_val_loss - self.config['early_stopping_min_delta']:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint(is_best=True)
                self.last_best_save_epoch = epoch
            else:
                self.epochs_without_improvement += 1
            
            # Save regular checkpoint
            self.save_checkpoint(is_best=False)
            
            # Save best model periodically
            if (epoch - self.last_best_save_epoch) >= self.config.get('save_best_interval', 10):
                self.save_checkpoint(is_best=True, is_periodic=True)
                self.last_best_save_epoch = epoch
            
            # Early stopping
            if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                self.logger.info("Early stopping triggered")
                break
        
        self.tb_logger.close()
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, is_best=False, is_periodic=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.exp_dir,
            'checkpoints',
            f'checkpoint_epoch_{self.current_epoch}.pt'
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            if is_periodic:
                best_model_path = os.path.join(
                    self.exp_dir,
                    'checkpoints',
                    f'best_model_epoch_{self.current_epoch}.pt'
                )
            else:
                best_model_path = os.path.join(
                    self.exp_dir,
                    'checkpoints',
                    'best_model.pt'
                )
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"Saved {'periodic ' if is_periodic else ''}best model at epoch {self.current_epoch}")

def main():
    """Main training function."""
    # Model configuration
    config = {
        'input_dim': 7,  # 7 features for history
        'output_dim': 3,  # [x, y, heading] for future
        'hidden_dim': 256,
        'num_modes': 3,  # Number of prediction modes for ego vehicle
        'prediction_horizon': 30,  # 3s future
        'num_agents': 6,  # Fixed number of agents
        'history_len': 10,  # 1s history
        'num_attn_layers': 2,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'batch_size': 512,
        'grad_clip': 1.0,
        'loss_alpha': 0.1,  # Weight for mode classification loss
        'save_dir': 'checkpoints',
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 1e-4,
        'save_best_interval': 10  # Save best model every 10 epochs
    }
    
    # Create save directory
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets and data loaders
    train_dataset = TrajectoryDataset(r'src\datasets\data\agent_mask_train_data\train_trajectories_with_mask.pkl')
    val_dataset = TrajectoryDataset(r'src\datasets\data\agent_mask_train_data\val_trajectories_with_mask.pkl')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = MambaWorldModel(config).to(device)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main() 