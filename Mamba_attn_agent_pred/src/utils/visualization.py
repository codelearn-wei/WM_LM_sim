import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from datetime import datetime

def plot_trajectories(ego_history, ego_future, ego_pred, agent_history, agent_future, agent_pred,
                     agent_future_mask=None, save_path=None, show=True, title=None):
    """
    Plot trajectories for ego vehicle and agents.
    
    Args:
        ego_history (np.ndarray): Ego vehicle history trajectory [history_len, 3]
        ego_future (np.ndarray): Ego vehicle ground truth future trajectory [pred_horizon, 3]
        ego_pred (np.ndarray): Ego vehicle predicted trajectories [num_modes, pred_horizon, 3]
        agent_history (np.ndarray): Agent history trajectories [num_agents, history_len, 3]
        agent_future (np.ndarray): Agent ground truth future trajectories [pred_horizon, num_agents, 3]
        agent_pred (np.ndarray): Agent predicted trajectories [num_agents, pred_horizon, 3]
        agent_future_mask (np.ndarray): Agent future mask [pred_horizon, num_agents], 1 for valid timestamps, 0 for invalid
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
        title (str, optional): Title of the plot
    """
    # Set style for publication-quality plots
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2,
        'lines.markersize': 8
    })
    
    plt.figure(figsize=(12, 8))
    
    # Plot ego vehicle history
    plt.plot(ego_history[:, 0], ego_history[:, 1], 'b-', linewidth=2, label='Ego History')
    plt.scatter(ego_history[:, 0], ego_history[:, 1], c='blue', s=50, alpha=0.6, marker='o')
    
    # Plot ego vehicle ground truth future
    plt.plot(ego_future[:, 0], ego_future[:, 1], 'g-', linewidth=2, label='Ego GT Future')
    plt.scatter(ego_future[:, 0], ego_future[:, 1], c='green', s=50, alpha=0.6, marker='o')
    
    # Plot ego vehicle predicted trajectories (all modes)
    colors = ['red', 'orange', 'purple']  # Different colors for different modes
    for mode in range(ego_pred.shape[0]):
        # Plot trajectory line
        plt.plot(ego_pred[mode, :, 0], ego_pred[mode, :, 1], '--', 
                color=colors[mode], linewidth=1.5, alpha=0.7, 
                label=f'Ego Pred Mode {mode+1}' if mode == 0 else None)
    
    # Plot agents
    agent_colors = ['purple', 'orange', 'cyan', 'magenta', 'yellow']
    num_agents = agent_future.shape[1]  # Get number of agents from agent_future shape
    
    for i in range(num_agents):
        color = agent_colors[i % len(agent_colors)]
        
        # Plot agent future with mask
        if agent_future_mask is not None:
            # Get valid timestamps for this agent
            valid_timestamps = np.where(agent_future_mask[:, i])[0]
            
            if len(valid_timestamps) > 0:
                # Plot valid future points with solid line
                valid_future_x = agent_future[valid_timestamps, i, 0]
                valid_future_y = agent_future[valid_timestamps, i, 1]
                plt.plot(valid_future_x, valid_future_y, '-', color=color, alpha=0.8, linewidth=2,
                        label=f'Agent {i+1} Future')
        else:
            # Plot all future points if no mask
            plt.plot(agent_future[:, i, 0], agent_future[:, i, 1], '-', 
                    color=color, alpha=0.8, linewidth=2, 
                    label=f'Agent {i+1} Future')
        
        # Plot agent predictions with mask
        if agent_future_mask is not None:
            valid_timestamps = np.where(agent_future_mask[:, i])[0]
            if len(valid_timestamps) > 0:
                valid_pred_x = agent_pred[i, valid_timestamps, 0]
                valid_pred_y = agent_pred[i, valid_timestamps, 1]
                plt.plot(valid_pred_x, valid_pred_y, '--', color=color, linewidth=1.5,
                        alpha=0.6, label=f'Agent {i+1} Prediction')
        else:
            plt.plot(agent_pred[i, :, 0], agent_pred[i, :, 1], '--', 
                    color=color, linewidth=1.5, alpha=0.6, 
                    label=f'Agent {i+1} Prediction')
    
    # Add legend with a semi-transparent background
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              framealpha=0.8, edgecolor='black')
    
    # Set labels and title
    plt.xlabel('X Position (m)', fontsize=14)
    plt.ylabel('Y Position (m)', fontsize=14)
    if title:
        plt.title(title, fontsize=16, pad=20)
    
    # Equal aspect ratio
    plt.axis('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        # Save without sRGB profile to fix libpng warning
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png', 
                   metadata={'icc_profile': None})
    
    if show:
        plt.show()
    else:
        plt.close()

def visualize_predictions(model, data_loader, config, num_samples=5, save_dir='visualization_results'):
    """
    Visualize predictions for multiple samples from the dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader containing test/validation data
        config: Configuration dictionary
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualization results
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(save_dir, f'vis_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get normalization parameters
    ego_mean = data_loader.dataset.future_ego_mean.to(device)
    ego_std = data_loader.dataset.future_ego_std.to(device)
    ego_history_mean = data_loader.dataset.history_ego_mean.to(device)
    ego_history_std = data_loader.dataset.history_ego_std.to(device)
    agent_mean = data_loader.dataset.future_agent_mean.to(device)
    agent_std = data_loader.dataset.future_agent_std.to(device)
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
                
            # Move data to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get predictions
            outputs = model(batch)
            
            # Denormalize trajectories
            ego_pred = outputs['ego_trajectories'].cpu().numpy()
            ego_pred = ego_pred * ego_std.cpu().numpy() + ego_mean.cpu().numpy()
            agent_pred = outputs['agent_trajectories'].cpu().numpy()
            agent_pred = agent_pred * agent_std.cpu().numpy() + agent_mean.cpu().numpy()
            
            # Get ground truth and history
            ego_history = batch['ego_history'].cpu().numpy()
            ego_history = ego_history * ego_history_std.cpu().numpy() + ego_history_mean.cpu().numpy()
            
            ego_future = batch['ego_future'].cpu().numpy()
            ego_future = ego_future * ego_std.cpu().numpy() + ego_mean.cpu().numpy()
            
            agent_history = batch['agent_history'].cpu().numpy()
            agent_future = batch['agent_future'].cpu().numpy()
            agent_future = agent_future * agent_std.cpu().numpy() + agent_mean.cpu().numpy()
            
            # Get agent mask
            # agent_history_mask = batch['agent_history_mask'].cpu().numpy()
            agent_future_mask = batch['agent_future_mask'].cpu().numpy()
            
            
            # Plot for each sample in the batch
            for j in range(min(ego_pred.shape[0], 1)):  # Plot only first sample from batch
                plot_trajectories(
                    ego_history[j],
                    ego_future[j],
                    ego_pred[j],
                    agent_history[j],
                    agent_future[j],
                    agent_pred[j],
                    # agent_history_mask=agent_history_mask[j],
                    agent_future_mask = agent_future_mask[j],
                    save_path=os.path.join(save_dir, f'sample_{i}_{j}.png'),
                    show=False,
                    title=f'Sample {i}_{j} Trajectories'
                )

def visualize_sequence(model, data_loader, config, sequence_length=10, save_dir='visualization_results'):
    """
    Visualize predictions for a sequence of frames.
    
    Args:
        model: Trained model
        data_loader: DataLoader containing test/validation data
        config: Configuration dictionary
        sequence_length: Number of consecutive frames to visualize
        save_dir: Directory to save visualization results
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(save_dir, f'sequence_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get normalization parameters
    ego_mean = data_loader.dataset.future_ego_mean.to(device)
    ego_std = data_loader.dataset.future_ego_std.to(device)
    ego_history_mean = data_loader.dataset.history_ego_mean.to(device)
    ego_history_std = data_loader.dataset.history_ego_std.to(device)
    agent_mean = data_loader.dataset.future_agent_mean.to(device)
    agent_std = data_loader.dataset.future_agent_std.to(device)
    
    # Get a sequence of frames
    sequence_data = []
    for i, batch in enumerate(data_loader):
        if i >= sequence_length:
            break
        sequence_data.append(batch)
    
    with torch.no_grad():
        for i, batch in enumerate(sequence_data):
            # Move data to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get predictions
            outputs = model(batch)
            
            # Denormalize trajectories
            ego_pred = outputs['ego_trajectories'].cpu().numpy()
            ego_pred = ego_pred * ego_std.cpu().numpy() + ego_mean.cpu().numpy()
            
            # Get ground truth and history
            ego_history = batch['ego_history'].cpu().numpy()
            ego_history = ego_history * ego_history_std.cpu().numpy() + ego_history_mean.cpu().numpy()
            
            ego_future = batch['ego_future'].cpu().numpy()
            ego_future = ego_future * ego_std.cpu().numpy() + ego_mean.cpu().numpy()
            
            agent_history = batch['agent_history'].cpu().numpy()
            agent_future = batch['agent_future'].cpu().numpy()
            agent_future = agent_future * agent_std.cpu().numpy() + agent_mean.cpu().numpy()
            
            # Get agent mask
            agent_mask = batch['agent_mask'].cpu().numpy()
            
            # Plot for each sample in the batch
            for j in range(min(ego_pred.shape[0], 1)):  # Plot only first sample from batch
                plot_trajectories(
                    ego_history[j],
                    ego_future[j],
                    ego_pred[j],
                    agent_history[j],
                    agent_future[j],
                    agent_future[j],
                    agent_mask=agent_mask[j],
                    save_path=os.path.join(save_dir, f'frame_{i:03d}.png'),
                    show=False,
                    title=f'Frame {i} Trajectories'
                ) 
                