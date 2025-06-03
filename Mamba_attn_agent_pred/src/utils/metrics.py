import torch

def compute_ade_fde(pred_trajectories, gt_trajectories, mode='min'):
    """
    Compute Average Displacement Error (ADE) and Final Displacement Error (FDE).
    
    Args:
        pred_trajectories (torch.Tensor): Predicted trajectories 
            - For single mode: [batch_size, pred_horizon, 2]
            - For multi-mode: [batch_size, num_modes, pred_horizon, 2]
        gt_trajectories (torch.Tensor): Ground truth trajectories [batch_size, pred_horizon, 2]
        mode (str): 'min' for minimum error across modes, 'mean' for mean error across modes
        
    Returns:
        tuple: (ADE, FDE)
            - ADE: Average displacement error over all timesteps
            - FDE: Final displacement error at the last timestep
    """
    if len(pred_trajectories.shape) == 4:  # Multi-mode prediction
        batch_size, num_modes, pred_horizon, _ = pred_trajectories.shape
        
        # Compute L2 error for each mode
        error = torch.norm(pred_trajectories[..., :2] - gt_trajectories.unsqueeze(1)[..., :2], dim=-1)
        
        if mode == 'min':
            # Get minimum error across modes
            min_error, _ = torch.min(error, dim=1)
            ade = min_error.mean()
            fde = min_error[:, -1].mean()
        else:  # mode == 'mean'
            # Get mean error across modes
            mean_error = error.mean(dim=1)
            ade = mean_error.mean()
            fde = mean_error[:, -1].mean()
            
    else:  # Single-mode prediction
        error = torch.norm(pred_trajectories[..., :2] - gt_trajectories[..., :2], dim=-1)
        ade = error.mean()
        fde = error[:, -1].mean()
    
    return ade.item(), fde.item()

def compute_metrics(pred_trajectories, gt_trajectories, mode='min'):
    """
    Compute all trajectory prediction metrics.
    
    Args:
        pred_trajectories (torch.Tensor): Predicted trajectories 
            - For single mode: [batch_size, pred_horizon, 2]
            - For multi-mode: [batch_size, num_modes, pred_horizon, 2]
            - For agents: [batch_size, num_agents, pred_horizon, 2]
        gt_trajectories (torch.Tensor): Ground truth trajectories 
            - For ego: [batch_size, pred_horizon, 2]
            - For agents: [batch_size, num_agents, pred_horizon, 2]
        mode (str): 'min' for minimum error across modes, 'mean' for mean error across modes
        
    Returns:
        dict: Dictionary containing all metrics
            - ade: Average displacement error
            - fde: Final displacement error
            - ade_x: Average displacement error in x direction
            - ade_y: Average displacement error in y direction
            - fde_x: Final displacement error in x direction
            - fde_y: Final displacement error in y direction
    """
    if len(pred_trajectories.shape) == 4:  # Multi-mode prediction
        batch_size, num_modes, pred_horizon, _ = pred_trajectories.shape
        
        # Check if this is agent prediction (num_modes == num_agents)
        if num_modes == gt_trajectories.shape[1]:  # This is agent prediction
            # Compute error for each agent
            error = pred_trajectories[..., :2] - gt_trajectories[..., :2]
            
            # First compute metrics for each agent
            agent_ade = torch.norm(error, dim=-1).mean(dim=-1)  # [batch_size, num_agents]
            agent_fde = torch.norm(error[:, :, -1], dim=-1)     # [batch_size, num_agents]
            agent_ade_x = torch.abs(error[..., 0]).mean(dim=-1)  # [batch_size, num_agents]
            agent_ade_y = torch.abs(error[..., 1]).mean(dim=-1)  # [batch_size, num_agents]
            agent_fde_x = torch.abs(error[:, :, -1, 0])         # [batch_size, num_agents]
            agent_fde_y = torch.abs(error[:, :, -1, 1])         # [batch_size, num_agents]
            
            # Then take mean across all agents
            ade = agent_ade.mean().item()
            fde = agent_fde.mean().item()
            ade_x = agent_ade_x.mean().item()
            ade_y = agent_ade_y.mean().item()
            fde_x = agent_fde_x.mean().item()
            fde_y = agent_fde_y.mean().item()
            
        else:  # This is ego prediction with multiple modes
            # Compute error for each mode
            error = pred_trajectories[..., :2] - gt_trajectories.unsqueeze(1)[..., :2]
            
            if mode == 'min':
                # Get minimum error across modes
                error_norm = torch.norm(error, dim=-1)
                min_error_idx = torch.argmin(error_norm.mean(dim=-1), dim=1)
                min_error = error[torch.arange(batch_size), min_error_idx]
                
                # Compute metrics
                ade = torch.norm(min_error, dim=-1).mean().item()
                fde = torch.norm(min_error[:, -1], dim=-1).mean().item()
                
                # Direction-wise metrics
                ade_x = torch.abs(min_error[..., 0]).mean().item()
                ade_y = torch.abs(min_error[..., 1]).mean().item()
                fde_x = torch.abs(min_error[:, -1, 0]).mean().item()
                fde_y = torch.abs(min_error[:, -1, 1]).mean().item()
                
            else:  # mode == 'mean'
                # Get mean error across modes
                mean_error = error.mean(dim=1)
                
                # Compute metrics
                ade = torch.norm(mean_error, dim=-1).mean().item()
                fde = torch.norm(mean_error[:, -1], dim=-1).mean().item()
                
                # Direction-wise metrics
                ade_x = torch.abs(mean_error[..., 0]).mean().item()
                ade_y = torch.abs(mean_error[..., 1]).mean().item()
                fde_x = torch.abs(mean_error[:, -1, 0]).mean().item()
                fde_y = torch.abs(mean_error[:, -1, 1]).mean().item()
            
    else:  # Single-mode prediction
        error = pred_trajectories[..., :2] - gt_trajectories[..., :2]
        
        # Compute metrics
        ade = torch.norm(error, dim=-1).mean().item()
        fde = torch.norm(error[:, -1], dim=-1).mean().item()
        
        # Direction-wise metrics
        ade_x = torch.abs(error[..., 0]).mean().item()
        ade_y = torch.abs(error[..., 1]).mean().item()
        fde_x = torch.abs(error[:, -1, 0]).mean().item()
        fde_y = torch.abs(error[:, -1, 1]).mean().item()
    
    return {
        'ade': ade,
        'fde': fde,
        'ade_x': ade_x,
        'ade_y': ade_y,
        'fde_x': fde_x,
        'fde_y': fde_y
    } 