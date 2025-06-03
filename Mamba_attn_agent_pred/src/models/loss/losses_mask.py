import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedTrajectoryLoss(nn.Module):
    """Loss function for masked trajectory prediction."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, pred_trajectories, gt_trajectories, future_mask):
        """
        Calculate masked trajectory prediction loss.
        
        Args:
            pred_trajectories (torch.Tensor): Predicted trajectories [batch_size, num_agents, pred_horizon, output_dim]
            gt_trajectories (torch.Tensor): Ground truth trajectories [batch_size, num_agents, pred_horizon, output_dim]
            future_mask (torch.Tensor): Mask for valid predictions [batch_size, pred_horizon, num_agents]
            
        Returns:
            torch.Tensor: Mean loss value
        """
        # Reshape mask for broadcasting
        future_mask = future_mask.permute(0, 2, 1).unsqueeze(-1)  # [batch_size, num_agents, pred_horizon, 1]
        
        # Compute L2 loss for each dimension
        gt_trajectories = gt_trajectories.permute(0, 2, 1, 3)
        l2_loss = self.mse_loss(pred_trajectories, gt_trajectories)  # [batch_size, num_agents, pred_horizon, output_dim]
        
        # Apply mask to loss
        masked_loss = l2_loss * future_mask
        
        # Compute mean over valid predictions
        num_valid = future_mask.sum() + 1e-8  # Add small epsilon to avoid division by zero
        total_loss = masked_loss.sum() / num_valid
        
        return total_loss



def compute_total_loss(outputs: dict, targets: dict, config: dict) -> torch.Tensor:
    """
    Compute the total loss for masked trajectory prediction.
    
    Args:
        outputs: Dictionary containing model outputs
            - agent_trajectories: [batch_size, num_agents, pred_horizon, output_dim]
        targets: Dictionary containing ground truth
            - agent_future: [batch_size, num_agents, pred_horizon, output_dim]
            - agent_future_mask: [batch_size, pred_horizon, num_agents]
        config: Configuration dictionary containing loss weights
    
    Returns:
        torch.Tensor: Total loss value
    """
    loss_fn = MaskedTrajectoryLoss()
    return loss_fn(
        outputs['agent_trajectories'],
        targets['agent_future'],
        targets['agent_future_mask']
    )

