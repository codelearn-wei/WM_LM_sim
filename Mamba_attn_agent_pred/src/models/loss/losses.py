import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class TrajectoryLoss(nn.Module):
    """Single-modal trajectory prediction loss."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target, mask=None):
        """
        Calculate trajectory prediction loss.
        
        Args:
            pred (torch.Tensor): Predicted trajectories [batch_size, pred_horizon, 3]
            target (torch.Tensor): Target trajectories [batch_size, pred_horizon, 3]
            mask (torch.Tensor, optional): Mask for valid timesteps [batch_size, pred_horizon]
            
        Returns:
            torch.Tensor: Mean loss value
        """
        # Calculate MSE loss
        loss = self.mse_loss(pred, target)  # [batch_size, pred_horizon, 3]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            loss = loss * mask
        
        # Calculate mean loss
        return loss.mean()

class MultiModalLoss(nn.Module):
    """Multi-modal trajectory prediction loss for ego vehicle."""
    
    def __init__(self, alpha=0.1):
        """
        Initialize the multi-modal loss.
        
        Args:
            alpha (float): Weight for the mode classification loss
        """
        super().__init__()
        self.trajectory_loss = TrajectoryLoss()
        self.alpha = alpha
        
    def calculate_gmm_nll(self, pred_probs, pred_mu, pred_sigma, y_true):
        """
        Calculate Gaussian Mixture Model negative log-likelihood loss.
        
        Args:
            pred_probs (torch.Tensor): Predicted mode probabilities [batch_size, num_modes]
            pred_mu (torch.Tensor): Predicted means [batch_size, num_modes, pred_horizon, 3]
            pred_sigma (torch.Tensor): Predicted covariances [batch_size, num_modes, pred_horizon, 3, 3]
            y_true (torch.Tensor): Target trajectories [batch_size, pred_horizon, 3]
            
        Returns:
            tuple: (total_nll_loss, selected_trajectory)
        """
        batch_size, M, T, D = pred_mu.size()
        
        # Step 1: Calculate log probability density for each mode
        nll_loss = []
        for i in range(M):
            dist_normal = dist.MultivariateNormal(
                loc=pred_mu[:, i],  # [batch_size, T, D]
                covariance_matrix=pred_sigma[:, i]  # [batch_size, T, D, D]
            )
            log_probs = dist_normal.log_prob(y_true)  # [batch_size, T]
            nll_loss.append(log_probs)
        nll_loss = torch.stack(nll_loss, dim=1)  # [batch_size, M, T]
        
        # Step 2: Select best mode based on Euclidean distance
        euclidean_dist = ((pred_mu - y_true.unsqueeze(1)) ** 2).sum(dim=-1)  # [batch_size, M, T]
        correct_mode = euclidean_dist.sum(dim=-1).argmin(dim=1)  # [batch_size]
        
        # Step 3: Calculate NLL loss for selected mode
        selected_nll_loss = nll_loss[torch.arange(batch_size), correct_mode]  # [batch_size, T]
        total_nll_loss = -torch.mean(
            selected_nll_loss.sum(dim=-1) + 
            torch.log(pred_probs[torch.arange(batch_size), correct_mode])
        )
        
        # Step 4: Get selected trajectory
        selected_trajectory = pred_mu[torch.arange(batch_size), correct_mode]  # [batch_size, T, D]
        
        return total_nll_loss, selected_trajectory
        
    def forward(self, pred_trajectories, pred_confidences, pred_sigmas, target_trajectories, target_mask=None):
        """
        Calculate multi-modal trajectory prediction loss.
        
        Args:
            pred_trajectories (torch.Tensor): Predicted trajectories [batch_size, num_modes, pred_horizon, 3]
            pred_confidences (torch.Tensor): Predicted mode confidences [batch_size, num_modes]
            pred_sigmas (torch.Tensor): Predicted covariances [batch_size, num_modes, pred_horizon, 3, 3]
            target_trajectories (torch.Tensor): Target trajectories [batch_size, pred_horizon, 3]
            target_mask (torch.Tensor, optional): Mask for valid timesteps [batch_size, pred_horizon]
            
        Returns:
            tuple: (total_loss, traj_loss, mode_loss)
        """
        # Calculate GMM loss
        gmm_loss, selected_trajectory = self.calculate_gmm_nll(
            pred_confidences,
            pred_trajectories,
            pred_sigmas,
            target_trajectories
        )
        
        # Calculate L2 loss for selected trajectory
        l2_loss = self.trajectory_loss(selected_trajectory, target_trajectories, target_mask)
        
        # Calculate mode classification loss
        mode_probs = F.softmax(pred_confidences, dim=1)  # [batch_size, num_modes]
        mode_loss = -(mode_probs * torch.log(mode_probs + 1e-8)).sum(dim=1).mean()
        
        # Combine losses
        total_loss = l2_loss + self.alpha * mode_loss
        
        return total_loss, l2_loss, mode_loss

def compute_total_loss(ego_outputs, agent_outputs, targets, config):
    """
    Compute total loss for ego and agent predictions.
    
    Args:
        ego_outputs (dict): Ego vehicle outputs containing:
            - ego_trajectories: [batch_size, num_modes, pred_horizon, 3]
            - mode_logits: [batch_size, num_modes]
            - ego_sigmas: [batch_size, num_modes, pred_horizon, 3, 3]
        agent_outputs (dict): Agent outputs containing:
            - agent_trajectories: [batch_size, num_agents, pred_horizon, 3]
        targets (dict): Target trajectories containing:
            - ego_future: [batch_size, pred_horizon, 3]
            - agent_future: [batch_size, num_agents, pred_horizon, 3]
            - agent_mask: [batch_size, num_agents]
        config (dict): Configuration dictionary containing:
            - loss_alpha: Weight for mode classification loss
            
    Returns:
        tuple: (total_loss, ego_loss, agent_loss)
    """
    # Initialize loss functions
    ego_loss_fn = MultiModalLoss(alpha=config['loss_alpha'])
    agent_loss_fn = TrajectoryLoss()
    
    # Calculate ego vehicle loss (multi-modal)
    ego_total_loss, ego_traj_loss, ego_mode_loss = ego_loss_fn(
        ego_outputs['ego_trajectories'],
        ego_outputs['mode_logits'],
        ego_outputs['ego_sigmas'],
        targets['ego_future']
    )
    
    # Calculate agent losses (single-modal)
    agent_losses = []
    for agent_idx in range(agent_outputs['agent_trajectories'].size(1)):
        # Get trajectories for current agent
        agent_trajectories = agent_outputs['agent_trajectories'][:, agent_idx]  # [batch_size, pred_horizon, 3]
        agent_target = targets['agent_future'][:, :, agent_idx]  # [batch_size, pred_horizon, 3]
        agent_mask = targets['agent_mask'][:, agent_idx]  # [batch_size]
        
        # Calculate loss for this agent
        agent_loss = agent_loss_fn(agent_trajectories, agent_target, agent_mask)
        agent_losses.append(agent_loss)
    
    # Average agent losses
    agent_loss = torch.stack(agent_losses).mean()
    
    # Combine total loss
    total_loss = ego_total_loss + agent_loss
    
    return total_loss, ego_total_loss, agent_loss 