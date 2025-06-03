# 解码成预测需要的状态
# 解码需要是主车的多模态轨迹和每辆车的绝对轨迹
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryDecoder(nn.Module):
    def __init__(self, d_model, pred_len, output_dim=3, num_modes=3, predict_covariance=False):
        """
        Initialize trajectory decoder.
        
        Args:
            d_model (int): Input feature dimension
            pred_len (int): Prediction horizon length
            output_dim (int): Output dimension (default: 3 for [x, y, heading])
            num_modes (int): Number of prediction modes
            predict_covariance (bool): Whether to predict covariance matrices
        """
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.num_modes = num_modes
        self.predict_covariance = predict_covariance
        
        # Mode selection head
        self.mode_selection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_modes)
        )
        
        # Trajectory prediction heads (one for each mode)
        self.trajectory_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, pred_len * output_dim)
            ) for _ in range(num_modes)
        ])
        
        # Covariance prediction heads (one for each mode)
        if predict_covariance:
            self.covariance_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, pred_len * output_dim * output_dim)
                ) for _ in range(num_modes)
            ])
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, T, N, D) - Encoded features
            
        Returns:
            trajectories: (B, N, M, T_pred, output_dim) - Predicted trajectories
            confidences: (B, N, M) - Mode confidences
            sigmas: (B, N, M, T_pred, output_dim, output_dim) - Covariance matrices (if predict_covariance=True)
        """
        B, T, N, D = x.shape
        
        # Use last timestep as prediction input
        x_last = x[:, -1]  # (B, N, D)
        
        # Predict mode confidences
        confidences = self.mode_selection_head(x_last)  # (B, N, M)
        confidences = F.softmax(confidences, dim=-1)
        
        # Predict trajectories for each mode
        trajectories = []
        sigmas = []
        for i in range(self.num_modes):
            # Predict trajectory
            traj = self.trajectory_heads[i](x_last)  # (B, N, pred_len*output_dim)
            traj = traj.view(B, N, self.pred_len, -1)  # (B, N, pred_len, output_dim)
            trajectories.append(traj)
            
            # Predict covariance if enabled
            if self.predict_covariance:
                sigma = self.covariance_heads[i](x_last)  # (B, N, pred_len*output_dim*output_dim)
                sigma = sigma.view(B, N, self.pred_len, self.output_dim, self.output_dim)
                
                # Ensure positive definiteness
                sigma = torch.matmul(sigma, sigma.transpose(-2, -1))  # (B, N, pred_len, output_dim, output_dim)
                sigma = sigma + torch.eye(self.output_dim, device=sigma.device) * 1e-6  # Add small diagonal for stability
                sigmas.append(sigma)
        
        # Stack predictions
        trajectories = torch.stack(trajectories, dim=2)  # (B, N, M, pred_len, output_dim)
        
        if self.predict_covariance:
            sigmas = torch.stack(sigmas, dim=2)  # (B, N, M, pred_len, output_dim, output_dim)
            return trajectories, confidences, sigmas
        else:
            return trajectories, confidences, None