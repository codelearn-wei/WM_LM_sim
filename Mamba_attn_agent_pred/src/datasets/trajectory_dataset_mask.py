import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class TrajectoryDatasetMask(Dataset):
    def __init__(self, data_path):
        """
        Initialize the trajectory dataset with masking support.
        
        Args:
            data_path (str): Path to the pickle file containing trajectory data with masks
        """
        # Load data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract trajectories and masks
        self.ego_history = torch.FloatTensor(np.stack([d['ego_history'] for d in data]))  # [num_samples, history_len, 7]
        self.ego_future = torch.FloatTensor(np.stack([d['ego_future'] for d in data]))    # [num_samples, pred_horizon, 3]
        self.agent_history = torch.FloatTensor(np.stack([d['agent_history'] for d in data]))  # [num_samples, num_agents, history_len, 7]
        self.agent_future = torch.FloatTensor(np.stack([d['agent_future'] for d in data]))    # [num_samples, num_agents, pred_horizon, 3]
        
        # Extract masks from data
        self.agent_history_mask = torch.BoolTensor(np.stack([d['agent_history_mask'] for d in data]))  # [num_samples, history_len, num_agents]
        self.agent_future_mask = torch.BoolTensor(np.stack([d['agent_future_mask'] for d in data]))    # [num_samples, pred_horizon, num_agents]
        
        # Calculate normalization parameters
        self._calculate_normalization_params()
        
        # Normalize data
        self._normalize_data()
    
    def _calculate_normalization_params(self):
        """Calculate normalization parameters for history and future trajectories."""
        # Ego vehicle normalization
        self.history_ego_mean = self.ego_history.mean(dim=(0, 1))  # [7]
        self.history_ego_std = self.ego_history.std(dim=(0, 1))    # [7]
        self.future_ego_mean = self.ego_future.mean(dim=(0, 1))    # [3]
        self.future_ego_std = self.ego_future.std(dim=(0, 1))      # [3]
        
        # Agent normalization (only consider valid agents using masks)
        # Reshape masks for broadcasting
        history_mask = self.agent_history_mask.unsqueeze(-1)  # [num_samples, history_len, num_agents, 1]
        future_mask = self.agent_future_mask.unsqueeze(-1)    # [num_samples, pred_horizon, num_agents, 1]
        
        # Calculate mean and std only for valid agents
        valid_history_sum = history_mask.sum()
        valid_future_sum = future_mask.sum()
        
        self.history_agent_mean = (self.agent_history * history_mask).sum(dim=(0, 1, 2)) / valid_history_sum
        self.history_agent_std = torch.sqrt(((self.agent_history - self.history_agent_mean) ** 2 * history_mask).sum(dim=(0, 1, 2)) / valid_history_sum)
        
        self.future_agent_mean = (self.agent_future * future_mask).sum(dim=(0, 1, 2)) / valid_future_sum
        self.future_agent_std = torch.sqrt(((self.agent_future - self.future_agent_mean) ** 2 * future_mask).sum(dim=(0, 1, 2)) / valid_future_sum)
    
    def _normalize_data(self):
        """Normalize the trajectory data."""
        # Normalize ego vehicle trajectories
        self.ego_history = (self.ego_history - self.history_ego_mean) / self.history_ego_std
        self.ego_future = (self.ego_future - self.future_ego_mean) / self.future_ego_std
        
        # Normalize agent trajectories
        self.agent_history = (self.agent_history - self.history_agent_mean) / self.history_agent_std
        self.agent_future = (self.agent_future - self.future_agent_mean) / self.future_agent_std
    
    def __len__(self):
        return len(self.ego_history)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing:
                - ego_history: Ego vehicle history trajectory [history_len, 7]
                - ego_future: Ego vehicle future trajectory [pred_horizon, 3]
                - agent_history: Agent history trajectories [num_agents, history_len, 7]
                - agent_future: Agent future trajectories [num_agents, pred_horizon, 3]
                - agent_history_mask: Mask for agent history trajectories [history_len, num_agents]
                - agent_future_mask: Mask for agent future trajectories [pred_horizon, num_agents]
        """
        return {
            'ego_history': self.ego_history[idx],
            'ego_future': self.ego_future[idx],
            'agent_history': self.agent_history[idx],
            'agent_future': self.agent_future[idx],
            'agent_history_mask': self.agent_history_mask[idx],
            'agent_future_mask': self.agent_future_mask[idx]
        }
    
    def denormalize_ego_history(self, normalized_data):
        """Denormalize ego vehicle history trajectory."""
        return normalized_data * self.history_ego_std + self.history_ego_mean
    
    def denormalize_ego_future(self, normalized_data):
        """Denormalize ego vehicle future trajectory."""
        return normalized_data * self.future_ego_std + self.future_ego_mean
    
    def denormalize_agent_history(self, normalized_data):
        """Denormalize agent history trajectories."""
        return normalized_data * self.history_agent_std + self.history_agent_mean
    
    def denormalize_agent_future(self, normalized_data):
        """Denormalize agent future trajectories."""
        return normalized_data * self.future_agent_std + self.future_agent_mean 