import torch
from torch.utils.data import Dataset
import pickle
import numpy as np


class TrajectoryDataset(Dataset):
    """Dataset for trajectory prediction task."""
    
    def __init__(self, data_path):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to the pickle file containing trajectory data
        """
        # Load data
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Pre-compute and store all data as tensors
        self._preprocess_data()
        
        # Compute normalization parameters
        self._compute_normalization_params()
        
    def _preprocess_data(self):
        """Preprocess and convert all data to tensors."""
        # Pre-allocate tensors
        num_samples = len(self.data)
        history_len = 10
        pred_horizon = 30
        num_agents = 6
        input_dim = 7
        output_dim = 3
        
        # Initialize tensors
        self.ego_history = torch.zeros((num_samples, history_len, input_dim), dtype=torch.float32)
        self.agent_history = torch.zeros((num_samples, history_len, num_agents, input_dim), dtype=torch.float32)
        self.ego_future = torch.zeros((num_samples, pred_horizon, output_dim), dtype=torch.float32)
        self.agent_future = torch.zeros((num_samples, pred_horizon, num_agents, output_dim), dtype=torch.float32)
        
        # self.agent_history_mask = torch.zeros((num_samples, history_len, num_agents, input_dim), dtype=torch.float32)
        self.agent_future_mask = torch.zeros((num_samples, pred_horizon, num_agents), dtype=torch.float32)
         
        
        # Fill tensors
        for i, item in enumerate(self.data):
            self.ego_history[i] = torch.FloatTensor(item['ego_history'][:history_len])
            self.agent_history[i] = torch.FloatTensor(item['agent_history'][:history_len])
            self.ego_future[i] = torch.FloatTensor(item['ego_future'])
            self.agent_future[i] = torch.FloatTensor(item['agent_future'])
            # self.agent_history_mask[i] =  torch.FloatTensor(item['agent_history_mask'])
            self.agent_future_mask[i] =  torch.FloatTensor(item['agent_future_mask'])
        
        # Clear original data to save memory
        self.data = None
        
    def _compute_normalization_params(self):
        """Compute normalization parameters from the dataset."""
        # Compute statistics for history data
        self.history_ego_mean = torch.mean(self.ego_history, dim=(0, 1))  # [input_dim]
        self.history_ego_std = torch.std(self.ego_history, dim=(0, 1))  # [input_dim]
        self.history_ego_std[self.history_ego_std < 1e-6] = 1.0  # Avoid division by zero
        
        self.history_agent_mean = torch.mean(self.agent_history, dim=(0, 1, 2))  # [input_dim]
        self.history_agent_std = torch.std(self.agent_history, dim=(0, 1, 2))  # [input_dim]
        self.history_agent_std[self.history_agent_std < 1e-6] = 1.0  # Avoid division by zero
        
        # Compute statistics for future data
        self.future_ego_mean = torch.mean(self.ego_future, dim=(0, 1))  # [output_dim]
        self.future_ego_std = torch.std(self.ego_future, dim=(0, 1))  # [output_dim]
        self.future_ego_std[self.future_ego_std < 1e-6] = 1.0  # Avoid division by zero
        
        self.future_agent_mean = torch.mean(self.agent_future, dim=(0, 1, 2))  # [output_dim]
        self.future_agent_std = torch.std(self.agent_future, dim=(0, 1, 2))  # [output_dim]
        self.future_agent_std[self.future_agent_std < 1e-6] = 1.0  # Avoid division by zero
        
    def _normalize_history(self, ego_data, agent_data):
        """
        Normalize history trajectory data.
        
        Args:
            ego_data (torch.Tensor): Ego vehicle history data [batch_size, seq_len, input_dim]
            agent_data (torch.Tensor): Agent history data [batch_size, seq_len, num_agents, input_dim]
            
        Returns:
            tuple: (normalized_ego_data, normalized_agent_data)
        """
        # Normalize ego data
        norm_ego = (ego_data - self.history_ego_mean) / self.history_ego_std
        
        # Normalize agent data
        norm_agent = (agent_data - self.history_agent_mean) / self.history_agent_std
        
        return norm_ego, norm_agent
    
    def _normalize_future(self, ego_data, agent_data):
        """
        Normalize future trajectory data.
        
        Args:
            ego_data (torch.Tensor): Ego vehicle future data [batch_size, seq_len, output_dim]
            agent_data (torch.Tensor): Agent future data [batch_size, seq_len, num_agents, output_dim]
            
        Returns:
            tuple: (normalized_ego_data, normalized_agent_data)
        """
        # Normalize ego data
        norm_ego = (ego_data - self.future_ego_mean) / self.future_ego_std
        
        # Normalize agent data
        norm_agent = (agent_data - self.future_agent_mean) / self.future_agent_std
        
        return norm_ego, norm_agent
    
    def _denormalize_future(self, norm_ego_data, norm_agent_data):
        """
        Denormalize future trajectory data.
        
        Args:
            norm_ego_data (torch.Tensor): Normalized ego vehicle future data
            norm_agent_data (torch.Tensor): Normalized agent future data
            
        Returns:
            tuple: (denormalized_ego_data, denormalized_agent_data)
        """
        # Denormalize ego data
        denorm_ego = norm_ego_data * self.future_ego_std + self.future_ego_mean
        
        # Denormalize agent data
        denorm_agent = norm_agent_data * self.future_agent_std + self.future_agent_mean
        
        return denorm_ego, denorm_agent
        
    def __len__(self):
        return len(self.ego_history)
    
    def __getitem__(self, idx):
        """
        Get a single data sample.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing:
                - ego_history: Normalized ego vehicle history [history_len, input_dim]
                - agent_history: Normalized other agents history [history_len, num_agents, input_dim]
                - ego_future: Normalized ego vehicle future [pred_horizon, output_dim]
                - agent_future: Normalized other agents future [pred_horizon, num_agents, output_dim]
                - agent_mask: Mask for valid agents [num_agents]
                - original_ego_future: Original (unnormalized) ego vehicle future [pred_horizon, output_dim]
                - original_agent_future: Original (unnormalized) other agents future [pred_horizon, num_agents, output_dim]
        """
        # Get data
        ego_history = self.ego_history[idx]
        agent_history = self.agent_history[idx]
        ego_future = self.ego_future[idx]
        agent_future = self.agent_future[idx]
        # agent_history_mask = self.agent_history_mask[idx]
        agent_future_mask = self.agent_future_mask[idx]
        
        
        # Store original data for metrics calculation
        original_ego_future = ego_future.clone()
        original_agent_future = agent_future.clone()
        
        # Normalize data
        ego_history, agent_history = self._normalize_history(
            ego_history.unsqueeze(0), 
            agent_history.unsqueeze(0)
        )
        ego_future, agent_future = self._normalize_future(
            ego_future.unsqueeze(0), 
            agent_future.unsqueeze(0)
        )
        
        return {
            'ego_history': ego_history.squeeze(0),
            'agent_history': agent_history.squeeze(0),
            'ego_future': ego_future.squeeze(0),
            'agent_future': agent_future.squeeze(0),
            'agent_mask': torch.ones(6),
            # 'agent_history_mask':agent_history_mask.squeeze(0),
             'agent_future_mask':agent_future_mask.squeeze(0),
            'original_ego_future': original_ego_future,
            'original_agent_future': original_agent_future
        }
    
    def denormalize_predictions(self, ego_pred, agent_pred):
        """
        Denormalize predicted trajectories.
        
        Args:
            ego_pred (torch.Tensor): Normalized ego predictions [batch_size, num_modes, pred_horizon, output_dim]
            agent_pred (torch.Tensor): Normalized agent predictions [batch_size, num_agents, pred_horizon, output_dim]
            
        Returns:
            tuple: (denormalized_ego_pred, denormalized_agent_pred)
        """
        return self._denormalize_future(ego_pred, agent_pred) 