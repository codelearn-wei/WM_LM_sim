import torch
import torch.nn as nn
from typing import Dict, Any
from registry.registry import MODELS
from models.base_model import BaseModel
from models.modules.encoder import TrajectoryEncoder
from models.modules.decoder import TrajectoryDecoder

@MODELS.register_module()
class MambaWorldModel(BaseModel):
    """Mamba-based World Model for trajectory prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model configuration
        self.input_dim = config.get('input_dim', 7)  # 7 features for history
        self.output_dim = config.get('output_dim', 3)  # [x, y, heading] for future
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_modes = config.get('num_modes', 3)  # Number of prediction modes for ego vehicle
        self.prediction_horizon = config.get('prediction_horizon', 30)  # 3s future
        self.num_agents = config.get('num_agents', 6)  # Fixed number of agents
        self.history_len = config.get('history_len', 10)  # 1s history
        
        # Map configuration
        self.map_feature_dim = config.get('map_feature_dim', None)
        self.use_map = config.get('use_map', False)
        
        # Encoder for all agents (including ego)
        self.trajectory_encoder = TrajectoryEncoder(
            in_features=self.input_dim,
            d_model=self.hidden_dim,
            map_feature_dim=self.map_feature_dim,
            use_map=self.use_map,
            num_attn_layers=config.get('num_attn_layers', 2)
        )
        
        # Decoder for ego vehicle (multi-modal)
        self.ego_decoder = TrajectoryDecoder(
            d_model=self.hidden_dim,
            pred_len=self.prediction_horizon,
            output_dim=self.output_dim,
            num_modes=self.num_modes,
            predict_covariance=True  # Enable covariance prediction
        )
        
        # Decoder for other agents (single-modal)
        self.agent_decoder = TrajectoryDecoder(
            d_model=self.hidden_dim,
            pred_len=self.prediction_horizon,
            output_dim=self.output_dim,
            num_modes=1,  # Single mode for agents
            predict_covariance=False  # No covariance for agents
        )
        
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Dictionary containing:
                - ego_history: Ego vehicle history [batch_size, history_len, input_dim]
                - agent_history: Other agents history [batch_size, history_len, num_agents, input_dim]
                - agent_mask: Mask for valid agents [batch_size, num_agents]
                - map_features: Map features [batch_size, num_map_features, map_feature_dim] (optional)
        
        Returns:
            Dictionary containing:
                - mode_logits: Mode prediction logits for ego vehicle [batch_size, num_modes]
                - ego_trajectories: Predicted ego trajectories for each mode [batch_size, num_modes, pred_horizon, output_dim]
                - ego_sigmas: Predicted covariance matrices for ego vehicle [batch_size, num_modes, pred_horizon, output_dim, output_dim]
                - agent_trajectories: Predicted agent trajectories [batch_size, num_agents, pred_horizon, output_dim]
        """
        batch_size = x['ego_history'].size(0)
        
        # Combine ego and agent histories
        # Add ego vehicle as the first agent
        combined_history = torch.cat([
            x['ego_history'].unsqueeze(2),  # [batch_size, history_len, 1, input_dim]
            x['agent_history']  # [batch_size, history_len, num_agents, input_dim]
        ], dim=2)  # [batch_size, history_len, num_agents+1, input_dim]
        
        # Get map features if available
        map_features = x.get('map_features', None)
        
        # Encode all trajectories with map features
        encoded_features = self.trajectory_encoder(combined_history, map_features)  # [batch_size, history_len, num_agents+1, hidden_dim]
        
        # Get encoded features for ego and agents
        ego_features = encoded_features[:, :, 0]  # [batch_size, history_len, hidden_dim]
        agent_features = encoded_features[:, :, 1:]  # [batch_size, history_len, num_agents, hidden_dim]
        
        # Decode ego trajectories (multi-modal)
        ego_trajectories, ego_confidences, ego_sigmas = self.ego_decoder(ego_features.unsqueeze(2))
        ego_trajectories = ego_trajectories.squeeze(1)  # [batch_size, num_modes, pred_horizon, output_dim]
        ego_confidences = ego_confidences.squeeze(1)  # [batch_size, num_modes]
        ego_sigmas = ego_sigmas.squeeze(1)  # [batch_size, num_modes, pred_horizon, output_dim, output_dim]
        
        # Decode agent trajectories (single-modal)
        agent_trajectories, agent_confidences, _ = self.agent_decoder(agent_features)  # [batch_size, num_agents, 1, pred_horizon, output_dim]
        agent_trajectories = agent_trajectories.squeeze(2)  # [batch_size, num_agents, pred_horizon, output_dim]
        
        # Apply agent mask to trajectories
        agent_mask = x['agent_mask'].unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_agents, 1, 1]
        agent_trajectories = agent_trajectories * agent_mask
        
        return {
            'mode_logits': ego_confidences,
            'ego_trajectories': ego_trajectories,
            'ego_sigmas': ego_sigmas,
            'agent_trajectories': agent_trajectories
        }
    


if __name__ == '__main__':
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 创建模型配置
    config = {
        'input_dim': 7,  # 7 features for history
        'output_dim': 3,  # [x, y, heading] for future
        'hidden_dim': 256,
        'num_modes': 3,
        'prediction_horizon': 30,  # 3s future
        'num_agents': 6,  # Fixed number of agents
        'history_len': 10,  # 1s history
        'num_attn_layers': 2
    }
    
    # 创建模型实例
    model = MambaWorldModel(config)
    print("Model created successfully!")
    
    # 创建测试数据
    batch_size = 2
    
    # 创建输入数据
    x = {
        'ego_history': torch.randn(batch_size, config['history_len'], config['input_dim']),
        'agent_history': torch.randn(batch_size, config['history_len'], config['num_agents'], config['input_dim']),
        'agent_mask': torch.ones(batch_size, config['num_agents'])
    }
    
    print("\nInput shapes:")
    print(f"ego_history: {x['ego_history'].shape}")
    print(f"agent_history: {x['agent_history'].shape}")
    print(f"agent_mask: {x['agent_mask'].shape}")
    
    # 前向传播
    try:
        outputs = model(x)
        print("\nOutput shapes:")
        print(f"mode_logits: {outputs['mode_logits'].shape}")
        print(f"ego_trajectories: {outputs['ego_trajectories'].shape}")
        print(f"ego_sigmas: {outputs['ego_sigmas'].shape}")
        print(f"agent_trajectories: {outputs['agent_trajectories'].shape}")
    except Exception as e:
        print(f"\nError during forward pass: {str(e)}")
        raise e
    
    # 创建目标数据
    target = {
        'mode_idx': torch.randint(0, config['num_modes'], (batch_size,)),
        'ego_future': torch.randn(batch_size, config['prediction_horizon'], config['output_dim']),
        'agent_future': torch.randn(batch_size, config['num_agents'], config['prediction_horizon'], config['output_dim']),
        'agent_mask': x['agent_mask']
    }
    
    print("\nTarget shapes:")
    print(f"mode_idx: {target['mode_idx'].shape}")
    print(f"ego_future: {target['ego_future'].shape}")
    print(f"agent_future: {target['agent_future'].shape}")
    
