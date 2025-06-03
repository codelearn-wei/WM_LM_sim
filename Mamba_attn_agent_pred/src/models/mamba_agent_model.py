import torch
import torch.nn as nn
from typing import Dict, Any
from registry.registry import MODELS
from models.base_model import BaseModel
from models.modules.encoder import TrajectoryEncoder
from models.modules.decoder import TrajectoryDecoder

@MODELS.register_module()
class MambaAgentModel(BaseModel):
    """Mamba-based Agent Model for trajectory prediction with mask support."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model configuration
        self.input_dim = config.get('input_dim', 7)  # 7 features for history
        self.output_dim = config.get('output_dim', 3)  # [x, y, heading] for future
        self.hidden_dim = config.get('hidden_dim', 256)
        self.prediction_horizon = config.get('prediction_horizon', 30)  # 3s future
        self.num_agents = config.get('num_agents', 6)  # Fixed number of agents
        self.history_len = config.get('history_len', 10)  # 1s history
        
        # Ensure history_len is divisible by 4 for Mamba block_len
        if self.history_len % 4 != 0:
            self.history_len = (self.history_len // 4 + 1) * 4
            print(f"Warning: history_len adjusted to {self.history_len} to be divisible by 4")
        
        # Encoder for all agents
        self.trajectory_encoder = TrajectoryEncoder(
            in_features=self.input_dim,
            d_model=self.hidden_dim,
            num_attn_layers=config.get('num_attn_layers', 2)
        )
        
        # Decoder for agents (single-modal)
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
                - agent_history: Agent history trajectories [batch_size, history_len, num_agents, input_dim]
                - agent_history_mask: Mask for valid agents in history [batch_size, history_len, num_agents]
                - agent_future_mask: Mask for valid agents in future [batch_size, prediction_horizon, num_agents]
        
        Returns:
            Dictionary containing:
                - agent_trajectories: Predicted agent trajectories [batch_size, num_agents, pred_horizon, output_dim]
        """
        batch_size = x['agent_history'].size(0)
        
        # Pad history if necessary
        current_history_len = x['agent_history'].size(1)
        if current_history_len < self.history_len:
            pad_len = self.history_len - current_history_len
            x['agent_history'] = torch.nn.functional.pad(
                x['agent_history'], 
                (0, 0, 0, 0, 0, pad_len, 0, 0)
            )
            x['agent_history_mask'] = torch.nn.functional.pad(
                x['agent_history_mask'],
                (0, 0, 0, pad_len, 0, 0)
            )
        
        # Encode agent trajectories
        encoded_features = self.trajectory_encoder(x['agent_history'])  # [batch_size, history_len, num_agents, hidden_dim]
        
        # Apply history mask to encoded features
        history_mask = x['agent_history_mask'].unsqueeze(-1)  # [batch_size, history_len, num_agents, 1]
        encoded_features = encoded_features * history_mask
        
        # Decode agent trajectories
        agent_trajectories, _, _ = self.agent_decoder(encoded_features)  # [batch_size, num_agents, 1, pred_horizon, output_dim]
        agent_trajectories = agent_trajectories.squeeze(2)  # [batch_size, num_agents, pred_horizon, output_dim]
        
        # Apply future mask to trajectories
        future_mask = x['agent_future_mask'].unsqueeze(-1)  # [batch_size, prediction_horizon, num_agents, 1]
        future_mask = future_mask.permute(0, 2, 1, 3)  # [batch_size, num_agents, prediction_horizon, 1]
        agent_trajectories = agent_trajectories * future_mask
        
        return {
            'agent_trajectories': agent_trajectories
        }

if __name__ == '__main__':
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 创建模型配置
    config = {
        'input_dim': 7,  # 7 features for history
        'output_dim': 3,  # [x, y, heading] for future
        'hidden_dim': 512,
        'prediction_horizon': 30,  # 3s future
        'num_agents': 6,  # Fixed number of agents
        'history_len': 10,  # 1s history
        'num_attn_layers': 3
    }
    
    # 创建模型实例
    model = MambaAgentModel(config)
    print("Model created successfully!")
    
    # 创建测试数据
    batch_size = 2
    
    # 创建输入数据
    x = {
        'agent_history': torch.randn(batch_size, config['history_len'], config['num_agents'], config['input_dim']),
        'agent_history_mask': torch.ones(batch_size, config['history_len'], config['num_agents']),
        'agent_future_mask': torch.ones(batch_size, config['prediction_horizon'], config['num_agents'])
    }
    
    print("\nInput shapes:")
    print(f"agent_history: {x['agent_history'].shape}")
    print(f"agent_history_mask: {x['agent_history_mask'].shape}")
    print(f"agent_future_mask: {x['agent_future_mask'].shape}")
    
    # 前向传播
    try:
        outputs = model(x)
        print("\nOutput shapes:")
        print(f"agent_trajectories: {outputs['agent_trajectories'].shape}")
    except Exception as e:
        print(f"\nError during forward pass: {str(e)}")
        raise e
    
    # 创建目标数据
    target = {
        'agent_future': torch.randn(batch_size, config['num_agents'], config['prediction_horizon'], config['output_dim']),
        'agent_future_mask': x['agent_future_mask']
    }
    
    print("\nTarget shapes:")
    print(f"agent_future: {target['agent_future'].shape}")
    print(f"agent_future_mask: {target['agent_future_mask'].shape}") 