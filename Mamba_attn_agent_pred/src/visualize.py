import os
import torch
from torch.utils.data import DataLoader
import argparse

from models.mamba_wm_model import MambaWorldModel
from datasets.trajectory_dataset import TrajectoryDataset
from utils.visualization import visualize_predictions, visualize_sequence

def main():
    # 直接设置参数，不需要命令行输入
    model_path = r'checkpoints\exp_20250528_101313\best_model.pth'  # 模型路径
    data_path = r'src\datasets\data\agent_mask_train_data\val_trajectories_with_mask.pkl'  # 数据路径
    save_dir = 'visualization_results'  # 保存目录
    num_samples = 5  # 可视化样本数
    sequence_length = 1  # 序列长度
    batch_size = 1  # batch大小
    
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
    }
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaWorldModel(config)
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create dataset and data loader
    dataset = TrajectoryDataset(data_path)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Set to 0 for visualization
    )
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # # Visualize individual samples
    # print("Visualizing individual samples...")
    visualize_predictions(
        model=model,
        data_loader=data_loader,
        config=config,
        num_samples=num_samples,
        save_dir=os.path.join(save_dir, 'samples')
    )
    
    # Visualize sequence
    print("Visualizing sequence...")
    # visualize_sequence(
    #     model=model,
    #     data_loader=data_loader,
    #     config=config,
    #     sequence_length=sequence_length,
    #     save_dir=os.path.join(save_dir, 'sequence')
    # )
    
    # print(f"Visualization results saved to {save_dir}")

if __name__ == '__main__':
    main() 