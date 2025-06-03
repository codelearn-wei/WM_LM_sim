import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(data_path, val_ratio=0.2, random_state=42):
    """
    Split the dataset into training and validation sets.
    
    Args:
        data_path (str): Path to the original dataset pickle file
        val_ratio (float): Ratio of validation set (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (train_data, val_data) containing the split datasets
    """
    # Load the original dataset
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Split the data
    train_data, val_data = train_test_split(
        data,
        test_size=val_ratio,
        random_state=random_state
    )
    
    # Create data directory if it doesn't exist
    data_dir = os.path.dirname(data_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save the split datasets
    train_path = os.path.join(data_dir, 'train_trajectories_ego.pkl')
    val_path = os.path.join(data_dir, 'val_trajectories_ego.pkl')
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)
    
    print(f"Dataset split complete:")
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Training set saved to: {train_path}")
    print(f"Validation set saved to: {val_path}")
    
    return train_data, val_data

if __name__ == '__main__':
    # 设置数据路径
    data_path = r'src\datasets\data\train_trajectories_ego.pkl'
    
    # 划分数据集
    split_dataset(data_path, val_ratio=0.2) 