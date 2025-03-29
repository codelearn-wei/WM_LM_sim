import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import json

class BEVDataset(Dataset):
    def __init__(self, data_dir, history_steps=10, action_dim=30):
        """
        初始化BEV数据集
        
        Args:
            data_dir (str): 数据目录路径
            history_steps (int): 历史帧数
            action_dim (int): 动作维度
        """
        self.data_dir = data_dir
        self.history_steps = history_steps
        self.action_dim = action_dim
        
        # 获取所有场景目录
        self.scene_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # 加载场景数据
        self.scenes = []
        for scene_dir in self.scene_dirs:
            scene_path = os.path.join(data_dir, scene_dir)
            # 加载场景配置文件
            config_path = os.path.join(scene_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.scenes.append({
                        'dir': scene_path,
                        'config': config,
                        'num_frames': len([f for f in os.listdir(scene_path) if f.endswith('.png')])
                    })
        
        # 计算总样本数
        self.total_samples = sum(scene['num_frames'] - history_steps for scene in self.scenes)
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (bev_frames, actions, next_frame)
                - bev_frames: 历史BEV图像 [history_steps, C, H, W]
                - actions: 历史动作 [history_steps, action_dim]
                - next_frame: 下一帧BEV图像 [C, H, W]
        """
        # 找到对应的场景和帧索引
        current_idx = idx
        scene_idx = 0
        frame_idx = 0
        
        for scene in self.scenes:
            frames_in_scene = scene['num_frames'] - self.history_steps
            if current_idx < frames_in_scene:
                frame_idx = current_idx
                break
            current_idx -= frames_in_scene
            scene_idx += 1
        
        scene = self.scenes[scene_idx]
        scene_dir = scene['dir']
        
        # 加载历史帧
        bev_frames = []
        for t in range(self.history_steps):
            frame_path = os.path.join(scene_dir, f'frame_{frame_idx + t}.png')
            frame = Image.open(frame_path).convert('RGB')
            frame = np.array(frame) / 255.0  # 归一化到[0, 1]
            frame = torch.from_numpy(frame).permute(2, 0, 1).float()
            bev_frames.append(frame)
        
        # 加载下一帧
        next_frame_path = os.path.join(scene_dir, f'frame_{frame_idx + self.history_steps}.png')
        next_frame = Image.open(next_frame_path).convert('RGB')
        next_frame = np.array(next_frame) / 255.0
        next_frame = torch.from_numpy(next_frame).permute(2, 0, 1).float()
        
        # 加载动作数据
        actions = []
        for t in range(self.history_steps):
            action_path = os.path.join(scene_dir, f'action_{frame_idx + t}.npy')
            if os.path.exists(action_path):
                action = np.load(action_path)
                actions.append(action)
            else:
                # 如果没有动作数据，使用零向量
                actions.append(np.zeros(self.action_dim))
        
        # 转换为张量
        bev_frames = torch.stack(bev_frames)
        actions = torch.tensor(actions, dtype=torch.float32)
        
        return bev_frames, actions, next_frame 