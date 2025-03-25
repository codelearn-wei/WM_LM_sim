import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np

class LMTrainingDataset(Dataset):
    def __init__(self, data_dir: str, history_steps: int = 20):
        """
        初始化训练数据集

        Args:
            data_dir (str): 数据目录路径
            history_steps (int, optional): 历史步长. 默认为 20.
        """
        self.data_dir = Path(data_dir)
        self.history_steps = history_steps
        self.sequences = self._build_sequences()
        
    def _build_sequences(self):
        """构建训练序列"""
        sequences = []
        
        # 遍历所有场景目录
        for scene_dir in self.data_dir.iterdir():
            if not scene_dir.is_dir():
                continue
                
            # 加载动作数据
            action_file = scene_dir / 'actions.npz'
            if not action_file.exists():
                continue
                
            action_data = np.load(action_file, allow_pickle=True)
            frame_actions = action_data['frame_actions'].item()
            
            # 获取所有帧
            frames = sorted(list(scene_dir.glob("frame_*.png")))
            frame_numbers = [int(f.stem.split('_')[1]) for f in frames]
            
            # 构建序列
            for i in range(len(frames) - self.history_steps - 1):
                start_frame = frame_numbers[i]
                end_frame = frame_numbers[i + self.history_steps]
                
                # 确保序列连续
                if end_frame - start_frame != self.history_steps:
                    continue
                
                seq = {
                    'scene_dir': scene_dir,
                    'start_frame': start_frame,
                    'frames': frames[i:i + self.history_steps + 1],
                    'actions': [frame_actions[j] for j in range(start_frame, end_frame + 1)]
                }
                sequences.append(seq)
        
        return sequences
    
    def _process_image(self, image_path):
        """处理图像"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """获取训练样本"""
        seq = self.sequences[idx]
        
        # 读取并处理BEV图像序列
        bev_frames = []
        for frame_path in seq['frames'][:-1]:  # 不包括最后一帧
            bev_frames.append(self._process_image(frame_path))
        bev_frames = torch.stack(bev_frames)
        
        # 读取下一帧
        next_frame = self._process_image(seq['frames'][-1])
        
        # 处理动作序列
        actions = []
        for frame_action in seq['actions'][:-1]:  # 不包括最后一帧的动作
            action_tensor = torch.from_numpy(frame_action).float()
            actions.append(action_tensor)
        actions = torch.stack(actions)
        
        return {
            'bev_frames': bev_frames,
            'actions': actions,
            'next_frame': next_frame
        }