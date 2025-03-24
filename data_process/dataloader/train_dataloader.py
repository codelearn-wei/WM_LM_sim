import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import typing

# TODO:每个场景选定一个主车，以相对主车的位置作为特征；

class TrafficDataset(Dataset):
    """
    交通轨迹数据集类，用于处理交通trajectory预测任务
    主要功能:
    - 支持可变长度的历史和未来时间步
    - 处理车辆数量约束
    - 计算车辆动作（加速度、航向角变化）
    - 将车道类型作为车辆特征的一部分
    """
    def __init__(
        self, 
        raw_data: typing.Dict[int, typing.List[dict]], 
        history_steps: int = 20,  # 历史时间步数 
        future_steps: int = 1,    # 预测未来时间步数
        max_vehicles: int = 20,   # 每帧最大车辆数
        max_aux_vehicles: int = 10  # 最大辅道车辆数
    ):
        """
        初始化交通数据集

        参数:
            raw_data: 原始交通数据，按时间戳组织的字典
            history_steps: 历史时间步数 (默认: 20)
            future_steps: 预测未来时间步数 (默认: 1)
            max_vehicles: 每帧最大车辆数 (默认: 20)
            max_aux_vehicles: 最大辅道车辆数 (默认: 10)
        """
        # 配置参数
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.max_vehicles = max_vehicles
        self.max_aux_vehicles = max_aux_vehicles
        
        # 验证输入数据
        if not raw_data:
            raise ValueError("输入数据不能为空")
        
        # 将原始数据转换为排序列表和DataFrame
        self.raw_data_list = [raw_data[t] for t in sorted(raw_data.keys())]
        self.data = pd.DataFrame([item for frame in self.raw_data_list for item in frame])
        
        # 按时间戳和车辆ID排序
        self.data.sort_values(by=['timestamp', 'track_id'], inplace=True)
        
        # 计算车辆动作
        self._compute_actions()
        
        # 构建数据集样本
        self.samples = self._build_samples()
        
        # 准备TensorDataset
        self._prepare_tensor_dataset()
    
    def _compute_actions(self):
        """
        计算车辆加速度和航向角变化
        使用NumPy向量化操作提高效率
        """
        data_np = self.data[['track_id', 'timestamp', 'vx', 'vy', 'psi_rad']].to_numpy()
        unique_ids = np.unique(data_np[:, 0])
        
        # 预分配数组
        ax = np.zeros(len(data_np))
        ay = np.zeros(len(data_np))
        delta_psi = np.zeros(len(data_np))
        
        # 每辆车计算动作
        for track_id in unique_ids:
            mask = data_np[:, 0] == track_id
            vx_series = data_np[mask, 2]
            vy_series = data_np[mask, 3]
            psi_series = data_np[mask, 4]
            
            ax[mask] = np.diff(vx_series, prepend=vx_series[0]) / 1.0
            ay[mask] = np.diff(vy_series, prepend=vy_series[0]) / 1.0
            delta_psi[mask] = np.diff(psi_series, prepend=psi_series[0])
        
        # 添加计算的动作到DataFrame
        self.data['ax'] = ax
        self.data['ay'] = ay
        self.data['delta_psi'] = delta_psi
        
        # 添加车道类型的数值表示
        self.data['is_lane_changing'] = (self.data['lane_type'] == '变道车辆').astype(float)
        self.data['is_main_lane'] = (self.data['lane_type'] != '变道车辆').astype(float)
    
    def _build_samples(self):
        """
        使用滑动窗口方法构建训练样本
        
        返回:
            包含历史轨迹、动作和未来轨迹的样本列表
        """
        samples = []
        timestamps = sorted(self.data['timestamp'].unique())
        
        # 滑动窗口生成样本
        for i in range(len(timestamps) - self.history_steps - self.future_steps + 1):
            window_timestamps = timestamps[i:i + self.history_steps + self.future_steps]
            window_data = self.data[self.data['timestamp'].isin(window_timestamps)]
            
            history_data = window_data[window_data['timestamp'].isin(window_timestamps[:self.history_steps])]
            future_data = window_data[window_data['timestamp'].isin(window_timestamps[-self.future_steps:])]
            
            history_traj, actions = self._organize_trajectory_and_actions(history_data, self.history_steps)
            future_traj, _ = self._organize_trajectory_and_actions(future_data, self.future_steps)
            
            samples.append({
                'history_trajectory': history_traj,  # 形状: (history_steps, max_vehicles, 7)
                'actions': actions,                  # 形状: (history_steps, 20)
                'future_trajectory': future_traj     # 形状: (future_steps, max_vehicles, 7)
            })
        return samples
    
    def _organize_trajectory_and_actions(self, data, T):
        """
        组织特定时间窗口的轨迹和动作
        
        参数:
            data: 特定时间窗口的DataFrame
            T: 时间步数
        
        返回:
            轨迹和动作数组
        """
        # 轨迹张量形状: (T, max_vehicles, 7)
        # 7个特征: x, y, vx, vy, psi_rad, is_lane_changing, is_main_lane
        trajectory = np.zeros((T, self.max_vehicles, 7))
        
        # 动作张量形状: (T, 20)
        # 20个动作维度: 辅助车辆的加速度和航向角变化
        actions = np.zeros((T, 20))
        
        timestamps = sorted(data['timestamp'].unique())
        for idx, t in enumerate(timestamps):
            if idx >= T:  # 防止数组越界
                break
                
            t_data = data[data['timestamp'] == t]
            
            if t_data.empty:
                continue
                
            vehicles = t_data['track_id'].values
            # 包含车道类型特征的完整特征集
            features = t_data[['x', 'y', 'vx', 'vy', 'psi_rad', 'is_lane_changing', 'is_main_lane']].values
            
            # 填充车辆特征
            if len(vehicles) > 0:
                num_vehicles = min(len(vehicles), self.max_vehicles)
                trajectory[idx, :num_vehicles, :] = features[:num_vehicles]
            
            # 填充辅助车辆动作
            aux_vehicles = t_data[t_data['lane_type'] == '变道车辆']
            if len(aux_vehicles) > 0:
                aux_actions = aux_vehicles[['ax', 'delta_psi']].values.flatten()
                num_actions = min(len(aux_actions), 20)
                actions[idx, :num_actions] = aux_actions[:num_actions]
        
        return trajectory, actions
    
    def _prepare_tensor_dataset(self):
        """
        准备TensorDataset，不进行归一化
        """
        # 准备数据的张量
        trajectories = []
        actions = []
        
        for sample in self.samples:
            # 组合历史和未来轨迹
            combined_traj = np.concatenate([sample['history_trajectory'], sample['future_trajectory']], axis=0)
            
            trajectories.append(combined_traj)
            actions.append(sample['actions'])
        
        # 转换为张量
        # 轨迹形状：(num_samples, H+F, max_vehicles, 7)
        # 动作形状：(num_samples, H, 20)
        self.tensor_dataset = TensorDataset(
            torch.tensor(np.array(trajectories), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32)
        )
    
    def __len__(self):
        """
        返回数据集中的样本数量
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        返回TensorDataset的项目
        """
        return self.tensor_dataset[idx]

