import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import typing
from collections import defaultdict
from typing import Callable, Optional

class RelativeTrafficDataset(Dataset):
    """
    相对位置交通轨迹数据集类，用于处理交通trajectory预测任务
    主要功能:
    - 支持可变长度的历史和未来时间步
    - 每辆车作为主车，计算其最近的10辆车的相对特征
    - 计算车辆动作（加速度、航向角变化）
    - 将车道类型作为车辆特征的一部分
    """
    def __init__(
        self, 
        raw_data: typing.Dict[int, typing.List[dict]], 
        history_steps: int = 20,  # 历史时间步数 
        future_steps: int = 1,    # 预测未来时间步数
        max_nearby_vehicles: int = 10,  # 每个主车最多考虑的附近车辆数
        max_aux_vehicles: int = 10,  # 最大辅道车辆数，用于计算actions
        progress_callback: Optional[Callable[[str, float], None]] = None  # 进度回调函数
    ):
        """
        初始化交通数据集

        参数:
            raw_data: 原始交通数据，按时间戳组织的字典
            history_steps: 历史时间步数 (默认: 20)
            future_steps: 预测未来时间步数 (默认: 1)
            max_nearby_vehicles: 每个主车考虑的最近车辆数 (默认: 10)
            max_aux_vehicles: 最大辅道车辆数，用于计算actions (默认: 10)
            progress_callback: 进度回调函数，接受阶段名称和进度百分比 (0-1)
        """
        # 配置参数
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.max_nearby_vehicles = max_nearby_vehicles
        self.max_aux_vehicles = max_aux_vehicles
        self.progress_callback = progress_callback
        
        # 验证输入数据
        if not raw_data:
            raise ValueError("输入数据不能为空")
        
        # 将原始数据转换为DataFrame
        self._update_progress("准备数据框架", 0.05)
        self.data = self._prepare_data_frame(raw_data)
        self._update_progress("准备数据框架", 0.1)

        
        # 计算车辆动作
        self._update_progress("计算车辆动作", 0.1)
        self._compute_actions()
        self.data.to_excel('process_data.xlsx', index=False) # 检查数据处理是否正确
        self._update_progress("计算车辆动作", 0.2)
        
        # 构建数据集样本
        self._update_progress("构建样本", 0.2)
        self.samples = self._build_samples()
        self._update_progress("构建样本", 0.7)
        
        # 准备TensorDataset
        self._update_progress("准备张量数据集", 0.7)
        self._prepare_tensor_dataset()
        self._update_progress("准备张量数据集", 1.0)
    
    def _update_progress(self, stage: str, progress: float):
        """更新进度回调"""
        if self.progress_callback:
            self.progress_callback(stage, progress)
    
    def _prepare_data_frame(self, raw_data):
        """
        将原始数据转换为DataFrame并预处理
        
        参数:
            raw_data: 原始交通数据字典
            
        返回:
            处理后的DataFrame
        """
        # 优化：预分配大小合适的列表，避免动态扩容
        total_frames = len(raw_data)
        avg_vehicles_per_frame = len(next(iter(raw_data.values()))) if raw_data else 0
        estimated_size = total_frames * avg_vehicles_per_frame
        raw_data_list = []
        
        # 预先定义所有字段，避免动态添加字段
        required_fields = ['timestamp', 'track_id', 'x', 'y', 'vx', 'vy', 'psi_rad', 
                           'lane_type', 'is_lane_changing', 'is_main_lane']
        
        # 使用排序后的时间戳迭代，避免重复排序
        sorted_timestamps = sorted(raw_data.keys())
        for i, t in enumerate(sorted_timestamps):
            if i % 50 == 0:  # 减少进度更新频率
                self._update_progress("准备数据框架", 0.05 + 0.025 * (i / total_frames))
                
            frame_data = raw_data[t]
            # 批量处理一帧中的所有车辆
            for item in frame_data:
                # 确保所有需要的字段都存在
                if 'lane_type' not in item:
                    item['lane_type'] = '主道车辆'  # 默认值
                # 直接设置派生字段
                item['is_lane_changing'] = 1.0 if item.get('lane_type') == '变道车辆' else 0.0
                item['is_main_lane'] = 0.0 if item.get('lane_type') == '变道车辆' else 1.0
                # 确保时间戳字段存在
                item['timestamp'] = t
            
            raw_data_list.extend(frame_data)
        
        # 优化：直接使用列表创建DataFrame，指定columns参数避免推断
        # 只选择需要的列，减少内存使用
        df = pd.DataFrame(raw_data_list, columns=required_fields)
        
        # 使用更高效的排序方法
        df = df.sort_values(by=['timestamp', 'track_id'])
        
        return df
    
    def _compute_actions(self):
        """
        计算车辆加速度和航向角变化
        使用向量化操作提高效率
        """
        # 预分配动作列
        self.data['ax'] = 0.0
        self.data['ay'] = 0.0
        self.data['delta_psi'] = 0.0
        
        # 优化：使用字典预先组织数据，避免多次查询
        track_groups = {}
        for track_id, group in self.data.groupby('track_id'):
            track_groups[track_id] = group.sort_values('timestamp')
        
        total_groups = len(track_groups)
        processed = 0
        
        # 批量处理车辆，减少DataFrame的索引操作
        batch_updates = {'index': [], 'ax': [], 'ay': [], 'delta_psi': []}
        
        for track_id, sorted_group in track_groups.items():
            processed += 1
            if processed % 50 == 0:  # 减少进度更新频率
                self._update_progress("计算车辆动作", 0.1 + 0.08 * (processed / total_groups))
            
            if len(sorted_group) >= 2:
                # 使用numpy操作计算差分，比pandas的diff()更快
                timestamps = sorted_group['timestamp'].values
                vx_values = sorted_group['vx'].values
                vy_values = sorted_group['vy'].values
                psi_values = sorted_group['psi_rad'].values
                
                # 计算相邻时间步的差值
                vx_diff = np.zeros_like(vx_values)
                vy_diff = np.zeros_like(vy_values)
                psi_diff = np.zeros_like(psi_values)
                
                vx_diff[1:] = vx_values[1:] - vx_values[:-1]
                vy_diff[1:] = vy_values[1:] - vy_values[:-1]
                psi_diff[1:] = psi_values[1:] - psi_values[:-1]
                
                # 批量收集更新
                batch_updates['index'].extend(sorted_group.index)
                batch_updates['ax'].extend(vx_diff)
                batch_updates['ay'].extend(vy_diff)
                batch_updates['delta_psi'].extend(psi_diff)
        
        # 一次性更新DataFrame
        update_df = pd.DataFrame({
            'ax': batch_updates['ax'],
            'ay': batch_updates['ay'],
            'delta_psi': batch_updates['delta_psi']
        }, index=batch_updates['index'])
        
        self.data.update(update_df)
    
    def _transform_to_ego_frame(self, ego_vehicle, other_vehicles):
        """
        将其他车辆的特征转换到以主车为参考的局部坐标系
        使用向量化操作提高效率
        
        参数:
            ego_vehicle: 主车特征
            other_vehicles: 其他车辆特征DataFrame
            
        返回:
            转换后的相对特征列表
        """
        if other_vehicles.empty:
            return []
            
        # 提取主车位置和朝向
        ego_x, ego_y = ego_vehicle['x'], ego_vehicle['y']
        ego_psi = ego_vehicle['psi_rad']
        ego_vx, ego_vy = ego_vehicle['vx'], ego_vehicle['vy']
        
        # 向量化计算相对位置
        dx = other_vehicles['x'].values - ego_x
        dy = other_vehicles['y'].values - ego_y
        
        # 向量化计算旋转 - 使用预计算的三角函数值
        cos_psi, sin_psi = np.cos(-ego_psi), np.sin(-ego_psi)
        rel_x = dx * cos_psi - dy * sin_psi
        rel_y = dx * sin_psi + dy * cos_psi
        
        # 向量化计算相对速度
        rel_vx = other_vehicles['vx'].values - ego_vx
        rel_vy = other_vehicles['vy'].values - ego_vy
        
        # 向量化计算相对朝向
        rel_psi = other_vehicles['psi_rad'].values - ego_psi
        # 标准化角度 - 使用向量化操作
        rel_psi = np.arctan2(np.sin(rel_psi), np.cos(rel_psi))
        
        # 优化：直接创建结果数组
        n_vehicles = len(other_vehicles)
        track_ids = other_vehicles['track_id'].values
        is_lane_changing = other_vehicles['is_lane_changing'].values
        is_main_lane = other_vehicles['is_main_lane'].values
        
        # 使用字典推导式一次性创建所有结果
        relative_features = [
            {
                'rel_x': rel_x[i], 
                'rel_y': rel_y[i],
                'rel_vx': rel_vx[i], 
                'rel_vy': rel_vy[i],
                'rel_psi': rel_psi[i],
                'is_lane_changing': is_lane_changing[i],
                'is_main_lane': is_main_lane[i],
                'track_id': track_ids[i]
            }
            for i in range(n_vehicles)
        ]
            
        return relative_features
    
    def _get_nearest_vehicles(self, ego_vehicle, frame_vehicles, max_vehicles=10):
        """
        获取距离主车最近的N辆车
        使用更高效的距离计算
        
        参数:
            ego_vehicle: 主车数据
            frame_vehicles: 当前帧所有车辆数据
            max_vehicles: 最大考虑的车辆数
            
        返回:
            最近N辆车的相对特征
        """
        # 排除主车自身 - 使用更高效的布尔索引
        ego_id = ego_vehicle['track_id']
        mask = frame_vehicles['track_id'] != ego_id
        other_vehicles = frame_vehicles[mask]
        
        if other_vehicles.empty:
            return []
        
        # 高效计算欧几里得距离 - 使用向量化操作
        ego_x, ego_y = ego_vehicle['x'], ego_vehicle['y']
        dx = other_vehicles['x'].values - ego_x
        dy = other_vehicles['y'].values - ego_y
        
        # 优化：直接计算平方距离，避免开方操作
        sq_distances = dx * dx + dy * dy
        
        # 只选择最近的max_vehicles辆车
        if len(sq_distances) > max_vehicles:
            # 获取最小的k个元素的索引，比完全排序更快
            nearest_indices = np.argpartition(sq_distances, max_vehicles)[:max_vehicles]
            nearest_vehicles = other_vehicles.iloc[nearest_indices]
        else:
            # 如果车辆数小于max_vehicles，直接使用所有车辆
            nearest_vehicles = other_vehicles
        
        # 转换到主车坐标系
        relative_features = self._transform_to_ego_frame(ego_vehicle, nearest_vehicles)
        
        return relative_features
    
    def _get_aux_vehicle_actions(self, frame_data):
        """
        获取辅道车辆的动作
        
        参数:
            frame_data: 当前帧的车辆数据
            
        返回:
            辅道车辆的动作数组
        """
        # 优化：直接使用布尔索引，避免创建中间DataFrame
        mask = frame_data['is_lane_changing'] > 0.5
        aux_vehicles = frame_data[mask]
        
        # 预分配数组
        actions = np.zeros(self.max_aux_vehicles * 2)  # 每辆辅道车辆有2个动作: ax和delta_psi
        
        if not aux_vehicles.empty:
            # 取最多max_aux_vehicles辆辅道车辆
            n_vehicles = min(len(aux_vehicles), self.max_aux_vehicles)
            aux_vehicles = aux_vehicles.iloc[:n_vehicles]
            
            # 直接提取动作，减少内存使用
            ax_values = aux_vehicles['ax'].values
            delta_psi_values = aux_vehicles['delta_psi'].values
            
            # 交错填充ax和delta_psi值
            actions[:n_vehicles*2:2] = ax_values
            actions[1:n_vehicles*2:2] = delta_psi_values
        
        return actions

    def _build_samples(self):
        """
        构建训练样本，每辆车都作为主车
        使用更高效的数据结构和处理方法
        
        返回:
            包含历史轨迹、动作和未来轨迹的样本列表
        """
        samples = []
        timestamps = sorted(self.data['timestamp'].unique())
        
        # 检查是否有足够的时间戳
        if len(timestamps) < self.history_steps + self.future_steps:
            raise ValueError(f"数据中的时间戳数量({len(timestamps)})少于所需的时间步数({self.history_steps + self.future_steps})")
        
        # 优化：使用字典而不是DataFrame进行时间戳索引
        self._update_progress("构建样本-准备时间戳数据", 0.25)
        
        # 优化：使用defaultdict和预处理数据，加快查询效率
        # 创建按时间戳分组的车辆字典，每个时间戳包含所有车辆的DataFrame
        data_by_timestamp = {}
        # 创建按时间戳和车辆ID索引的数据，加速单车辆查询
        vehicle_data_by_timestamp_id = defaultdict(dict)
        
        for t in timestamps:
            frame_data = self.data[self.data['timestamp'] == t]
            data_by_timestamp[t] = frame_data
            
            # 将每辆车的数据组织为字典，加速查询
            for _, row in frame_data.iterrows():
                vehicle_id = row['track_id']
                vehicle_data_by_timestamp_id[t][vehicle_id] = row.to_dict()
        
        self._update_progress("构建样本-准备时间戳数据", 0.3)
        
        # 计算可用的窗口数量
        total_windows = len(timestamps) - self.history_steps - self.future_steps + 1
        
        # 优化：预计算每个窗口中的车辆出现次数
        vehicle_presence = defaultdict(lambda: defaultdict(int))
        
        for i in range(total_windows):
            window_timestamps = timestamps[i:i + self.history_steps + self.future_steps]
            history_timestamps = window_timestamps[:self.history_steps]
            
            # 统计每辆车在历史窗口中出现的次数
            for t in history_timestamps:
                frame_data = data_by_timestamp[t]
                for vehicle_id in frame_data['track_id'].unique():
                    vehicle_presence[i][vehicle_id] += 1
        
        # 滑动窗口生成样本
        for i in range(total_windows):
            if i % 20 == 0 or i == total_windows - 1:  # 减少进度更新频率
                progress = 0.3 + 0.35 * (i / total_windows)
                self._update_progress(f"构建样本-处理窗口 {i+1}/{total_windows}", progress)
                
            window_timestamps = timestamps[i:i + self.history_steps + self.future_steps]
            history_timestamps = window_timestamps[:self.history_steps]
            future_timestamps = window_timestamps[-self.future_steps:]
            
            # 快速查找在所有历史时间步中都存在的车辆
            valid_vehicles = [
                vehicle_id for vehicle_id, count in vehicle_presence[i].items()
                if count == self.history_steps
            ]
            
            # 优化：批量处理同一窗口内的样本
            batch_samples = []
            
            # 为每辆有效车辆并行创建样本
            for vehicle_id in valid_vehicles:
                # 创建基于该车辆作为主车的样本
                sample = self._create_vehicle_sample(
                    vehicle_id, 
                    data_by_timestamp, 
                    vehicle_data_by_timestamp_id,
                    history_timestamps, 
                    future_timestamps
                )
                
                if sample:
                    batch_samples.append(sample)
            
            # 批量添加样本
            samples.extend(batch_samples)
        
        self._update_progress("构建样本-完成", 0.95)
        return samples
    
    def _create_vehicle_sample(self, ego_id, data_by_timestamp, vehicle_data_by_ts_id, 
                               history_timestamps, future_timestamps):
        """
        为特定车辆创建样本，使用预计算的数据加速处理
        
        参数:
            ego_id: 主车ID
            data_by_timestamp: 按时间戳分组的数据字典
            vehicle_data_by_ts_id: 按时间戳和车辆ID索引的数据
            history_timestamps: 历史时间戳
            future_timestamps: 未来时间戳
            
        返回:
            包含主车和最近车辆相对特征的样本
        """
        # 初始化轨迹张量和动作张量 - 使用zeros提前分配内存
        total_steps = self.history_steps + self.future_steps
        trajectory = np.zeros((total_steps, self.max_nearby_vehicles, 7), dtype=np.float32)
        actions = np.zeros((self.history_steps, 2 * self.max_aux_vehicles), dtype=np.float32)
        
        # 处理历史和未来的每个时间步
        all_timestamps = history_timestamps + future_timestamps
        for t_idx, timestamp in enumerate(all_timestamps):
            frame_data = data_by_timestamp[timestamp]
            
            # 快速查找主车数据 - 使用预计算的索引
            if ego_id not in vehicle_data_by_ts_id[timestamp]:
                continue
            
            ego_vehicle = vehicle_data_by_ts_id[timestamp][ego_id]
            
            # 获取最近的车辆及其相对特征
            nearest_vehicles = self._get_nearest_vehicles(
                ego_vehicle, 
                frame_data, 
                self.max_nearby_vehicles
            )
            
            # 如果是历史时间步，处理辅道车辆动作
            if timestamp in history_timestamps:
                h_idx = history_timestamps.index(timestamp)
                aux_actions = self._get_aux_vehicle_actions(frame_data)
                actions[h_idx] = aux_actions
            
            # 填充轨迹张量 - 优化内存访问模式
            for v_idx, rel_vehicle in enumerate(nearest_vehicles):
                if v_idx >= self.max_nearby_vehicles:
                    break
                
                # 批量填充特征 - 减少循环内部的索引操作
                features = np.array([
                    rel_vehicle['rel_x'],
                    rel_vehicle['rel_y'],
                    rel_vehicle['rel_vx'],
                    rel_vehicle['rel_vy'],
                    rel_vehicle['rel_psi'],
                    rel_vehicle['is_lane_changing'],
                    rel_vehicle['is_main_lane']
                ], dtype=np.float32)
                
                trajectory[t_idx, v_idx] = features
        
        # 分离历史和未来轨迹
        history_trajectory = trajectory[:self.history_steps]
        future_trajectory = trajectory[self.history_steps:]
        
        return {
            'ego_id': ego_id,
            'history_trajectory': history_trajectory,  # 形状: (history_steps, max_nearby_vehicles, 7)
            'future_trajectory': future_trajectory,    # 形状: (future_steps, max_nearby_vehicles, 7)
            'actions': actions                         # 形状: (history_steps, 2*max_aux_vehicles)
        }
    
    def _prepare_tensor_dataset(self):
        """
        准备TensorDataset，使用批处理减少内存使用
        """
        if not self.samples:
            raise ValueError("没有可用的样本来准备TensorDataset")
            
        # 优化：直接创建numpy数组，避免列表操作
        total_samples = len(self.samples)
        
        # 预分配numpy数组 - 减少内存重分配和复制
        combined_shape = (total_samples, self.history_steps + self.future_steps, 
                           self.max_nearby_vehicles, 7)
        actions_shape = (total_samples, self.history_steps, 2 * self.max_aux_vehicles)
        
        all_trajectories = np.zeros(combined_shape, dtype=np.float32)
        all_actions = np.zeros(actions_shape, dtype=np.float32)
        
        # 批量填充数组
        for i, sample in enumerate(self.samples):
            if i % 200 == 0 or i == total_samples - 1:  # 减少进度更新频率
                progress = 0.7 + 0.28 * (i / total_samples)
                self._update_progress(f"准备张量数据集 {i+1}/{total_samples}", progress)
                
            # 组合历史和未来轨迹
            combined_traj = np.concatenate([sample['history_trajectory'], sample['future_trajectory']], axis=0)
            all_trajectories[i] = combined_traj
            all_actions[i] = sample['actions']
        
        # 转换为张量 - 使用torch.from_numpy避免额外复制
        self._update_progress("转换为张量", 0.98)
        self.tensor_dataset = TensorDataset(
            torch.from_numpy(all_trajectories),
            torch.from_numpy(all_actions)
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
    
     # 添加自定义序列化方法
    def __getstate__(self):
        """自定义序列化行为，排除progress_callback属性"""
        state = self.__dict__.copy()
        # 排除progress_callback，因为它不能被序列化
        if 'progress_callback' in state:
            del state['progress_callback']
        return state

    def __setstate__(self, state):
        """自定义反序列化行为，添加默认progress_callback"""
        # 恢复状态
        self.__dict__.update(state)
        # 添加空的progress_callback
        self.progress_callback = None
