import pickle
import os
import numpy as np
import math
import heapq
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def load_frame_data(file_path: str):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"Data successfully loaded from {file_path}")
    return data

def cal_relative_data_vehicles(frame_data):
    """计算相对数据，但保留绝对位置信息用于后续统一变换"""
    all_relative_data = {}
    
    for ego_vehicle in frame_data:
        ego_id = ego_vehicle['track_id']
        ego_x = ego_vehicle['x']
        ego_y = ego_vehicle['y']
        ego_vx = ego_vehicle['vx']
        ego_vy = ego_vehicle['vy']
        ego_psi = ego_vehicle['psi_rad']
        
        relative_data = {}
        
        for vehicle in frame_data:
            vehicle_id = vehicle['track_id']
            
            # 计算绝对位置差值（用于距离计算）
            dx = vehicle['x'] - ego_x
            dy = vehicle['y'] - ego_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            relative_data[vehicle_id] = {
                'track_id': vehicle_id,
                'timestamp': vehicle['timestamp'],
                'length': vehicle['length'],
                'width': vehicle['width'],
                'lane_type': vehicle['lane_type'],
                'is_ego': vehicle_id == ego_id,
                'distance': distance,
                # 保留绝对位置和速度信息
                'abs_x': vehicle['x'],
                'abs_y': vehicle['y'],
                'abs_vx': vehicle['vx'],
                'abs_vy': vehicle['vy'],
                'abs_psi': vehicle['psi_rad']
            }
        
        all_relative_data[ego_id] = relative_data
    
    return all_relative_data

class TrajectoryDataGenerator:
    def __init__(self, config=None):
        default_config = {
            'history_sec': 1,
            'future_sec': 3,
            'fps': 10,
            'num_agents': 6,
            'input_dim': 7,
            'output_dim': 3,
            'sliding_step': 1,
            'use_multiprocessing': True,
            'num_workers': None
        }
        self.config = default_config if config is None else {**default_config, **config}
        self.history_frames = int(self.config['history_sec'] * self.config['fps'])
        self.future_frames = int(self.config['future_sec'] * self.config['fps'])
        self.total_frames = self.history_frames + self.future_frames
        
        if self.config['num_workers'] is None:
            self.config['num_workers'] = max(1, mp.cpu_count() - 1)

    def transform_to_ego_frame(self, target_x, target_y, target_psi, ego_x, ego_y, ego_psi):
        """将目标位置和朝向变换到ego车辆坐标系"""
        # 位置变换
        dx = target_x - ego_x
        dy = target_y - ego_y
        cos_psi = math.cos(ego_psi)
        sin_psi = math.sin(ego_psi)
        rel_x = cos_psi * dx + sin_psi * dy
        rel_y = -sin_psi * dx + cos_psi * dy
        
        # 朝向变换
        rel_psi = (target_psi - ego_psi + math.pi) % (2 * math.pi) - math.pi
        
        return rel_x, rel_y, rel_psi
    
    def get_vehicle_features(self, abs_x, abs_y, abs_vx, abs_vy, abs_psi, length, width, ego_x, ego_y, ego_psi):
        """基于绝对位置计算车辆特征，统一变换到ego坐标系"""
        rel_x, rel_y, rel_psi = self.transform_to_ego_frame(abs_x, abs_y, abs_psi, ego_x, ego_y, ego_psi)
        
        return np.array([
            rel_x,
            rel_y,
            abs_vx,  # 速度保持绝对值
            abs_vy,
            rel_psi,  # 相对朝向
            length,
            width,
        ])
    
    def get_vehicle_future_features(self, abs_x, abs_y, abs_psi, ego_x, ego_y, ego_psi):
        """获取未来预测需要的特征"""
        rel_x, rel_y, rel_psi = self.transform_to_ego_frame(abs_x, abs_y, abs_psi, ego_x, ego_y, ego_psi)
        
        return np.array([
            rel_x,
            rel_y,
            rel_psi,
        ])
    
    @staticmethod
    def find_closest_vehicles(relative_data, ego_id, n=6):
        distances = [(v_data['distance'], v_id) for v_id, v_data in relative_data.items() if v_id != ego_id]
        return [v_id for _, v_id in heapq.nsmallest(n, distances)]
    
    def process_single_file(self, file_item):
        file_id, file_data = file_item
        frame_ids = sorted(file_data.keys())
        vehicle_trajectories = defaultdict(lambda: defaultdict(dict))
        
        for frame_id in tqdm(frame_ids, desc=f"Processing file {file_id}", leave=False):
            frames = file_data[frame_id]
            relative_data = cal_relative_data_vehicles(frames)
            
            for ego_id, ego_relative_data in relative_data.items():
                closest_vehicles = self.find_closest_vehicles(ego_relative_data, ego_id, self.config['num_agents'])
                
                # 存储原始绝对位置信息
                vehicle_trajectories[ego_id][frame_id] = {
                    'ego_data': ego_relative_data[ego_id],
                    'agents_data': [ego_relative_data[v_id] for v_id in closest_vehicles if v_id in ego_relative_data]
                }
        
        return self.process_vehicle_trajectories(vehicle_trajectories, file_id)
    
    def process_vehicle_trajectories(self, vehicle_trajectories, file_id):
        trajectories = []
        
        for ego_id, trajectory_data in vehicle_trajectories.items():
            frame_ids = sorted(trajectory_data.keys())
            
            if len(frame_ids) < self.total_frames:
                continue
            
            for i in range(0, len(frame_ids) - self.total_frames + 1, self.config['sliding_step']):
                history_frame_ids = frame_ids[i:i+self.history_frames]
                future_frame_ids = frame_ids[i+self.history_frames:i+self.total_frames]
                
                # 使用历史序列的最后一帧作为参考帧
                current_frame_id = history_frame_ids[-1]
                ego_reference = trajectory_data[current_frame_id]['ego_data']
                ego_ref_x = ego_reference['abs_x']
                ego_ref_y = ego_reference['abs_y']
                ego_ref_psi = ego_reference['abs_psi']
                
                ego_history = []
                agent_history = []
                ego_future = []
                agent_future = []
                ego_history_absolute_positions = []
                ego_future_absolute_positions = []
                
                all_frame_ids = history_frame_ids + future_frame_ids
                
                for j, frame_id in enumerate(all_frame_ids):
                    frame_data = trajectory_data[frame_id]
                    ego_data = frame_data['ego_data']
                    
                    # 基于参考帧统一变换ego车辆特征
                    ego_features = self.get_vehicle_features(
                        ego_data['abs_x'], ego_data['abs_y'], 
                        ego_data['abs_vx'], ego_data['abs_vy'], ego_data['abs_psi'],
                        ego_data['length'], ego_data['width'],
                        ego_ref_x, ego_ref_y, ego_ref_psi
                    )
                    
                    # 处理agent车辆
                    agents_features = []
                    agents_data = frame_data['agents_data']
                    
                    for agent_data in agents_data:
                        agent_features = self.get_vehicle_features(
                            agent_data['abs_x'], agent_data['abs_y'],
                            agent_data['abs_vx'], agent_data['abs_vy'], agent_data['abs_psi'],
                            agent_data['length'], agent_data['width'],
                            ego_ref_x, ego_ref_y, ego_ref_psi
                        )
                        agents_features.append(agent_features)
                    
                    # 填充不足的agent数量
                    while len(agents_features) < self.config['num_agents']:
                        agents_features.append(np.zeros(self.config['input_dim']))
                    
                    # 确保agent数量不超过配置
                    agents_features = agents_features[:self.config['num_agents']]
                    
                    ego_abs_pos = np.array([ego_data['abs_x'], ego_data['abs_y'], ego_data['abs_psi']])
                    
                    if j < self.history_frames:
                        ego_history.append(ego_features)
                        agent_history.append(agents_features)
                        ego_history_absolute_positions.append(ego_abs_pos)
                    else:
                        # 未来帧的特征（用于预测目标）
                        ego_future_feat = self.get_vehicle_future_features(
                            ego_data['abs_x'], ego_data['abs_y'], ego_data['abs_psi'],
                            ego_ref_x, ego_ref_y, ego_ref_psi
                        )
                        ego_future.append(ego_future_feat)
                        
                        agents_future_feat = []
                        for agent_data in agents_data:
                            agent_future_feat = self.get_vehicle_future_features(
                                agent_data['abs_x'], agent_data['abs_y'], agent_data['abs_psi'],
                                ego_ref_x, ego_ref_y, ego_ref_psi
                            )
                            agents_future_feat.append(agent_future_feat)
                        
                        while len(agents_future_feat) < self.config['num_agents']:
                            agents_future_feat.append(np.zeros(self.config['output_dim']))
                        
                        agents_future_feat = agents_future_feat[:self.config['num_agents']]
                        agent_future.append(agents_future_feat)
                        ego_future_absolute_positions.append(ego_abs_pos)
                
                # 将当前帧的ego位置设为原点
                ego_history[-1][0] = 0.0
                ego_history[-1][1] = 0.0
                
                trajectories.append({
                    'file_id': file_id,
                    'ego_id': ego_id,
                    'start_frame': history_frame_ids[0],
                    'middle_frame': current_frame_id,
                    'end_frame': future_frame_ids[-1],
                    'ego_history': np.array(ego_history),
                    'agent_history': np.array(agent_history),
                    'ego_future': np.array(ego_future),
                    'agent_future': np.array(agent_future),
                    'ego_absolute_positions': ego_history_absolute_positions,
                    'ego_future_absolute_positions': ego_future_absolute_positions
                })
        
        return trajectories
    
    def generate_trajectories(self, loaded_data):
        all_trajectories = []
        
        if self.config['use_multiprocessing'] and len(loaded_data) > 1:
            print(f"Using multiprocessing with {self.config['num_workers']} workers")
            
            with mp.Pool(processes=self.config['num_workers']) as pool:
                results = []
                for file_item in loaded_data.items():
                    result = pool.apply_async(self.process_single_file, (file_item,))
                    results.append(result)
                
                for i, result in enumerate(tqdm(results, desc="Processing files")):
                    trajectories = result.get()
                    all_trajectories.extend(trajectories)
                    print(f"File {i+1}/{len(results)} completed, generated {len(trajectories)} trajectories")
        else:
            print("Using single process")
            for i, file_item in enumerate(tqdm(loaded_data.items(), desc="Processing files")):
                trajectories = self.process_single_file(file_item)
                all_trajectories.extend(trajectories)
                print(f"File {i+1}/{len(loaded_data)} completed, generated {len(trajectories)} trajectories")
        
        return all_trajectories
    
    def process_data(self, loaded_data, save_path=None):
        print(f"Processing {len(loaded_data)} files...")
        trajectories = self.generate_trajectories(loaded_data)
        
        print(f"Generated {len(trajectories)} trajectory samples")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(trajectories, f)
            print(f"Trajectories saved to {save_path}")
        
        return trajectories


if __name__ == "__main__":
    input_dir = r"data_process\behaviour\data"
    output_dir = r"Mamba_attn_agent_pred\src\datasets\data"
    
    config = {
        'history_sec': 1.1,
        'future_sec': 3,
        'fps': 10,
        'num_agents': 6,
        'input_dim': 7,
        'output_dim': 3,
        'sliding_step': 2,
        'use_multiprocessing': True,
        'num_workers': 4
    }
    
    loaded_data = load_frame_data(os.path.join(input_dir, "all_frame_data.pkl"))
    
    data_generator = TrajectoryDataGenerator(config)
    trajectories = data_generator.process_data(loaded_data, os.path.join(output_dir, "train_trajectories_ego.pkl"))