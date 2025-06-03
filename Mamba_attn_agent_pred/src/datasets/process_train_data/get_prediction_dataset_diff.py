import pickle
import os
import numpy as np
import math
import heapq
from collections import defaultdict

def load_frame_data(file_path: str):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"Data successfully loaded from {file_path}")
    return data

def cal_relative_data_vehicles(frame_data):
    all_relative_data = {}
    
    for ego_vehicle in frame_data:
        ego_id = ego_vehicle['track_id']
        
        ego_x = ego_vehicle['x']
        ego_y = ego_vehicle['y']
        ego_vx = ego_vehicle['vx']
        ego_vy = ego_vehicle['vy']
        ego_psi = ego_vehicle['psi_rad']
        
        rotation_matrix = np.array([
            [math.cos(ego_psi), math.sin(ego_psi)],
            [-math.sin(ego_psi), math.cos(ego_psi)]
        ])
        
        relative_data = {}
        
        for vehicle in frame_data:
            vehicle_id = vehicle['track_id']
            
            dx = vehicle['x'] - ego_x
            dy = vehicle['y'] - ego_y
            
            rel_pos_global = np.array([dx, dy])
            rel_pos_local = rotation_matrix.dot(rel_pos_global)
            
            dvx = vehicle['vx'] - ego_vx
            dvy = vehicle['vy'] - ego_vy
            
            rel_vel_global = np.array([dvx, dvy])
            rel_vel_local = rotation_matrix.dot(rel_vel_global)
            
            rel_psi = vehicle['psi_rad'] - ego_psi
            rel_psi = (rel_psi + math.pi) % (2 * math.pi) - math.pi
            
            relative_data[vehicle_id] = {
                'track_id': vehicle_id,
                'timestamp': vehicle['timestamp'],
                'rel_x': rel_pos_local[0],
                'rel_y': rel_pos_local[1],
                'rel_vx': rel_vel_local[0],
                'rel_vy': rel_vel_local[1],
                'rel_psi': rel_psi,
                'length': vehicle['length'],
                'width': vehicle['width'],
                'lane_type': vehicle['lane_type'],
                'is_ego': vehicle_id == ego_id,
                'distance': np.sqrt(rel_pos_local[0]**2 + rel_pos_local[1]**2),
                'abs_x': vehicle['x'],
                'abs_y': vehicle['y'],
                'abs_vx': vehicle['vx'],
                'abs_vy': vehicle['vy'],
                'abs_psi': vehicle['psi_rad']
            }
        
        all_relative_data[ego_id] = relative_data
    
    return all_relative_data

def differential_encoding(features):
    diff_features = np.zeros_like(features)
    diff_features[0] = features[0]
    diff_features[1:] = features[1:] - features[:-1]
    return diff_features

def differential_decoding(diff_features):
    features = np.zeros_like(diff_features)
    features[0] = diff_features[0]
    for i in range(1, len(diff_features)):
        features[i] = features[i-1] + diff_features[i]
    return features

def convert_features_to_absolute(features, initial_ego_state):
    abs_features = np.zeros_like(features)
    
    ego_x = initial_ego_state['abs_x']
    ego_y = initial_ego_state['abs_y']
    ego_psi = initial_ego_state['abs_psi']
    
    rotation_matrix = np.array([
        [math.cos(ego_psi), -math.sin(ego_psi)],
        [math.sin(ego_psi), math.cos(ego_psi)]
    ])
    
    for i in range(len(features)):
        rel_x, rel_y, psi = features[i, 0], features[i, 1], features[i, 2]
        
        rel_pos_local = np.array([rel_x, rel_y])
        rel_pos_global = rotation_matrix.dot(rel_pos_local)
        
        abs_x = ego_x + rel_pos_global[0]
        abs_y = ego_y + rel_pos_global[1]
        abs_psi = ego_psi + psi
        abs_psi = (abs_psi + math.pi) % (2 * math.pi) - math.pi
        
        abs_features[i] = [abs_x, abs_y, abs_psi]
    
    return abs_features

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
            'min_seq_len': 40
        }
        self.config = default_config if config is None else {**default_config, **config}
        self.history_frames = int(self.config['history_sec'] * self.config['fps'])
        self.future_frames = int(self.config['future_sec'] * self.config['fps'])
        self.total_frames = self.history_frames + self.future_frames
    
    def get_vehicle_features(self, vehicle_data):
        return np.array([
            vehicle_data['rel_x'],
            vehicle_data['rel_y'],
            vehicle_data['rel_vx'],
            vehicle_data['rel_vy'],
            vehicle_data['rel_psi'],
            vehicle_data['length'],
            vehicle_data['width'],
        ])
    
    def get_vehicle_future_features(self, vehicle_data):
        return np.array([
            vehicle_data['rel_x'],
            vehicle_data['rel_y'],
            vehicle_data['rel_psi'],
        ])
    
    def get_original_data(self, vehicle_data):
        return {
            'abs_x': vehicle_data['abs_x'],
            'abs_y': vehicle_data['abs_y'],
            'abs_vx': vehicle_data['abs_vx'],
            'abs_vy': vehicle_data['abs_vy'],
            'abs_psi': vehicle_data['abs_psi'],
            'length': vehicle_data['length'],
            'width': vehicle_data['width'],
        }
    
    def get_ego_features(self, ego_data):
        return np.array([
            ego_data['abs_x'],
            ego_data['abs_y'],
            ego_data['abs_vx'],
            ego_data['abs_vy'],
            ego_data['abs_psi'],
            ego_data['length'],
            ego_data['width'],
        ])
    
    def get_ego_future_features(self, ego_data):
        return np.array([
            ego_data['abs_x'],
            ego_data['abs_y'],
            ego_data['abs_psi'],
        ])
    
    def find_closest_vehicles(self, relative_data, ego_id, n=6):
        vehicles = []
        for v_id, v_data in relative_data.items():
            if v_id != ego_id:
                vehicles.append((v_data['distance'], v_id))
        
        closest = heapq.nsmallest(n, vehicles)
        return [v_id for _, v_id in closest]
    
    def generate_trajectories(self, loaded_data):
        all_trajectories = []
        
        for file_id, file_data in loaded_data.items():
            frame_ids = sorted(file_data.keys())
            
            vehicle_trajectories = defaultdict(lambda: defaultdict(dict))
            
            for frame_id in frame_ids:
                frames = file_data[frame_id]
                relative_data = cal_relative_data_vehicles(frames)
                
                for ego_id, ego_relative_data in relative_data.items():
                    closest_vehicles = self.find_closest_vehicles(ego_relative_data, ego_id, self.config['num_agents'])
                    
                    ego_features = self.get_ego_features(ego_relative_data[ego_id])
                    ego_future_features = self.get_ego_future_features(ego_relative_data[ego_id])
                    ego_original_data = self.get_original_data(ego_relative_data[ego_id])
                    
                    vehicle_trajectories[ego_id][frame_id]['ego'] = ego_features
                    vehicle_trajectories[ego_id][frame_id]['ego_future'] = ego_future_features
                    vehicle_trajectories[ego_id][frame_id]['ego_original'] = ego_original_data
                    
                    agents_features = []
                    agents_future_features = []
                    agents_original_data = []
                    
                    for v_id in closest_vehicles:
                        if v_id in ego_relative_data:
                            agents_features.append(self.get_vehicle_features(ego_relative_data[v_id]))
                            agents_future_features.append(self.get_vehicle_future_features(ego_relative_data[v_id]))
                            agents_original_data.append(self.get_original_data(ego_relative_data[v_id]))
                    
                    while len(agents_features) < self.config['num_agents']:
                        agents_features.append(np.zeros(self.config['input_dim']))
                        agents_future_features.append(np.zeros(self.config['output_dim']))
                        agents_original_data.append({})
                    
                    vehicle_trajectories[ego_id][frame_id]['agents'] = np.array(agents_features)
                    vehicle_trajectories[ego_id][frame_id]['agents_future'] = np.array(agents_future_features)
                    vehicle_trajectories[ego_id][frame_id]['agents_original'] = agents_original_data
            
            for ego_id, trajectory_data in vehicle_trajectories.items():
                frame_ids = sorted(trajectory_data.keys())
                
                if len(frame_ids) < self.total_frames:
                    continue
                
                for i in range(0, len(frame_ids) - self.total_frames + 1, self.config['sliding_step']):
                    history_frame_ids = frame_ids[i:i+self.history_frames]
                    future_frame_ids = frame_ids[i+self.history_frames:i+self.total_frames]
                    
                    ego_history = np.array([trajectory_data[f_id]['ego'] for f_id in history_frame_ids])
                    agent_history = np.array([trajectory_data[f_id]['agents'] for f_id in history_frame_ids])
                    
                    ego_future = np.array([trajectory_data[f_id]['ego_future'] for f_id in future_frame_ids])
                    agent_future = np.array([trajectory_data[f_id]['agents_future'] for f_id in future_frame_ids])
                    
                    ego_original = [trajectory_data[f_id]['ego_original'] for f_id in history_frame_ids + future_frame_ids]
                    agent_original = [trajectory_data[f_id]['agents_original'] for f_id in history_frame_ids + future_frame_ids]
                    
                    diff_ego_history = differential_encoding(ego_history)
                    diff_agent_history = np.zeros_like(agent_history)
                    for agent_idx in range(agent_history.shape[1]):
                        diff_agent_history[:, agent_idx] = differential_encoding(agent_history[:, agent_idx])
                    
                    diff_ego_future = differential_encoding(ego_future)
                    diff_agent_future = np.zeros_like(agent_future)
                    for agent_idx in range(agent_future.shape[1]):
                        diff_agent_future[:, agent_idx] = differential_encoding(agent_future[:, agent_idx])
                    
                    all_trajectories.append({
                        # 'file_id': file_id,
                        # 'ego_id': ego_id,
                        # 'start_frame': history_frame_ids[0],
                        # 'middle_frame': history_frame_ids[-1],
                        # 'end_frame': future_frame_ids[-1],
                        # 'ego_history': ego_history,
                        # 'agent_history': agent_history,
                        # 'ego_future': ego_future,
                        # 'agent_future': agent_future,
                        'diff_ego_history': diff_ego_history,
                        'diff_agent_history': diff_agent_history,
                        'diff_ego_future': diff_ego_future,
                        'diff_agent_future': diff_agent_future,
                        # 'ego_original': ego_original,
                        # 'agent_original': agent_original
                    })
        
        return all_trajectories
    
    def process_data(self, loaded_data, save_path=None):
        trajectories = self.generate_trajectories(loaded_data)
        
        print(f"Generated {len(trajectories)} trajectory samples")
        
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(trajectories, f)
            print(f"Trajectories saved to {save_path}")
        
        return trajectories
    
    def format_batch_data(self, sample_batch):
        ego_history = sample_batch['ego_history']
        agent_history = sample_batch['agent_history']
        ego_future = sample_batch['ego_future']
        agent_future = sample_batch['agent_future']
        diff_ego_history = sample_batch['diff_ego_history']
        diff_agent_history = sample_batch['diff_agent_history']
        diff_ego_future = sample_batch['diff_ego_future']
        diff_agent_future = sample_batch['diff_agent_future']
        meta = sample_batch['meta']
        
        formatted_batch = {
            'ego_history': ego_history,
            'agent_history': agent_history,
            'ego_future': ego_future,
            'agent_future': agent_future,
            'diff_ego_history': diff_ego_history,
            'diff_agent_history': diff_agent_history,
            'diff_ego_future': diff_ego_future,
            'diff_agent_future': diff_agent_future,
            'meta': meta
        }
        
        return formatted_batch
    
    def process_batch_data(self, sample_batch):
        processed_batch = self.format_batch_data(sample_batch)
        return processed_batch
    
    def reconstruct_trajectory(self, predicted_future, initial_ego_state):
        return convert_features_to_absolute(predicted_future, initial_ego_state)
    
    def features_to_original_data(self, features, is_ego=False):
        if len(features.shape) == 1:
            if is_ego:
                return {
                    'x': features[0],
                    'y': features[1],
                    'vx': features[2] if len(features) > 2 else 0.0,
                    'vy': features[3] if len(features) > 3 else 0.0,
                    'psi_rad': features[2] if len(features) == 3 else features[4],
                    'length': features[5] if len(features) > 5 else 0.0,
                    'width': features[6] if len(features) > 6 else 0.0,
                }
            else:
                return {
                    'rel_x': features[0],
                    'rel_y': features[1],
                    'rel_vx': features[2] if len(features) > 2 else 0.0,
                    'rel_vy': features[3] if len(features) > 3 else 0.0,
                    'rel_psi': features[2] if len(features) == 3 else features[4],
                    'length': features[5] if len(features) > 5 else 0.0,
                    'width': features[6] if len(features) > 6 else 0.0,
                }
        else:
            return [self.features_to_original_data(feat, is_ego) for feat in features]
    
    def convert_relative_to_absolute(self, rel_data, ego_data):
        ego_x = ego_data['x'] if 'x' in ego_data else ego_data['abs_x']
        ego_y = ego_data['y'] if 'y' in ego_data else ego_data['abs_y']
        ego_vx = ego_data['vx'] if 'vx' in ego_data else ego_data['abs_vx']
        ego_vy = ego_data['vy'] if 'vy' in ego_data else ego_data['abs_vy']
        ego_psi = ego_data['psi_rad'] if 'psi_rad' in ego_data else ego_data['abs_psi']
        
        rotation_matrix = np.array([
            [math.cos(ego_psi), -math.sin(ego_psi)],
            [math.sin(ego_psi), math.cos(ego_psi)]
        ])
        
        rel_x = rel_data['rel_x'] if 'rel_x' in rel_data else rel_data[0]
        rel_y = rel_data['rel_y'] if 'rel_y' in rel_data else rel_data[1]
        
        rel_pos_local = np.array([rel_x, rel_y])
        rel_pos_global = rotation_matrix.dot(rel_pos_local)
        
        abs_x = ego_x + rel_pos_global[0]
        abs_y = ego_y + rel_pos_global[1]
        
        if 'rel_vx' in rel_data and 'rel_vy' in rel_data:
            rel_vel_local = np.array([rel_data['rel_vx'], rel_data['rel_vy']])
            rel_vel_global = rotation_matrix.dot(rel_vel_local)
            abs_vx = ego_vx + rel_vel_global[0]
            abs_vy = ego_vy + rel_vel_global[1]
        else:
            abs_vx = 0.0
            abs_vy = 0.0
        
        rel_psi = rel_data['rel_psi'] if 'rel_psi' in rel_data else rel_data[2]
        abs_psi = ego_psi + rel_psi
        abs_psi = (abs_psi + math.pi) % (2 * math.pi) - math.pi
        
        return {
            'x': abs_x,
            'y': abs_y,
            'vx': abs_vx,
            'vy': abs_vy,
            'psi_rad': abs_psi,
            'length': rel_data['length'] if 'length' in rel_data else 0.0,
            'width': rel_data['width'] if 'width' in rel_data else 0.0
        }
    
    def decode_and_convert_to_absolute(self, diff_features, initial_state):
        features = differential_decoding(diff_features)
        absolute_data = self.convert_relative_to_absolute(features[0], initial_state)
        return absolute_data
    
    def convert_predictions_to_absolute(self, predictions, ego_initial_state):
        batch_size = predictions.shape[0]
        seq_len = predictions.shape[1]
        
        absolute_trajectories = np.zeros((batch_size, seq_len, 3))
        
        for i in range(batch_size):
            absolute_trajectories[i] = convert_features_to_absolute(predictions[i], ego_initial_state[i])
        
        return absolute_trajectories
    
    def recover_absolute_data(self, diff_ego_history, diff_agent_history, diff_ego_future, diff_agent_future, initial_ego_state):
        ego_history_abs = differential_decoding(diff_ego_history)
        # ego_history_abs = convert_features_to_absolute(ego_history_rel[:, :3], initial_ego_state)
        
        last_ego_history_state = {'abs_x': ego_history_abs[-1, 0], 'abs_y': ego_history_abs[-1, 1], 'abs_psi': ego_history_abs[-1, 2]}
        ego_future_abs = differential_decoding(diff_ego_future)
        # ego_future_abs = convert_features_to_absolute(ego_future_rel, last_ego_history_state)
        
        agent_history_abs = np.zeros((diff_agent_history.shape[0], diff_agent_history.shape[1], 3))
        for agent_idx in range(diff_agent_history.shape[1]):
            agent_history_rel = differential_decoding(diff_agent_history[:, agent_idx, :3])
            agent_history_abs[:, agent_idx] = convert_features_to_absolute(agent_history_rel, initial_ego_state)
        
        agent_future_abs = np.zeros((diff_agent_future.shape[0], diff_agent_future.shape[1], 3))
        for agent_idx in range(diff_agent_future.shape[1]):
            agent_future_rel = differential_decoding(diff_agent_future[:, agent_idx])
            agent_future_abs[:, agent_idx] = convert_features_to_absolute(agent_future_rel, last_ego_history_state)
        
        return ego_history_abs, agent_history_abs, ego_future_abs, agent_future_abs

def convert_trajectories_to_sample_batch(trajectories, batch_size=4):
    if len(trajectories) < batch_size:
        raise ValueError(f"Not enough trajectories. Found {len(trajectories)}, need at least {batch_size}")
    
    sample_batch = {
        'ego_history': np.array([t['ego_history'] for t in trajectories[:batch_size]]),
        'agent_history': np.array([t['agent_history'] for t in trajectories[:batch_size]]),
        'ego_future': np.array([t['ego_future'] for t in trajectories[:batch_size]]),
        'agent_future': np.array([t['agent_future'] for t in trajectories[:batch_size]]),
        'diff_ego_history': np.array([t['diff_ego_history'] for t in trajectories[:batch_size]]),
        'diff_agent_history': np.array([t['diff_agent_history'] for t in trajectories[:batch_size]]),
        'diff_ego_future': np.array([t['diff_ego_future'] for t in trajectories[:batch_size]]),
        'diff_agent_future': np.array([t['diff_agent_future'] for t in trajectories[:batch_size]]),
        'meta': [{
            'file_id': t['file_id'],
            'ego_id': t['ego_id'],
            'start_frame': t['start_frame'],
            'middle_frame': t['middle_frame'],
            'end_frame': t['end_frame'],
            'ego_original': t['ego_original'],
            'agent_original': t['agent_original']
        } for t in trajectories[:batch_size]]
    }
    
    return sample_batch

def process_sample_batch(sample_batch, config=None):
    data_generator = TrajectoryDataGenerator(config)
    processed_batch = data_generator.process_batch_data(sample_batch)
    
    for i in range(len(processed_batch['meta'])):
        initial_ego_state = processed_batch['meta'][i]['ego_original'][0]
        diff_ego_history = processed_batch['diff_ego_history'][i]
        diff_agent_history = processed_batch['diff_agent_history'][i]
        diff_ego_future = processed_batch['diff_ego_future'][i]
        diff_agent_future = processed_batch['diff_agent_future'][i]
        
        ego_history_abs, agent_history_abs, ego_future_abs, agent_future_abs = data_generator.recover_absolute_data(
            diff_ego_history, diff_agent_history, diff_ego_future, diff_agent_future, initial_ego_state
        )
        
        original_ego_history = np.array([[state['abs_x'], state['abs_y'], state['abs_psi']] 
            for state in processed_batch['meta'][i]['ego_original'][:len(ego_history_abs)]])
        original_ego_future = np.array([[state['abs_x'], state['abs_y'], state['abs_psi']] 
            for state in processed_batch['meta'][i]['ego_original'][len(ego_history_abs):]])
        
        print(f"Sample {i}:")
        print("Recovered ego_history_abs:", ego_history_abs[:,[0 , 1 , 4]])
        print("Original ego_history:", original_ego_history)
        print("Recovered ego_future_abs:", ego_future_abs)
        print("Original ego_future:", original_ego_future)
    
    return processed_batch, data_generator
  
def save_trajectories_to_pickle(trajectories, file_path):
    """
    将轨迹数据保存为一个 pickle 文件。
    
    参数:
        trajectories: 要保存的轨迹数据，应为可序列化的对象（如列表、字典等）。
        file_path: 保存文件的路径，字符串类型。
    """
    # 获取文件路径的目录部分，并创建目录（如果不存在）
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    
    # 以二进制写模式打开文件并保存数据
    with open(file_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    # 打印保存成功的提示信息
    print(f"轨迹已保存至 {file_path}")

def get_train_data(trajectories):
    """
    获取训练数据
    """
    train_data = []
    
    for trajectory in trajectories:
        ego_history = trajectory['ego_history']
        agent_history = trajectory['agent_history']
        ego_future = trajectory['ego_future']
        agent_future = trajectory['agent_future']
        
        train_data.append({
            'diff_ego_history': ego_history,
            'diff_agent_history': agent_history,
            'diff_ego_future': ego_future,
            'diff_agent_future': agent_future
        })
    
    return train_data    

if __name__ == "__main__":
    intput_dir = r"data_process\behaviour\data"
    output_dir = r"Mamba_attn_agent_pred\src\datasets\data"
    
    
    config = {
        'history_sec': 1,
        'future_sec': 3,
        'fps': 10,
        'num_agents': 6,
        'input_dim': 7,
        'output_dim': 3,
        'sliding_step': 5,
        'min_seq_len': 40
    }
    
    # loaded_data = load_frame_data(os.path.join(intput_dir, "all_frame_data.pkl"))
    
    # data_generator = TrajectoryDataGenerator(config)
    # trajectories = data_generator.process_data(loaded_data)
    # save_trajectories_to_pickle(trajectories, os.path.join(output_dir, "train_trajectories.pkl"))
    
    # # 检验数据是否存在问题
    trajectories = load_frame_data(os.path.join(output_dir, "trajectories.pkl"))
    
    get_train_data(trajectories)
    
    
    # sample_batch = convert_trajectories_to_sample_batch(trajectories, batch_size=4)
    
    # processed_batch, _ = process_sample_batch(sample_batch, config)
    
    # print("Successfully processed sample batch")