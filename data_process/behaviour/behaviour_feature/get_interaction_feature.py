import math
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_interaction_features(merging_trajectories):
    """
    提取车辆之间的交互特征，并记录相应时刻的车辆轨迹数据
    
    Args:
        merging_trajectories (dict): 包含车辆轨迹的字典数据
    
    Returns:
        dict: 包含各种交互特征和车辆轨迹的字典
    """
    all_features = {}
    
    for file_id, file_data in merging_trajectories.items():
        all_features[file_id] = {}
        
        for scene_id, scene_changes in file_data.items():
            all_features[file_id][scene_id] = []
            
            for scenario in scene_changes:
                vehicles = scenario['vehicles']
                main_track_id = scenario['main_track_id']
                center_timestamp = scenario['center_timestamp']
                
                # 只处理至少有FV和BZV的场景
                if vehicles['FV'] is None or vehicles['BZV'] is None:
                    continue
                
                # 提取时间点列表（所有车辆轨迹时间点的并集）
                all_timestamps = set()
                for v_type, v_data in vehicles.items():
                    if v_data is not None:
                        all_timestamps.update([point['timestamp'] for point in v_data['trajectory']])
                
                timestamps = sorted(list(all_timestamps))
                
                # 初始化特征字典，添加轨迹数据存储
                features = {
                    'main_track_id': main_track_id,
                    'center_timestamp': center_timestamp,
                    'timestamps': timestamps,
                    'features_by_time': {ts: {} for ts in timestamps},
                    'trajectories_by_time': {ts: {} for ts in timestamps},  # 新增：存储轨迹数据
                    'global_features': {}
                }
                
                # 为每个时间点计算特征并存储轨迹数据
                for ts in timestamps:
                    ts_features = features['features_by_time'][ts]
                    ts_trajectories = features['trajectories_by_time'][ts]  # 新增：轨迹数据字典
                    
                    # 存储主要车辆的轨迹点
                    for vehicle_type in ['FV', 'BZV', 'BBZV', 'LZV']:
                        if vehicles[vehicle_type] is not None:
                            point = find_closest_point(vehicles[vehicle_type]['trajectory'], ts)
                            if point:
                                # 存储该车辆在此时刻的轨迹点
                                ts_trajectories[vehicle_type] = {
                                    'x': point['x'],
                                    'y': point['y'],
                                    'v': point['v'],
                                    'length': point['length'],
                                    'width': point['width'],
                                    'heading': point['heading'],
                                    'timestamp': point['timestamp']
                                }
                                # 如果有加速度信息，也存储
                                if 'ax' in point and 'ay' in point:
                                    ts_trajectories[vehicle_type]['ax'] = point['ax']
                                    ts_trajectories[vehicle_type]['ay'] = point['ay']
                    
                    # 1. FV和BZV之间的交互特征
                    fv_bzv_features = extract_pair_features(vehicles['FV'], vehicles['BZV'], ts)
                    if fv_bzv_features:
                        ts_features['FV_BZV'] = fv_bzv_features
                    
                    # 2. BZV和BBZV之间的交互特征
                    if vehicles['BBZV'] is not None:
                        bzv_bbzv_features = extract_pair_features(vehicles['BZV'], vehicles['BBZV'], ts)
                        if bzv_bbzv_features:
                            ts_features['BZV_BBZV'] = bzv_bbzv_features
                    
                    # 3. BZV和LZV之间的特征
                    if vehicles['LZV'] is not None:
                        bzv_lzv_features = extract_pair_features(vehicles['BZV'], vehicles['LZV'], ts)
                        if bzv_lzv_features:
                            ts_features['BZV_LZV'] = bzv_lzv_features
                    
                    # 4. FV和LZV之间的特征 (辅助分析)
                    if vehicles['LZV'] is not None:
                        fv_lzv_features = extract_pair_features(vehicles['FV'], vehicles['LZV'], ts)
                        if fv_lzv_features:
                            ts_features['FV_LZV'] = fv_lzv_features
                
                # 计算全局特征（整个变道过程的特征）
                global_features = compute_global_features(vehicles, timestamps, features['features_by_time'])
                features['global_features'] = global_features
                
                # 添加到结果中
                all_features[file_id][scene_id].append(features)
    
    return all_features

def extract_pair_features(vehicle1, vehicle2, timestamp):
    """
    提取两辆车在特定时间点的交互特征，考虑车辆尺寸
    
    Args:
        vehicle1 (dict): 第一辆车的数据
        vehicle2 (dict): 第二辆车的数据
        timestamp (float): 时间点
    
    Returns:
        dict: 交互特征
    """
    if vehicle1 is None or vehicle2 is None:
        return None
    
    # 在轨迹中找到最接近指定时间点的数据
    v1_point = find_closest_point(vehicle1['trajectory'], timestamp)
    v2_point = find_closest_point(vehicle2['trajectory'], timestamp)
    
    if not v1_point or not v2_point:
        return None
    
    # 计算基本特征 - 中心点之间的距离
    dx = v2_point['x'] - v1_point['x']
    dy = v2_point['y'] - v1_point['y']
    center_distance = math.sqrt(dx**2 + dy**2)
    
    # 获取车辆尺寸
    v1_length = v1_point['length']
    v1_width = v1_point['width']
    v2_length = v2_point['length']
    v2_width = v2_point['width']
    
    # 计算考虑车辆尺寸的距离（车身边缘之间的距离）
    edge_distance = calculate_edge_distance(
        v1_point['x'], v1_point['y'], v1_point['heading'], v1_length, v1_width,
        v2_point['x'], v2_point['y'], v2_point['heading'], v2_length, v2_width
    )
    
    # 速度差
    speed_diff = v2_point['v'] - v1_point['v']
    
    # 相对角度
    rel_angle = math.atan2(dy, dx)
    
    # 根据两车的朝向计算是否在同一车道
    heading_diff = abs(v1_point['heading'] - v2_point['heading'])
    heading_diff = min(heading_diff, 2*math.pi - heading_diff)  # 确保不超过180度
    same_direction = heading_diff < math.pi/4  # 朝向差小于45度
    
    # 计算两车的时间间隔 (Time-To-Collision, TTC)
    # 使用边缘距离而不是中心距离计算TTC
    ttc = None
    if abs(speed_diff) > 0.1:  # 避免除以接近0的数
        # 如果第二辆车比第一辆车慢，TTC为负数，表示不会碰撞
        if speed_diff < 0:
            ttc = -edge_distance / abs(speed_diff)
        else:
            ttc = edge_distance / speed_diff
    
    # 计算纵向和横向距离（相对于车辆1的朝向）
    v1_heading = v1_point['heading']
    long_dist = dx * math.cos(v1_heading) + dy * math.sin(v1_heading)
    lat_dist = -dx * math.sin(v1_heading) + dy * math.cos(v1_heading)
    
    # 计算考虑车身尺寸的纵向和横向距离
    # 这里计算车身边缘之间的距离，而不是中心点之间的距离
    edge_long_dist, edge_lat_dist = calculate_edge_longitudinal_lateral_distance(
        v1_point['x'], v1_point['y'], v1_point['heading'], v1_length, v1_width,
        v2_point['x'], v2_point['y'], v2_point['heading'], v2_length, v2_width
    )
    
    # 计算相对加速度（如果有加速度数据）
    rel_acc = None
    if 'ax' in v1_point and 'ay' in v1_point and 'ax' in v2_point and 'ay' in v2_point:
        a1 = math.sqrt(v1_point['ax']**2 + v1_point['ay']**2)
        a2 = math.sqrt(v2_point['ax']**2 + v2_point['ay']**2)
        rel_acc = a2 - a1
    
    # 汇总特征
    features = {
        'center_distance': center_distance,  # 中心点之间的距离
        'edge_distance': edge_distance,      # 车身边缘之间的距离
        'speed_diff': speed_diff,            # 速度差 (v2-v1)
        'rel_angle': rel_angle,              # 相对角度
        'heading_diff': heading_diff,        # 朝向差异
        'same_direction': same_direction,    # 是否同向
        'ttc': ttc,                          # 碰撞时间
        'long_dist': long_dist,              # 中心点纵向距离
        'lat_dist': lat_dist,                # 中心点横向距离
        'edge_long_dist': edge_long_dist,    # 边缘纵向距离
        'edge_lat_dist': edge_lat_dist,      # 边缘横向距离
        'rel_acc': rel_acc,                  # 相对加速度
        'v1_speed': v1_point['v'],           # 车辆1速度
        'v2_speed': v2_point['v'],           # 车辆2速度
        'v1_pos': (v1_point['x'], v1_point['y']),  # 车辆1位置
        'v2_pos': (v2_point['x'], v2_point['y']),  # 车辆2位置
        'v1_dim': (v1_length, v1_width),     # 车辆1尺寸
        'v2_dim': (v2_length, v2_width),     # 车辆2尺寸
    }
    
    return features

def calculate_edge_distance(x1, y1, heading1, length1, width1, x2, y2, heading2, length2, width2):
    """
    计算两个矩形（车辆）边缘之间的最短距离
    
    简化版本：使用近似计算方法，考虑车辆的长度和宽度的投影
    
    Args:
        x1, y1: 车辆1中心点坐标
        heading1: 车辆1朝向角度（弧度）
        length1, width1: 车辆1的长度和宽度
        x2, y2: 车辆2中心点坐标
        heading2: 车辆2朝向角度（弧度）
        length2, width2: 车辆2的长度和宽度
    
    Returns:
        float: 车辆边缘之间的最短距离
    """
    # 计算中心点之间的距离
    dx = x2 - x1
    dy = y2 - y1
    center_distance = math.sqrt(dx**2 + dy**2)
    
    # 计算两车连线方向与车辆1朝向的夹角
    angle_between = math.atan2(dy, dx) - heading1
    
    # 计算车辆1在连线方向上的半长度
    half_length1 = max(
        abs(length1/2 * math.cos(angle_between)),
        abs(width1/2 * math.sin(angle_between))
    )
    
    # 计算两车连线方向与车辆2朝向的夹角
    angle_between = math.atan2(-dy, -dx) - heading2
    
    # 计算车辆2在连线方向上的半长度
    half_length2 = max(
        abs(length2/2 * math.cos(angle_between)),
        abs(width2/2 * math.sin(angle_between))
    )
    
    # 计算边缘之间的距离
    edge_distance = center_distance - half_length1 - half_length2
    
    # 如果距离为负，说明车辆重叠，返回0
    return max(0, edge_distance)

def calculate_edge_longitudinal_lateral_distance(x1, y1, heading1, length1, width1, x2, y2, heading2, length2, width2):
    """
    计算考虑车身尺寸的纵向和横向距离
    
    Args:
        x1, y1: 车辆1中心点坐标
        heading1: 车辆1朝向角度（弧度）
        length1, width1: 车辆1的长度和宽度
        x2, y2: 车辆2中心点坐标
        heading2: 车辆2朝向角度（弧度）
        length2, width2: 车辆2的长度和宽度
    
    Returns:
        tuple: (边缘纵向距离, 边缘横向距离)
    """
    # 计算中心点之间的纵向和横向距离
    dx = x2 - x1
    dy = y2 - y1
    
    # 相对于车辆1坐标系的纵向和横向距离
    long_dist = dx * math.cos(heading1) + dy * math.sin(heading1)
    lat_dist = -dx * math.sin(heading1) + dy * math.cos(heading1)
    
    # 确定车辆相对位置关系
    # 车辆2在车辆1前方还是后方
    front_or_rear = 1 if long_dist >= 0 else -1
    # 车辆2在车辆1左侧还是右侧
    left_or_right = 1 if lat_dist >= 0 else -1
    
    # 计算车辆1和车辆2在纵向上的半长度
    half_length1 = length1/2
    half_length2 = length2/2 * abs(math.cos(heading2 - heading1)) + width2/2 * abs(math.sin(heading2 - heading1))
    
    # 计算车辆1和车辆2在横向上的半宽度
    half_width1 = width1/2
    half_width2 = width2/2 * abs(math.cos(heading2 - heading1)) + length2/2 * abs(math.sin(heading2 - heading1))
    
    # 计算考虑车身尺寸的纵向距离
    edge_long_dist = abs(long_dist) - front_or_rear * half_length1 - half_length2
    edge_long_dist = max(0, edge_long_dist) * front_or_rear
    
    # 计算考虑车身尺寸的横向距离
    edge_lat_dist = abs(lat_dist) - left_or_right * half_width1 - half_width2
    edge_lat_dist = max(0, edge_lat_dist) * left_or_right
    
    return edge_long_dist, edge_lat_dist

def find_closest_point(trajectory, timestamp):
    """
    在轨迹中找到最接近指定时间点的数据点
    
    Args:
        trajectory (list): 轨迹数据列表
        timestamp (float): 目标时间点
    
    Returns:
        dict: 最接近的数据点
    """
    if not trajectory:
        return None
    
    # 找到时间差最小的点
    closest_point = min(trajectory, key=lambda p: abs(p['timestamp'] - timestamp))
    
    # 如果时间差太大（例如超过1秒），则认为没有合适的点
    if abs(closest_point['timestamp'] - timestamp) > 1.0:
        return None
    
    return closest_point
 
def compute_global_features(vehicles, timestamps, features_by_time):
    """
    计算全局交互特征
    
    Args:
        vehicles (dict): 所有车辆数据
        timestamps (list): 时间点列表
        features_by_time (dict): 按时间点存储的特征
    
    Returns:
        dict: 全局特征
    """
    global_features = {}
    
    # 1. FV和BZV之间的关键全局特征
    fv_bzv_features = []
    for ts in timestamps:
        if 'FV_BZV' in features_by_time[ts]:
            fv_bzv_features.append(features_by_time[ts]['FV_BZV'])
    
    if fv_bzv_features:
        # 最小距离及其发生时间 (使用车身边缘距离)
        min_edge_dist_idx = min(range(len(fv_bzv_features)), key=lambda i: fv_bzv_features[i]['edge_distance'])
        min_edge_dist = fv_bzv_features[min_edge_dist_idx]['edge_distance']
        min_edge_dist_time = timestamps[min_edge_dist_idx]
        
        # 最小TTC及其发生时间
        ttc_values = [(i, feat['ttc']) for i, feat in enumerate(fv_bzv_features) if feat['ttc'] is not None and feat['ttc'] > 0]
        min_ttc = None
        min_ttc_time = None
        if ttc_values:
            min_ttc_idx, min_ttc = min(ttc_values, key=lambda x: x[1])
            min_ttc_time = timestamps[min_ttc_idx]
        
        # 计算平均和最大速度差
        speed_diffs = [feat['speed_diff'] for feat in fv_bzv_features]
        avg_speed_diff = sum(speed_diffs) / len(speed_diffs) if speed_diffs else None
        max_speed_diff = max(speed_diffs) if speed_diffs else None
        min_speed_diff = min(speed_diffs) if speed_diffs else None
        
        # 横向距离变化范围（车道变换幅度），使用边缘横向距离
        edge_lat_dists = [feat['edge_lat_dist'] for feat in fv_bzv_features]
        edge_lat_dist_range = max(edge_lat_dists) - min(edge_lat_dists) if edge_lat_dists else None
        
        global_features['FV_BZV'] = {
            'min_center_distance': fv_bzv_features[min_edge_dist_idx]['center_distance'],  # 保留中心点距离做对比
            'min_edge_distance': min_edge_dist,                                            # 边缘最小距离
            'min_distance_time': min_edge_dist_time,
            'min_ttc': min_ttc,
            'min_ttc_time': min_ttc_time,
            'avg_speed_diff': avg_speed_diff,
            'max_speed_diff': max_speed_diff,
            'min_speed_diff': min_speed_diff,
            'edge_lat_dist_range': edge_lat_dist_range,
        }
    
    # 2. BZV和BBZV之间的关键全局特征
    if vehicles['BBZV'] is not None:
        bzv_bbzv_features = []
        for ts in timestamps:
            if 'BZV_BBZV' in features_by_time[ts]:
                bzv_bbzv_features.append(features_by_time[ts]['BZV_BBZV'])
        
        if bzv_bbzv_features:
            # 最小距离及其发生时间
            min_edge_dist_idx = min(range(len(bzv_bbzv_features)), key=lambda i: bzv_bbzv_features[i]['edge_distance'])
            min_edge_dist = bzv_bbzv_features[min_edge_dist_idx]['edge_distance']
            min_edge_dist_time = timestamps[min_edge_dist_idx]
            
            # 计算平均和最大速度差
            speed_diffs = [feat['speed_diff'] for feat in bzv_bbzv_features]
            avg_speed_diff = sum(speed_diffs) / len(speed_diffs) if speed_diffs else None
            max_speed_diff = max(speed_diffs) if speed_diffs else None
            
            global_features['BZV_BBZV'] = {
                'min_center_distance': bzv_bbzv_features[min_edge_dist_idx]['center_distance'],
                'min_edge_distance': min_edge_dist,
                'min_distance_time': min_edge_dist_time,
                'avg_speed_diff': avg_speed_diff,
                'max_speed_diff': max_speed_diff,
            }
    
    # 3. BZV和LZV之间的关键全局特征
    if vehicles['LZV'] is not None:
        bzv_lzv_features = []
        for ts in timestamps:
            if 'BZV_LZV' in features_by_time[ts]:
                bzv_lzv_features.append(features_by_time[ts]['BZV_LZV'])
        
        if bzv_lzv_features:
            # 计算间隙大小 (gap size) - 使用边缘纵向距离
            edge_long_dists = [feat['edge_long_dist'] for feat in bzv_lzv_features]
            avg_gap_size = sum(edge_long_dists) / len(edge_long_dists) if edge_long_dists else None
            min_gap_size = min(edge_long_dists) if edge_long_dists else None
            
            global_features['BZV_LZV'] = {
                'avg_gap_size': avg_gap_size,
                'min_gap_size': min_gap_size,
            }
    
    # 4. 计算汇入窗口特征
    if vehicles['LZV'] is not None and 'FV_LZV' in features_by_time[timestamps[0]] and 'FV_BZV' in features_by_time[timestamps[0]]:
        # 计算汇入窗口大小（LZV和BZV之间的纵向距离）- 使用边缘距离
        merging_window_sizes = []
        for ts in timestamps:
            if 'FV_LZV' in features_by_time[ts] and 'FV_BZV' in features_by_time[ts]:
                lzv_edge_long_dist = features_by_time[ts]['FV_LZV']['edge_long_dist']
                bzv_edge_long_dist = features_by_time[ts]['FV_BZV']['edge_long_dist']
                # 确保LZV在前方，BZV在后方
                if lzv_edge_long_dist > 0 and bzv_edge_long_dist < 0:
                    window_size = lzv_edge_long_dist - bzv_edge_long_dist
                    merging_window_sizes.append((ts, window_size))
        
        if merging_window_sizes:
            # 汇入窗口大小及变化
            avg_window_size = sum(w[1] for w in merging_window_sizes) / len(merging_window_sizes)
            min_window_size = min(merging_window_sizes, key=lambda x: x[1])
            max_window_size = max(merging_window_sizes, key=lambda x: x[1])
            
            global_features['merging_window'] = {
                'avg_size': avg_window_size,
                'min_size': min_window_size[1],
                'min_size_time': min_window_size[0],
                'max_size': max_window_size[1],
                'max_size_time': max_window_size[0],
            }
    
    return global_features

def main():
    # 加载之前保存的汇入轨迹数据
    data_path = r"data_process\behaviour\data\merging_vehicles_trajectories.pkl"
    output_dir = r"data_process\behaviour\data"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    with open(data_path, 'rb') as f:
        merging_trajectories = pickle.load(f)
    print("已加载汇入轨迹数据")
    
    # 提取交互特征
    print("提取交互特征与车辆轨迹...")
    interaction_features = extract_interaction_features(merging_trajectories)
    
    # 保存交互特征和轨迹数据
    output_file = os.path.join(output_dir, 'interaction_features.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(interaction_features, f)
    print(f"交互特征和车辆轨迹已保存到 {output_file}")
    

if __name__ == "__main__":
    main()