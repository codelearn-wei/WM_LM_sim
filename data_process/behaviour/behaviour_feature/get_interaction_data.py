# 组织获得交互场景数据，建模和价值分析都基于此展开
import os
import pickle
import math
import matplotlib.pyplot as plt

def extract_nearby_vehicles_at_merging(lane_change_info, max_nearby_vehicles=6):
    """
    找到汇入点centerpoints时，周围最近的六辆车，在主车汇入前后5s的轨迹
    
    Args:
        lane_change_info (dict): 变道信息数据
        max_nearby_vehicles (int): 需要找到的周围车辆数量
    
    Returns:
        dict: 包含主车和周围车辆轨迹的信息
    """
    merging_trajectories = {}
    
    for file_id, file_data in lane_change_info.items():
        merging_trajectories[file_id] = {}
        
        for scene_id, scene_changes in file_data.items():
            merging_trajectories[file_id][scene_id] = []
            
            for lane_change in scene_changes:
                main_track_id = lane_change['track_id']
                center_timestamp = lane_change['crossing_point']['timestamp']
                
                # 提取主车在变道点的位置和朝向
                main_vehicle_position = None
                main_vehicle_heading = None
                
                for _, vehicle_data in lane_change['main_vehicle_frames']:
                    if vehicle_data['timestamp'] == center_timestamp:
                        main_vehicle_position = (vehicle_data['x'], vehicle_data['y'])
                        main_vehicle_heading = vehicle_data['psi_rad']
                        break
                
                if not main_vehicle_position:
                    continue
            
                start_ts = lane_change['lane_change_window']['start_timestamp']
                end_ts = lane_change['lane_change_window']['end_timestamp']
                
                # 提取主车轨迹
                main_vehicle_trajectory = []
                for frame_idx, vehicle_data in lane_change['main_vehicle_frames']:
                    ts = vehicle_data['timestamp']
                    if start_ts <= ts <= end_ts:
                        main_vehicle_trajectory.append({
                            'timestamp': ts,
                            'x': vehicle_data['x'],
                            'y': vehicle_data['y'],
                            'vx': vehicle_data.get('vx', 0),
                            'vy': vehicle_data.get('vy', 0),
                            'v': vehicle_data.get('v', math.sqrt(vehicle_data.get('vx', 0)**2 + vehicle_data.get('vy', 0)**2)),
                            'length': vehicle_data.get('length', 0),
                            'width': vehicle_data.get('width', 0),
                            'heading': vehicle_data.get('psi_rad', 0),
                            'frame_idx': frame_idx
                        })
                
                # 按照时间戳排序主车轨迹
                main_vehicle_trajectory.sort(key=lambda x: x['timestamp'])
                
                # 计算主车在变道点时的位置和朝向
                main_pos_at_center = None
                for traj_point in main_vehicle_trajectory:
                    if traj_point['timestamp'] == center_timestamp:
                        main_pos_at_center = (traj_point['x'], traj_point['y'])
                        break
                
                if not main_pos_at_center:
                    # 找不到主车在变道点的位置
                    continue
                
                # 计算所有周围车辆在变道点时与主车的距离
                surrounding_vehicles_at_center = []
                
                for surr_id, surr_frames in lane_change['surrounding_vehicles'].items():
                    # 寻找周围车辆在变道点时刻的位置
                    surr_pos_at_center = None
                    min_ts_diff = float('inf')
                    closest_surr_data = None
                    
                    for _, surr_data, _ in surr_frames:
                        ts_diff = abs(surr_data['timestamp'] - center_timestamp)
                        if ts_diff < min_ts_diff:
                            min_ts_diff = ts_diff
                            surr_pos_at_center = (surr_data['x'], surr_data['y'])
                            closest_surr_data = surr_data
                    
                    if surr_pos_at_center and closest_surr_data:
                        # 计算与主车的距离
                        distance = math.sqrt((surr_pos_at_center[0] - main_pos_at_center[0])**2 + 
                                            (surr_pos_at_center[1] - main_pos_at_center[1])**2)
                        
                        # 计算相对于主车的角度
                        rel_x = surr_pos_at_center[0] - main_pos_at_center[0]
                        rel_y = surr_pos_at_center[1] - main_pos_at_center[1]
                        
                        # 计算相对角度（考虑主车朝向）
                        angle = math.atan2(rel_y, rel_x)
                        if main_vehicle_heading is not None:
                            # 将角度转换到主车坐标系
                            rel_angle = (angle - main_vehicle_heading) % (2 * math.pi)
                        else:
                            rel_angle = angle
                        
                        # 判断车道类型（辅道/主道）
                        lane_type = closest_surr_data.get('lane_type', '')
                        
                        surrounding_vehicles_at_center.append({
                            'track_id': surr_id,
                            'distance': distance,
                            'rel_angle': rel_angle,
                            'rel_x': rel_x,
                            'rel_y': rel_y,
                            'position': (surr_pos_at_center[0], surr_pos_at_center[1]),
                            'lane_type': lane_type
                        })
                
                # 对周围车辆按距离排序
                surrounding_vehicles_at_center.sort(key=lambda x: x['distance'])
                
                # 只取最近的max_nearby_vehicles辆车
                closest_vehicles = surrounding_vehicles_at_center[:max_nearby_vehicles] if surrounding_vehicles_at_center else []
                
                # 分类车辆位置（主道前车LZV，主道后车BZV，辅道前车LFV，辅道后车BFV，主道后方第二辆车BBZV）
                classified_vehicles = {
                    'FV': {'track_id': main_track_id, 'trajectory': main_vehicle_trajectory},
                    'LFV': None,
                    'BFV': None,
                    'LZV': None,
                    'BZV': None,
                    'BBZV': None  # 新增：主道后方第二辆车
                }
                
                # 定义角度范围
                # 假设主车朝向是0度，前方是[-π/4, π/4]，右侧是[π/4, 3π/4]，后方是[3π/4, 5π/4]，左侧是[5π/4, 7π/4]
                FRONT_RANGE = (-math.pi/4, math.pi/4)
                RIGHT_RANGE = (math.pi/4, 3*math.pi/4)
                BACK_RANGE = (3*math.pi/4, 5*math.pi/4)
                LEFT_RANGE = (5*math.pi/4, 7*math.pi/4)
                
                # 辅助函数：检查角度是否在范围内
                def is_angle_in_range(angle, angle_range):
                    # 将角度标准化到[-π, π]
                    normalized_angle = ((angle + math.pi) % (2 * math.pi)) - math.pi
                    return angle_range[0] <= normalized_angle <= angle_range[1]
                
                # 临时存储主道后方车辆，用于识别BZV和BBZV
                main_lane_behind_vehicles = []
                
                # 分类周围车辆
                for vehicle in closest_vehicles:
                    # 提取该车辆的轨迹
                    surr_trajectory = []
                    surr_id = vehicle['track_id']
                    
                    for frame_idx, surr_data, dist in lane_change['surrounding_vehicles'][surr_id]:
                        ts = surr_data['timestamp']
                        if start_ts <= ts <= end_ts:
                            surr_trajectory.append({
                                'timestamp': ts,
                                'x': surr_data['x'],
                                'y': surr_data['y'],
                                'vx': surr_data.get('vx', 0),
                                'vy': surr_data.get('vy', 0),
                                'v': surr_data.get('v', math.sqrt(surr_data.get('vx', 0)**2 + surr_data.get('vy', 0)**2)),
                                'heading': surr_data.get('psi_rad', 0),
                                'length': vehicle_data.get('length', 0),
                                'width': vehicle_data.get('width', 0),
                                'frame_idx': frame_idx
                            })
                    
      
                    surr_trajectory.sort(key=lambda x: x['timestamp'])
                    

                    rel_angle = vehicle['rel_angle']
                    lane_type = vehicle['lane_type']
                    

                    rel_x = vehicle['rel_x']
                    rel_y = vehicle['rel_y']
                    

                    if main_vehicle_heading is not None:
  
                        main_dir_vector = (math.cos(main_vehicle_heading), math.sin(main_vehicle_heading))
                        
 
                        dot_product = rel_x * main_dir_vector[0] + rel_y * main_dir_vector[1]
                        is_in_front = dot_product > 0
                        
                        # 计算叉积来判断左右
                        cross_product = rel_x * main_dir_vector[1] - rel_y * main_dir_vector[0]
                        is_on_right = cross_product < 0
                        
                        # 根据车道类型和相对位置分类
                        if '变道车辆' in lane_type or lane_type.lower() == 'auxiliary':
                            if is_in_front:
                                classified_vehicles['LFV'] = {'track_id': surr_id, 'trajectory': surr_trajectory}
                            else:
                                classified_vehicles['BFV'] = {'track_id': surr_id, 'trajectory': surr_trajectory}
                        else:  # 主道
                            if is_in_front:
                                classified_vehicles['LZV'] = {'track_id': surr_id, 'trajectory': surr_trajectory}
                            else:
                                # 主道后方车辆，暂存以排序确定BZV和BBZV
                                main_lane_behind_vehicles.append({
                                    'track_id': surr_id, 
                                    'trajectory': surr_trajectory,
                                    'distance': vehicle['distance']
                                })
                    else:
                        # 如果没有主车朝向信息，使用近似判断
                        if '变道车辆' in lane_type or lane_type.lower() == 'auxiliary':
                            if is_angle_in_range(rel_angle, FRONT_RANGE):
                                classified_vehicles['LFV'] = {'track_id': surr_id, 'trajectory': surr_trajectory}
                            elif is_angle_in_range(rel_angle, BACK_RANGE):
                                classified_vehicles['BFV'] = {'track_id': surr_id, 'trajectory': surr_trajectory}
                        else:  # 主道
                            if is_angle_in_range(rel_angle, FRONT_RANGE) or is_angle_in_range(rel_angle, RIGHT_RANGE):
                                classified_vehicles['LZV'] = {'track_id': surr_id, 'trajectory': surr_trajectory}
                            elif is_angle_in_range(rel_angle, BACK_RANGE) or is_angle_in_range(rel_angle, LEFT_RANGE):
                                # 主道后方车辆，暂存以排序确定BZV和BBZV
                                main_lane_behind_vehicles.append({
                                    'track_id': surr_id, 
                                    'trajectory': surr_trajectory,
                                    'distance': vehicle['distance']
                                })
                
                # 根据距离对主道后方车辆排序，确定BZV和BBZV
                if main_lane_behind_vehicles:
                    main_lane_behind_vehicles.sort(key=lambda x: x['distance'])
                    
                    # 分配BZV（最近）
                    if len(main_lane_behind_vehicles) >= 1:
                        classified_vehicles['BZV'] = {
                            'track_id': main_lane_behind_vehicles[0]['track_id'],
                            'trajectory': main_lane_behind_vehicles[0]['trajectory']
                        }
                    
                    # 分配BBZV（第二近）
                    if len(main_lane_behind_vehicles) >= 2:
                        classified_vehicles['BBZV'] = {
                            'track_id': main_lane_behind_vehicles[1]['track_id'],
                            'trajectory': main_lane_behind_vehicles[1]['trajectory']
                        }
                
                # 添加到结果中
                merging_trajectories[file_id][scene_id].append({
                    'main_track_id': main_track_id,
                    'center_timestamp': center_timestamp,
                    'vehicles': classified_vehicles
                })
    
    return merging_trajectories

def visualize_merging_scene(merging_data, file_id, scene_id, merging_idx=0):
    """
    可视化汇入场景中各车辆的轨迹
    
    Args:
        merging_data (dict): 包含车辆轨迹的字典
        file_id (str): 文件ID
        scene_id (str): 场景ID
        merging_idx (int): 汇入事件索引
    """
    if file_id not in merging_data or scene_id not in merging_data[file_id] or not merging_data[file_id][scene_id]:
        print(f"未找到指定的数据: file_id={file_id}, scene_id={scene_id}")
        return
    
    if merging_idx >= len(merging_data[file_id][scene_id]):
        print(f"索引超出范围: merging_idx={merging_idx}, 可用范围: 0-{len(merging_data[file_id][scene_id])-1}")
        return
    
    merging_scene = merging_data[file_id][scene_id][merging_idx]
    vehicles = merging_scene['vehicles']
    
    plt.figure(figsize=(12, 8))
    
    # 定义颜色和标记
    vehicle_styles = {
        'FV': {'color': 'red', 'marker': 'o', 'label': 'FV (Main Vehicle)'},
        'LFV': {'color': 'blue', 'marker': 's', 'label': 'LFV (Auxiliary Lane Front)'},
        'BFV': {'color': 'green', 'marker': '^', 'label': 'BFV (Auxiliary Lane Behind)'},
        'LZV': {'color': 'purple', 'marker': 'd', 'label': 'LZV (Main Lane Front)'},
        'BZV': {'color': 'orange', 'marker': 'p', 'label': 'BZV (Main Lane Behind)'},
        'BBZV': {'color': 'brown', 'marker': 'x', 'label': 'BBZV (Main Lane Second Behind)'}  # 新增BBZV样式
    }
    
    # 绘制每辆车的轨迹
    for vehicle_type, vehicle_data in vehicles.items():
        if vehicle_data is None:
            continue
            
        trajectory = vehicle_data['trajectory']
        x_coords = [point['x'] for point in trajectory]
        y_coords = [point['y'] for point in trajectory]
        
        style = vehicle_styles.get(vehicle_type, {'color': 'black', 'marker': '.', 'label': vehicle_type})
        
        plt.plot(x_coords, y_coords, '-', color=style['color'], label=style['label'])
        plt.scatter(x_coords, y_coords, marker=style['marker'], color=style['color'], s=30)
        
        # 标记起点和终点
        if x_coords:
            plt.scatter(x_coords[0], y_coords[0], marker='>', color=style['color'], s=100)
            plt.scatter(x_coords[-1], y_coords[-1], marker='<', color=style['color'], s=100)
    
    # 标记汇入点
    center_time = merging_scene['center_timestamp']
    main_traj = vehicles['FV']['trajectory']
    center_point = None
    
    for point in main_traj:
        if point['timestamp'] == center_time:
            center_point = (point['x'], point['y'])
            break
    
    if center_point:
        plt.scatter(center_point[0], center_point[1], marker='*', color='black', s=200, label='Merging Point')
    
    plt.title(f'Merging Scene Visualization\nFile: {file_id}, Scene: {scene_id}')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    
    return plt

def main():
    # 加载变道信息数据
    data_path = r"data_process\behaviour\data\lane_change_info_with_surroundings.pkl"
    result_path = r"data_process\behaviour\data"
    
    # 创建输出目录
    os.makedirs(result_path, exist_ok=True)
    
    # 加载数据
    with open(data_path, 'rb') as f:
        lane_change_info = pickle.load(f)
    print("已加载变道信息数据")
    
    # 提取汇入点附近车辆
    print("提取汇入点附近车辆数据...")
    merging_trajectories = extract_nearby_vehicles_at_merging(lane_change_info)
    
    # 计算所有提取到的汇入场景数量
    total_scenarios = 0
    complete_scenarios = 0  # 同时包含所有6种车辆的场景
    scenarios_with_bbzv = 0  # 包含BBZV的场景
    
    for file_id, file_data in merging_trajectories.items():
        for scene_id, scene_changes in file_data.items():
            for scenario in scene_changes:
                total_scenarios += 1
                vehicles = scenario['vehicles']
                if all(vehicles[vtype] is not None for vtype in ['FV', 'LFV', 'BFV', 'LZV', 'BZV', 'BBZV']):
                    complete_scenarios += 1
                if vehicles['BBZV'] is not None:
                    scenarios_with_bbzv += 1
    
    print(f"共提取到 {total_scenarios} 个汇入场景")
    print(f"其中 {complete_scenarios} 个场景同时包含所有6种车辆")
    print(f"有 {scenarios_with_bbzv} 个场景包含BBZV车辆")
    
    # 保存结果
    output_file = os.path.join(result_path, 'merging_vehicles_interactions_trajectories.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(merging_trajectories, f)
    
    print(f"汇入点车辆轨迹已保存到 {output_file}")
    
    # 可视化示例（如果有数据）
    if total_scenarios > 0:
        # 尝试找到包含BBZV的场景进行可视化
        found_bbzv_scene = False
        for file_id, file_data in merging_trajectories.items():
            for scene_id, scene_changes in file_data.items():
                for i, scenario in enumerate(scene_changes):
                    if scenario['vehicles']['BBZV'] is not None:
                        # 可视化并保存图像
                        plt = visualize_merging_scene(merging_trajectories, file_id, scene_id, i)
                        if plt:
                            fig_path = os.path.join(result_path, f'merging_scene_with_bbzv_{file_id}_{scene_id}_{i}.png')
                            plt.savefig(fig_path)
                            plt.close()
                            print(f"包含BBZV的场景可视化已保存到 {fig_path}")
                            found_bbzv_scene = True
                            break
                if found_bbzv_scene:
                    break
            if found_bbzv_scene:
                break
        
        # 如果没有找到包含BBZV的场景，就可视化第一个有效场景
        if not found_bbzv_scene:
            for file_id, file_data in merging_trajectories.items():
                for scene_id, scene_changes in file_data.items():
                    if scene_changes:
                        # 可视化并保存图像
                        plt = visualize_merging_scene(merging_trajectories, file_id, scene_id)
                        if plt:
                            fig_path = os.path.join(result_path, f'merging_scene_{file_id}_{scene_id}.png')
                            plt.savefig(fig_path)
                            plt.close()
                            print(f"示例场景可视化已保存到 {fig_path}")
                            break
                if plt:
                    break
    
    print("分析完成。")

if __name__ == "__main__":
    main()