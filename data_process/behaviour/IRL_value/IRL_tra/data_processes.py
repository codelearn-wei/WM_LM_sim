import pickle
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator

def get_LM_feature(trajectories):
    """
    从轨迹中提取特征，用于车辆变道IRL建模
    
    :param trajectories: 包含多车轨迹的字典，键为车辆标识，值为轨迹数据
    :return: 特征数据列表，每个时间步一组特征
    """
    # 提取各车辆轨迹，处理可能为None的情况
    fv_tra = trajectories['vehicles'].get('FV')      # 变道车辆 (必须存在)
    if fv_tra is None:
        raise ValueError("FV (变道车辆) is required but not found in trajectory data")
    
    # 其他车辆可以为None
    lfv_tra = trajectories['vehicles'].get('LFV')    # 前车
    lzv_tra = trajectories['vehicles'].get('LZV')    # 目标车道车辆
    bzv_tra = trajectories['vehicles'].get('BZV')    # 后方车辆
    bbzv_tra = trajectories['vehicles'].get('BBZV')  # 更远后方车辆
    
    # 获取轨迹时间步数
    n_steps = len(fv_tra['trajectory'])
    
    # 初始化特征列表
    features = []
    
    # 初始化FV轨迹列表 - 新增
    fv_trajectory = []
    
    # 对每个时间步进行特征提取
    for t in range(n_steps):
        # 获取当前时间步的变道车辆状态 (必须存在)
        fv_state = fv_tra['trajectory'][t]
        
        # 将FV轨迹点添加到轨迹列表 - 新增
        fv_trajectory.append({
            'x': fv_state['x'],
            'y': fv_state['y'],
            'timestamp': fv_state['timestamp']
        })
        
        # 获取其他车辆同一时间步的状态（如果存在）
        lfv_state = lfv_tra['trajectory'][t] if (lfv_tra is not None and t < len(lfv_tra['trajectory'])) else None
        lzv_state = lzv_tra['trajectory'][t] if (lzv_tra is not None and t < len(lzv_tra['trajectory'])) else None
        bzv_state = bzv_tra['trajectory'][t] if (bzv_tra is not None and t < len(bzv_tra['trajectory'])) else None
        bbzv_state = bbzv_tra['trajectory'][t] if (bbzv_tra is not None and t < len(bbzv_tra['trajectory'])) else None
        
        # 初始化当前时间步的特征字典
        feature_dict = {'timestamp': fv_state['timestamp']}
        
        # 1. 效率特征
        # 1.1 本车速度
        feature_dict['fv_speed'] = fv_state['v']
        
        # 2. 舒适度特征
        # 2.1 计算加速度（如果不是第一帧）
        if t > 0:
            prev_state = fv_tra['trajectory'][t-1]
            dt = (fv_state['timestamp'] - prev_state['timestamp']) / 1000.0  # 转换为秒
            if dt > 0:
                acc_x = (fv_state['vx'] - prev_state['vx']) / dt
                acc_y = (fv_state['vy'] - prev_state['vy']) / dt
                acceleration = np.sqrt(acc_x**2 + acc_y**2)
                feature_dict['acceleration'] = acceleration
            else:
                feature_dict['acceleration'] = 0
        else:
            feature_dict['acceleration'] = 0
        
        # 2.2 计算加加速度（如果至少有三帧）
        if t > 1:
            prev_acc = feature_dict['acceleration']
            prev_prev_state = fv_tra['trajectory'][t-2]
            prev_state = fv_tra['trajectory'][t-1]
            
            dt_prev = (prev_state['timestamp'] - prev_prev_state['timestamp']) / 1000.0
            if dt_prev > 0:
                prev_acc_x = (prev_state['vx'] - prev_prev_state['vx']) / dt_prev
                prev_acc_y = (prev_state['vy'] - prev_prev_state['vy']) / dt_prev
                prev_acceleration = np.sqrt(prev_acc_x**2 + prev_acc_y**2)
                
                dt = (fv_state['timestamp'] - prev_state['timestamp']) / 1000.0
                if dt > 0:
                    jerk = (feature_dict['acceleration'] - prev_acceleration) / dt
                    feature_dict['jerk'] = jerk
                else:
                    feature_dict['jerk'] = 0
            else:
                feature_dict['jerk'] = 0
        else:
            feature_dict['jerk'] = 0
        
        # 3. 风险特征 - 全部使用THW而不是TTC
        # 3.1 计算THW (FV和LZV之间) - 如果LZV存在，否则为0
        if lzv_state:
            thw = calculate_thw(fv_state, lzv_state)
            feature_dict['thw_fv_lzv'] = thw
        else:
            feature_dict['thw_fv_lzv'] = 0  # 目标车道无车辆时，使用0填充
        
        # 3.2 计算THW (FV和BZV之间) - 如果BZV存在，否则为0
        if bzv_state:
            thw = calculate_thw(bzv_state, fv_state)  # 注意顺序，BZV在FV后方
            feature_dict['thw_bzv_fv'] = thw
        else:
            feature_dict['thw_bzv_fv'] = 0  # 后方无车辆时，使用0填充
        
        # 4. 角度特征
        # 4.1 计算FV和BZV之间的夹角 - 如果BZV存在，否则为0
        if bzv_state:
            angle_diff = calculate_angle_difference(fv_state['heading'], bzv_state['heading'])
            feature_dict['angle_diff_fv_bzv'] = angle_diff
        else:
            feature_dict['angle_diff_fv_bzv'] = 0  # 后方无车辆时，使用0填充
        
        # 5. 其他相关特征 - 可以添加与LFV和BBZV相关的特征
        # 与前车LFV相关的特征
        if lfv_state:
            # 例如，计算与前车的距离
            dist_to_lfv = calculate_distance(fv_state['x'], fv_state['y'], lfv_state['x'], lfv_state['y'])
            feature_dict['dist_to_lfv'] = dist_to_lfv
            
            # 与前车的相对速度
            rel_speed_lfv = fv_state['v'] - lfv_state['v']
            feature_dict['rel_speed_lfv'] = rel_speed_lfv
            
            # 与前车的THW
            thw_lfv = calculate_thw(fv_state, lfv_state)
            feature_dict['thw_fv_lfv'] = thw_lfv
        else:
            feature_dict['dist_to_lfv'] = 0
            feature_dict['rel_speed_lfv'] = 0
            feature_dict['thw_fv_lfv'] = 0
        
        # 与更远后方车辆BBZV相关的特征
        if bbzv_state:
            # 例如，计算与更远后方车辆的距离
            dist_to_bbzv = calculate_distance(fv_state['x'], fv_state['y'], bbzv_state['x'], bbzv_state['y'])
            feature_dict['dist_to_bbzv'] = dist_to_bbzv
            
            # 与更远后方车辆的相对速度
            rel_speed_bbzv = fv_state['v'] - bbzv_state['v']
            feature_dict['rel_speed_bbzv'] = rel_speed_bbzv
            
            # 与更远后方车辆的THW
            thw_bbzv = calculate_thw(bbzv_state, fv_state)  # 注意顺序，BBZV在FV后方
            feature_dict['thw_bbzv_fv'] = thw_bbzv
        else:
            feature_dict['dist_to_bbzv'] = 0
            feature_dict['rel_speed_bbzv'] = 0
            feature_dict['thw_bbzv_fv'] = 0
        
        # 将当前时间步的特征添加到特征列表
        features.append(feature_dict)
    
    # 对加速度和加加速度进行平滑处理，减少噪声
    smooth_acceleration(features)
    
    # 将FV轨迹添加为一个单独的键 - 新增
    features_with_trajectory = {
        'features': features,
        'fv_trajectory': fv_trajectory
    }
    
    return features_with_trajectory

def calculate_distance(x1, y1, x2, y2):
    """计算两点之间的欧式距离"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_thw(front_vehicle, following_vehicle):
    """
    计算Time Headway (THW)
    考虑车辆在同一车道上的前后位置
    
    :param front_vehicle: 前方车辆状态
    :param following_vehicle: 后方车辆状态
    :return: THW值，单位为秒
    """
    # 提取位置和速度
    x_front, y_front = front_vehicle['x'], front_vehicle['y']
    x_follow, y_follow = following_vehicle['x'], following_vehicle['y']
    v_follow = following_vehicle['v']
    
    # 如果速度接近零，返回最大值
    if v_follow < 0.1:
        return float('inf')
    
    # 计算两车中心点间距离
    center_dist = calculate_distance(x_follow, y_follow, x_front, y_front)
    
    # 考虑车长，计算车头到车尾距离
    # 简化处理，实际应基于heading计算投影距离
    length_margin = (following_vehicle['length'] + front_vehicle['length']) / 2
    
    # 计算THW
    thw = (center_dist - length_margin) / v_follow
    
    # THW不应为负
    return max(0, thw)

def calculate_angle_difference(heading1, heading2):
    """计算两个朝向角之间的夹角（弧度）"""
    # 确保角度在[-π, π]范围内
    diff = heading1 - heading2
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return abs(diff)

def get_vehicle_corners(vehicle):
    """
    计算车辆的四个角点坐标
    基于中心点、长宽和朝向
    """
    x, y = vehicle['x'], vehicle['y']
    length, width = vehicle['length'], vehicle['width']
    heading = vehicle['heading']
    
    # 计算四个角点相对于中心的偏移
    half_length = length / 2
    half_width = width / 2
    
    # 旋转矩阵
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    
    # 计算四个角点（前左、前右、后右、后左）
    corners = []
    
    # 前左
    dx = half_length * cos_h - half_width * sin_h
    dy = half_length * sin_h + half_width * cos_h
    corners.append((x + dx, y + dy))
    
    # 前右
    dx = half_length * cos_h + half_width * sin_h
    dy = half_length * sin_h - half_width * cos_h
    corners.append((x + dx, y + dy))
    
    # 后右
    dx = -half_length * cos_h + half_width * sin_h
    dy = -half_length * sin_h - half_width * cos_h
    corners.append((x + dx, y + dy))
    
    # 后左
    dx = -half_length * cos_h - half_width * sin_h
    dy = -half_length * sin_h + half_width * cos_h
    corners.append((x + dx, y + dy))
    
    return corners

def smooth_acceleration(features, window_length=5, polyorder=2):
    """
    使用Savitzky-Golay滤波器平滑加速度和加加速度数据
    """
    if len(features) < window_length:
        return  # 数据点不足，无法平滑
    
    # 提取加速度数据
    acc_values = [f['acceleration'] for f in features]
    jerk_values = [f['jerk'] for f in features]
    
    # 应用Savitzky-Golay滤波
    if len(acc_values) > window_length:
        smoothed_acc = savgol_filter(acc_values, window_length, polyorder)
        for i, f in enumerate(features):
            f['acceleration'] = smoothed_acc[i]
    
    if len(jerk_values) > window_length:
        smoothed_jerk = savgol_filter(jerk_values, window_length, polyorder)
        for i, f in enumerate(features):
            f['jerk'] = smoothed_jerk[i]

def set_research_plot_style():
    """设置符合科研论文标准的绘图风格"""
    plt.style.use('default')  # 重置为默认样式
    
    # 设置Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体设置
    
    # 设置图形大小和分辨率
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['figure.dpi'] = 300
    
    # 设置线条宽度和样式
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.linewidth'] = 1.0
    
    # 设置刻度线方向和字体大小
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # 设置轴标签和标题字体大小
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    
    # 启用网格
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'



def extract_features(visualize=False, save_path=None):
    """
    提取特征并可选进行可视化分析
    
    :param trajectories: 车辆轨迹数据
    :param visualize: 是否可视化特征
    :param save_path: 可视化图表保存路径
    :return: 提取的特征
    """
    # 检查必要的数据是否存在
    trajectories = example_tra()
    if 'vehicles' not in trajectories:
        raise ValueError("轨迹数据缺少'vehicles'字段")
    
    if 'FV' not in trajectories['vehicles'] or trajectories['vehicles']['FV'] is None:
        raise ValueError("轨迹数据缺少变道车辆FV")
    
    # 提取特征
    feature_data = get_LM_feature(trajectories)
    
    
    return feature_data

def example_tra():
    data_path = r"data_process\behaviour\IRL_value\IRL_tra\tra_data\merging_vehicles_trajectories.pkl"
    # 加载数据
    with open(data_path, 'rb') as f:
        merging_trajectories = pickle.load(f)
        tras = merging_trajectories['vehicle_tracks_000'][0][0]
    return tras


# 使用示例
if __name__ == "__main__":
   
    # 提取特征并可视化
    feature_data = extract_features(visualize=True)
    
    # 输出特征格式示例
    if feature_data:
        print("\n特征格式示例:")
        print("特征列表的第一个元素:")
        for key, value in feature_data['features'][0].items():
            print(f"{key}: {value}")
        
        print("\nFV轨迹示例 (前3个点):")
        for i, point in enumerate(feature_data['fv_trajectory'][:3]):
            print(f"点 {i+1}: x={point['x']}, y={point['y']}, timestamp={point['timestamp']}")