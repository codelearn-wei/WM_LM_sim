# 场景数据结构注释
# scenes_data                             # 字典，包含多个场景的车辆轨迹数据
# ├── 'vehicle_tracks_000'                # 列表，场景000的所有帧数据
# │   ├── ['0']                             # 列表，第0帧数据
# │   ├── ['1']                             # 列表，第1帧数据，包含多个车辆信息
# │   │    └── ['0']                          # 表示第一辆车
    # │   │      └── {                           
    # │   │           'track_id': 8,          # 整数，车辆唯一标识符
    # │   │           'timestamp': 200,       # 整数，时间戳
    # │   │           'x': 1095.497,          # 浮点数，x坐标位置(米)
    # │   │           'y': 962.418,           # 浮点数，y坐标位置(米)
    # │   │           'vx': -5.022,           # 浮点数，x方向速度(米/秒)
    # │   │           'vy': -1.029,           # 浮点数，y方向速度(米/秒)
    # │   │           'psi_rad': -2.939,      # 浮点数，航向角(弧度)
    # │   │           'length': 4.74,         # 浮点数，车辆长度(米)
    # │   │           'width': 1.78,          # 浮点数，车辆宽度(米)
    # │   │           'lane_type': '变道车辆', # 字符串，车道类型('主道车辆'/'变道车辆')
    # │   │           'is_main_vehicle': True # 布尔值，是否为主车辆
    # │   │             },
# │   │       ['1']                          # 表示第二辆车 
# │   │       # ... 更多车辆
# │   │   ]
# │   ├── ['2']                             # 第2帧数据
# │   │   └── [...]
# │   └── ...                             # 更多帧
# ├── 'vehicle_tracks_001'                # 场景001的数据
# │   └── ...
# └── ...                                 # 更多场景


## 场景特征数据结构注释

# scene_statistics = {
#     "scene_id_1": {
#         "frame_count": int,                      # 场景中的帧数
#         "total_unique_vehicles": int,            # 场景中唯一车辆的总数
#         "main_vehicle_count": int,               # 主车数量
#         "lane_change_vehicle_count": int,        # 变道车辆数量
#         "main_lane_vehicle_count": int,          # 主道车辆数量
#         "lane_change_ratio": float,              # 变道车辆比例
        
#         # 速度统计
#         "avg_speed": float,                      # 平均速度(米/秒)
#         "max_speed": float,                      # 最大速度(米/秒)
#         "min_speed": float,                      # 最小速度(米/秒)
#         "speed_std": float,                      # 速度标准差
        
#         # 车辆尺寸
#         "avg_vehicle_size": float,               # 平均车辆尺寸(面积=长×宽)
        
#         # 原始数据
#         "speed_distribution": [float, ...],      # 所有车辆在所有帧的速度列表
#         "position_data": [(float, float), ...]   # 所有车辆在所有帧的位置坐标列表(x,y)
#     },
#     "scene_id_2": {
#         # 与上面结构相同的数据
#     },
#     # 更多场景...
# }

from get_all_scene_data import load_scenes_data 
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
# Add Time New Roman font
font_path = font_manager.findfont('Times New Roman')
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'Times New Roman'


######文件信息统计#######

def calculate_file_statistics(scenes_data):
    """计算每个场景的统计特征并将结果存储回场景数据中"""
    
    file_statistics = {}
    
    for scene_id, frames in scenes_data.items():
        # 初始化该场景的统计数据
        stats = {
            'frame_count': len(frames),
            'unique_vehicles': set(),
            'main_vehicles': set(),
            'lane_change_vehicles': set(),
            'main_lane_vehicles': set(),
            'speeds': [],
            'vehicle_sizes': [],
            'positions': [],
            'main_lane_speeds': [],  # 新增主道车辆速度列表
            'lane_change_speeds': []  # 新增辅道车辆速度列表
        }
        
        # 遍历每一帧
        for frame_idx, frame in frames.items():
            for vehicles in frame:
                for vehicle_data in vehicles:
                             
                    # 记录唯一车辆ID
                    track_id = vehicle_data['track_id']
                    stats['unique_vehicles'].add(track_id)
                    
                    # 区分主车和其他车辆
                    if vehicle_data['is_main_vehicle']:
                        stats['main_vehicles'].add(track_id)
                    
                    # 计算速度(米/秒)
                    speed = sqrt(vehicle_data['vx']**2 + vehicle_data['vy']**2)
                    stats['speeds'].append(speed)
                    
                    # 区分变道车辆和主道车辆及其速度
                    if vehicle_data['lane_type'] == '变道车辆':
                        stats['lane_change_vehicles'].add(track_id)
                        stats['lane_change_speeds'].append(speed)
                    else:  # '主道车辆'
                        stats['main_lane_vehicles'].add(track_id)
                        stats['main_lane_speeds'].append(speed)
                    
                    # 记录车辆尺寸
                    vehicle_size = vehicle_data['length'] * vehicle_data['width']
                    stats['vehicle_sizes'].append(vehicle_size)
                    
                    # 记录位置信息(用于后续密度计算)
                    stats['positions'].append((vehicle_data['x'], vehicle_data['y']))
        
        # 计算汇总统计信息
        scene_stats = {
            'frame_count': stats['frame_count'],
            'total_unique_vehicles': len(stats['unique_vehicles']),
            'main_vehicle_count': len(stats['main_vehicles']),
            'lane_change_vehicle_count': len(stats['lane_change_vehicles']),
            'main_lane_vehicle_count': len(stats['main_lane_vehicles']),
            'lane_change_ratio': len(stats['lane_change_vehicles']) / max(1, len(stats['unique_vehicles'])),
            
            # 速度统计
            'avg_speed': np.mean(stats['speeds']) if stats['speeds'] else 0,
            'max_speed': max(stats['speeds']) if stats['speeds'] else 0,
            'min_speed': min(stats['speeds']) if stats['speeds'] else 0,
            'speed_std': np.std(stats['speeds']) if stats['speeds'] else 0,
            
            # 主道车辆速度统计
            'main_lane_avg_speed': np.mean(stats['main_lane_speeds']) if stats['main_lane_speeds'] else 0,
            'main_lane_speed_std': np.std(stats['main_lane_speeds']) if stats['main_lane_speeds'] else 0,
            
            # 辅道车辆速度统计
            'lane_change_avg_speed': np.mean(stats['lane_change_speeds']) if stats['lane_change_speeds'] else 0,
            'lane_change_speed_std': np.std(stats['lane_change_speeds']) if stats['lane_change_speeds'] else 0,
            
            # 车辆尺寸统计
            'avg_vehicle_size': np.mean(stats['vehicle_sizes']) if stats['vehicle_sizes'] else 0,
            
            # 保存原始数据用于分布图
            'speed_distribution': stats['speeds'],
            'position_data': stats['positions'],
            'main_lane_speed_distribution': stats['main_lane_speeds'],
            'lane_change_speed_distribution': stats['lane_change_speeds']
        }
        
        # 存储计算的统计信息
        file_statistics[scene_id] = scene_stats
    
    return file_statistics

######场景信息绘图分析#######

def calculate_scene_statistics(scenes_data):
    """计算每个场景的统计特征并将结果存储回场景数据中"""
    
    scene_statistics = {}
    
    # 假设scenes_data的结构是{file_id: {scene_id: frames}}
    for file_id, file_scenes in scenes_data.items():
        # 每个文件下有多个场景
        for scene_id, frames in file_scenes.items():
            # 初始化该场景的统计数据
            stats = {
                'frame_count': len(frames),
                'unique_vehicles': set(),
                'main_vehicles': set(),
                'lane_change_vehicles': set(),
                'main_lane_vehicles': set(),
                'speeds': [],
                'vehicle_sizes': [],
                'positions': [],
                'main_lane_speeds': [],  # 主道车辆速度列表
                'lane_change_speeds': []  # 辅道车辆速度列表
            }
            
            # 遍历每一帧
  
            for vehicles in frames:
                for vehicle_data in vehicles:
                                 
                        # 记录唯一车辆ID
                        track_id = vehicle_data['track_id']
                        stats['unique_vehicles'].add(track_id)
                        
                        # 区分主车和其他车辆
                        if vehicle_data['is_main_vehicle']:
                            stats['main_vehicles'].add(track_id)
                        
                        # 计算速度(米/秒)
                        speed = sqrt(vehicle_data['vx']**2 + vehicle_data['vy']**2)
                        stats['speeds'].append(speed)
                        
                        # 区分变道车辆和主道车辆及其速度
                        if vehicle_data['lane_type'] == '变道车辆':
                            stats['lane_change_vehicles'].add(track_id)
                            stats['lane_change_speeds'].append(speed)
                        else:  # '主道车辆'
                            stats['main_lane_vehicles'].add(track_id)
                            stats['main_lane_speeds'].append(speed)
                        
                        # 记录车辆尺寸
                        vehicle_size = vehicle_data['length'] * vehicle_data['width']
                        stats['vehicle_sizes'].append(vehicle_size)
                        
                        # 记录位置信息(用于后续密度计算)
                        stats['positions'].append((vehicle_data['x'], vehicle_data['y']))
            
            # 计算汇总统计信息
            scene_stats = {
                'file_id': file_id,
                'scene_id': scene_id,
                'frame_count': stats['frame_count'],
                'total_unique_vehicles': len(stats['unique_vehicles']),
                'main_vehicle_count': len(stats['main_vehicles']),
                'lane_change_vehicle_count': len(stats['lane_change_vehicles']),
                'main_lane_vehicle_count': len(stats['main_lane_vehicles']),
                'lane_change_ratio': len(stats['lane_change_vehicles']) / max(1, len(stats['unique_vehicles'])),
                
                # 速度统计
                'avg_speed': np.mean(stats['speeds']) if stats['speeds'] else 0,
                'max_speed': max(stats['speeds']) if stats['speeds'] else 0,
                'min_speed': min(stats['speeds']) if stats['speeds'] else 0,
                'speed_std': np.std(stats['speeds']) if stats['speeds'] else 0,
                
                # 主道车辆速度统计
                'main_lane_avg_speed': np.mean(stats['main_lane_speeds']) if stats['main_lane_speeds'] else 0,
                'main_lane_speed_std': np.std(stats['main_lane_speeds']) if stats['main_lane_speeds'] else 0,
                
                # 辅道车辆速度统计
                'lane_change_avg_speed': np.mean(stats['lane_change_speeds']) if stats['lane_change_speeds'] else 0,
                'lane_change_speed_std': np.std(stats['lane_change_speeds']) if stats['lane_change_speeds'] else 0,
                
                # 车辆尺寸统计
                'avg_vehicle_size': np.mean(stats['vehicle_sizes']) if stats['vehicle_sizes'] else 0,
                
                # 保存原始数据用于分布图
                'speed_distribution': stats['speeds'],
                'position_data': stats['positions'],
                'main_lane_speed_distribution': stats['main_lane_speeds'],
                'lane_change_speed_distribution': stats['lane_change_speeds']
            }
            
            # 存储计算的统计信息
            scene_key = f"{file_id}_{scene_id}"  # 创建唯一的场景标识符
            scene_statistics[scene_key] = scene_stats
    
    return scene_statistics

## 绘制图形，不同文件的对比情况
def plot_speed_statistics(file_statistics, save_path=None):
    """
    Plot average speeds with improved error representations
    
    Parameters:
        file_statistics (dict): Dictionary containing statistics for each scene
        save_path (str, optional): Path to save the figure, if not provided the figure is displayed
    
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import re
    
    # Set Times New Roman font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    # Extract and sort file_ids based on numeric order (000-010)
    file_ids = list(file_statistics.keys())
    
    # Sort the file_ids based on numeric pattern
    def extract_number(file_id):
        match = re.search(r'(\d+)', file_id)
        return int(match.group(1)) if match else 0
    
    file_ids.sort(key=extract_number)
    
    # Extract data in sorted order
    avg_speeds = [file_statistics[file_id]['avg_speed'] for file_id in file_ids]
    speed_stds = [file_statistics[file_id]['speed_std'] for file_id in file_ids]
    
    main_lane_avg_speeds = [file_statistics[file_id]['main_lane_avg_speed'] for file_id in file_ids]
    main_lane_speed_stds = [file_statistics[file_id]['main_lane_speed_std'] for file_id in file_ids]
    
    lane_change_avg_speeds = [file_statistics[file_id]['lane_change_avg_speed'] for file_id in file_ids]
    lane_change_speed_stds = [file_statistics[file_id]['lane_change_speed_std'] for file_id in file_ids]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # X-axis positions
    x = np.arange(len(file_ids))
    
    # Plot lines for average speeds
    line1, = ax.plot(x - 0.2, main_lane_avg_speeds, 'o-', linewidth=2, markersize=6, 
                     label='Main Lane', color='#3274A1')
    
    line2, = ax.plot(x, avg_speeds, 's-', linewidth=2, markersize=6, 
                     label='All Vehicles', color='#E1812C')
    
    line3, = ax.plot(x + 0.2, lane_change_avg_speeds, '^-', linewidth=2, markersize=6, 
                     label='Lane Change', color='#3A923A')
    
    # Add subtle shaded areas for standard deviations
    ax.fill_between(x - 0.2, 
                   [m - s for m, s in zip(main_lane_avg_speeds, main_lane_speed_stds)],
                   [m + s for m, s in zip(main_lane_avg_speeds, main_lane_speed_stds)],
                   alpha=0.2, color='#3274A1')
    
    ax.fill_between(x, 
                   [m - s for m, s in zip(avg_speeds, speed_stds)],
                   [m + s for m, s in zip(avg_speeds, speed_stds)],
                   alpha=0.2, color='#E1812C')
    
    ax.fill_between(x + 0.2, 
                   [m - s for m, s in zip(lane_change_avg_speeds, lane_change_speed_stds)],
                   [m + s for m, s in zip(lane_change_avg_speeds, lane_change_speed_stds)],
                   alpha=0.2, color='#3A923A')
    
    # Configure axis
    ax.set_xlabel('Scene ID', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average Speed (m/s)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(file_ids, rotation=45 if len(file_ids) > 5 else 0)
    ax.tick_params(axis='both', labelsize=10)
    
    # Remove grid
    ax.grid(False)
    
    # Add legend inside the plot with larger font and no frame
    ax.legend(loc='upper right', fontsize=12, frameon=False)
    
    # Set title
    plt.title('Speed Statistics Across Scenes', fontsize=14, fontweight='bold')
    
    # Add data point annotations
    for i, (avg, std) in enumerate(zip(avg_speeds, speed_stds)):
        ax.annotate(f'{avg:.1f}±{std:.1f}', xy=(i, avg), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display figure
    plt.savefig(f"{save_path}/speed_statistics.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_path}/speed_statistics.pdf", bbox_inches='tight',  dpi=300)


### 绘制场景数据分布
def plot_vehicle_count_distribution(scene_statistics, save_path=None):
    """
    Plot the distribution of main lane vehicles and lane change vehicles across scenes.
    Vehicle counts are grouped in intervals of 3.
    
    Parameters:
        scene_statistics (dict): Dictionary containing statistics for each scene
        save_path (str, optional): Path to save the figure, if not provided the figure is displayed
    
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter
    
    # Set Times New Roman font and improve aesthetics
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.edgecolor'] = '#333333'
    
    # Extract scene keys and relevant data
    scene_keys = list(scene_statistics.keys())
    
    # Count vehicles in each scene
    main_vehicle_counts = [scene_statistics[key]['main_lane_vehicle_count'] 
                          for key in scene_keys if 'main_lane_vehicle_count' in scene_statistics[key]]
    lane_change_counts = [scene_statistics[key]['lane_change_vehicle_count'] 
                         for key in scene_keys if 'lane_change_vehicle_count' in scene_statistics[key]]
    
    # Create bins for vehicle counts with intervals of 3 instead of 5
    max_count = max(max(main_vehicle_counts, default=0), max(lane_change_counts, default=0))
    bins = list(range(0, max_count + 4, 3))  # Bins in steps of 3
    
    # Count distribution of scenes by vehicle count
    main_dist = Counter()
    lane_change_dist = Counter()
    
    for count in main_vehicle_counts:
        bin_idx = count // 3
        main_dist[bin_idx * 3] += 1
        
    for count in lane_change_counts:
        bin_idx = count // 3
        lane_change_dist[bin_idx * 3] += 1
    
    # Create figure with improved size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting
    x_bins = np.array([b for b in bins if b in main_dist or b in lane_change_dist])
    x_bins.sort()
    
    width = 0.35  # Wider bars
    
    main_values = [main_dist[b] for b in x_bins]
    lane_change_values = [lane_change_dist[b] for b in x_bins]
    
    x = np.arange(len(x_bins))
    
    # Plot bars with enhanced colors and styles
    ax.bar(x - width/2, main_values, width, label='Main Lane Vehicles', 
          color='#3274A1', edgecolor='black', linewidth=1, alpha=0.85)
    ax.bar(x + width/2, lane_change_values, width, label='Lane Change Vehicles', 
          color='#3A923A', edgecolor='black', linewidth=1, alpha=0.85)
    
    # Enhance plot appearance
    ax.set_xlabel('Number of Vehicles', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Scenes', fontsize=16, fontweight='bold')
    ax.set_title('Vehicle Count Distribution Across Scenes', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}" for b in x_bins], fontsize=14)
    ax.tick_params(axis='both', labelsize=14, width=1.5, length=6)
    
    # Add value labels on top of bars with increased font size
    for i, v in enumerate(main_values):
        if v > 0:
            ax.text(i - width/2, v + 0.5, str(v), ha='center', fontsize=14, fontweight='bold')
    for i, v in enumerate(lane_change_values):
        if v > 0:
            ax.text(i + width/2, v + 0.5, str(v), ha='center', fontsize=14, fontweight='bold')
    
    # Enhance legend - removed frame as requested
    ax.legend(fontsize=16, frameon=False, loc='upper right')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Calculate and display statistics
    avg_main = sum(main_vehicle_counts) / len(main_vehicle_counts) if main_vehicle_counts else 0
    avg_lane_change = sum(lane_change_counts) / len(lane_change_counts) if lane_change_counts else 0
    
    stats_text = f"Average Main Lane Vehicles: {avg_main:.2f}\nAverage Lane Change Vehicles: {avg_lane_change:.2f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=14,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Improve overall appearance
    plt.tight_layout()
    
    # Add subtle background color
    fig.patch.set_facecolor('#f9f9f9')
    ax.set_facecolor('#f5f5f5')
    
    # Save or display figure
    plt.savefig(f"{save_path}/vehicle_count.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_path}/vehicle_count.pdf", bbox_inches='tight',  dpi=300)


def plot_scene_speed_distribution(scene_statistics, save_path=None):
    """
    Plot the average speed distribution across scenes.
    Speed values above 6 m/s are grouped together.
    
    Parameters:
        scene_statistics (dict): Dictionary containing statistics for each scene
        save_path (str, optional): Path to save the figure, if not provided the figure is displayed
    
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter
    
    # Set Times New Roman font and improve aesthetics
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.edgecolor'] = '#333333'
    
    # Extract average speeds from each scene
    main_lane_avg_speeds = []
    lane_change_avg_speeds = []
    
    for scene_key, stats in scene_statistics.items():
        # Get average speed for main lane vehicles
        if ('main_lane_speed_distribution' in stats and 
            stats['main_lane_speed_distribution'] and 
            len(stats['main_lane_speed_distribution']) > 0):
            main_lane_avg_speeds.append(np.mean(stats['main_lane_speed_distribution']))
        
        # Get average speed for lane change vehicles
        if ('lane_change_speed_distribution' in stats and 
            stats['lane_change_speed_distribution'] and 
            len(stats['lane_change_speed_distribution']) > 0):
            lane_change_avg_speeds.append(np.mean(stats['lane_change_speed_distribution']))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define speed bins from 0 to 6 m/s with 1 m/s intervals, and group speeds > 6 together
    speed_bins = np.linspace(0, 6, 7)
    
    # Count scenes in each speed bin
    main_lane_speed_hist = Counter()
    lane_change_speed_hist = Counter()
    
    for speed in main_lane_avg_speeds:
        if speed <= 6:
            bin_idx = int(speed)
            main_lane_speed_hist[bin_idx] += 1
        else:
            # Group speeds > 6 into the 6 bin
            main_lane_speed_hist[6] += 1
            
    for speed in lane_change_avg_speeds:
        if speed <= 6:
            bin_idx = int(speed)
            lane_change_speed_hist[bin_idx] += 1
        else:
            # Group speeds > 6 into the 6 bin
            lane_change_speed_hist[6] += 1
    
    # Create x-axis labels
    x_labels = [f"{i}" for i in range(6)] + ["≥6"]
    x = np.arange(len(x_labels))
    
    # Prepare data for plotting
    width = 0.35
    main_values = [main_lane_speed_hist[i] for i in range(7)]
    lane_change_values = [lane_change_speed_hist[i] for i in range(7)]
    
    # Plot histogram bars with enhanced colors
    ax.bar(x - width/2, main_values, width, label='Main Lane Vehicles', 
          color='#3274A1', edgecolor='black', linewidth=1, alpha=0.85)
    ax.bar(x + width/2, lane_change_values, width, label='Lane Change Vehicles', 
          color='#3A923A', edgecolor='black', linewidth=1, alpha=0.85)
    
    # Enhance plot appearance
    ax.set_xlabel('Average Speed (m/s)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Scenes', fontsize=16, fontweight='bold')
    ax.set_title('Distribution of Average Vehicle Speed Across Scenes', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=14)
    ax.tick_params(axis='both', labelsize=14, width=1.5, length=6)
    
    # Add value labels on top of bars
    for i, v in enumerate(main_values):
        if v > 0:
            ax.text(i - width/2, v + 0.3, str(v), ha='center', fontsize=14, fontweight='bold')
    for i, v in enumerate(lane_change_values):
        if v > 0:
            ax.text(i + width/2, v + 0.3, str(v), ha='center', fontsize=14, fontweight='bold')
    
    # Enhanced legend - removed frame as requested
    ax.legend(fontsize=16, frameon=False, loc='upper right')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Calculate and display overall statistics
    if main_lane_avg_speeds:
        main_mean = np.mean(main_lane_avg_speeds)
        main_median = np.median(main_lane_avg_speeds)
    else:
        main_mean = main_median = 0
        
    if lane_change_avg_speeds:
        lane_mean = np.mean(lane_change_avg_speeds)
        lane_median = np.median(lane_change_avg_speeds)
    else:
        lane_mean = lane_median = 0
    
    stats_text = (f"Main Lane - Mean: {main_mean:.2f} m/s, Median: {main_median:.2f} m/s\n"
                 f"Lane Change - Mean: {lane_mean:.2f} m/s, Median: {lane_median:.2f} m/s")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=14,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Improve overall appearance
    plt.tight_layout()
    
    # Add subtle background color
    fig.patch.set_facecolor('#f9f9f9')
    ax.set_facecolor('#f5f5f5')
    
    plt.savefig(f"{save_path}/vehicle_scene_speed.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_path}/vehicle_scene_speed.pdf", bbox_inches='tight',  dpi=300)

    
if __name__ == "__main__":
    
    data_path = r"data_process\behaviour\data\all_scenes_data.pkl"
    result_path = r"data_process\behaviour\results\scene"
    ## 划分出案例数据
    scenes_data = load_scenes_data(data_path)
    file_statistics = calculate_file_statistics(scenes_data)
    
    scene_statistics = calculate_scene_statistics(scenes_data)
    # print(scene_statistics)
    
    # 绘制几个文件的对比情况
    plot_speed_statistics(file_statistics, result_path)
    plot_vehicle_count_distribution(scene_statistics, result_path)
    plot_scene_speed_distribution(scene_statistics, result_path)
    

    
    
    
    
    
    

