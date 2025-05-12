
# !基于数据集特征的分布分析

from get_all_tra_data import load_vehicle_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import colorsys
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 计算车辆的速度
def compute_vehicle_speeds(vehicle):
    vx = np.array(vehicle.vx_values)
    vy = np.array(vehicle.vy_values)
    speeds = np.sqrt(vx**2 + vy**2)
    return speeds

def extract_all_speeds(track_dict):
    """提取某个车道所有车辆的速度"""
    speeds_all = []
    for vehicles in track_dict.values():
      for vehicle in vehicles.values():
        speeds = compute_vehicle_speeds(vehicle)
        speeds_all.extend(speeds)
    return np.array(speeds_all)


########## 速度分布图 ##########
def plot_improved_speed_distribution(main_dict, merge_dict, save_dir=None):
    """绘制主道与辅道车辆速度分布图（简洁科研风格）"""
    # 提取速度数据
    main_speeds = extract_all_speeds(main_dict)
    merge_speeds = extract_all_speeds(merge_dict)
    
    # 过滤速度数据，仅保留0-12 m/s范围内的数据
    main_speeds = main_speeds[(main_speeds >= 0) & (main_speeds <= 12)]
    merge_speeds = merge_speeds[(merge_speeds >= 0) & (merge_speeds <= 12)]
    
    # 设置统一的绘图样式 - 简洁科研风格
    sns.set_style("ticks")
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,  # 增大图例字体
    })
    
    # 主道车辆速度分布图
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    # 直方图和KDE组合图
    sns.histplot(main_speeds, kde=False, color='#1f77b4', alpha=0.6, 
                 bins=30, stat='density', ax=ax1, label='Histogram')
    sns.kdeplot(main_speeds, color='#d62728', linewidth=2, ax=ax1, label='KDE')
    
    ax1.set_xlabel('Speed (m/s)')
    ax1.set_ylabel('Density')
    ax1.set_title('Speed Distribution - Main Road Vehicles')
    ax1.legend(frameon=False)  # 移除图例边框
    ax1.set_xlim(0, 12)  # 设置x轴范围从0到12
    sns.despine()  # 移除上边框和右边框
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f'{save_dir}/main_road_speed_distribution.png', dpi=300, bbox_inches='tight')
    
    # 辅道车辆速度分布图
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # 直方图和KDE组合图
    sns.histplot(merge_speeds, kde=False, color='#ff7f0e', alpha=0.6, 
                 bins=30, stat='density', ax=ax2, label='Histogram')
    sns.kdeplot(merge_speeds, color='#d62728', linewidth=2, ax=ax2, label='KDE')
    
    ax2.set_xlabel('Speed (m/s)')
    ax2.set_ylabel('Density')
    ax2.set_title('Speed Distribution - Merging Road Vehicles')
    ax2.legend(frameon=False)  # 移除图例边框
    ax2.set_xlim(0, 12)  # 设置x轴范围从0到12
    sns.despine()  # 移除上边框和右边框
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f'{save_dir}/speed_distrub/merge_road_speed_distribution.png', dpi=300, bbox_inches='tight')
    
    # 主辅道对比图（可选）
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    
    # 使用不同的颜色和半透明效果
    sns.kdeplot(main_speeds, color='#1f77b4', linewidth=2.5, ax=ax3, label='Main Road')
    sns.kdeplot(merge_speeds, color='#ff7f0e', linewidth=2.5, ax=ax3, label='Merging Road')
    
    # 添加垂直线表示均值
    main_mean = np.mean(main_speeds)
    merge_mean = np.mean(merge_speeds)
    ax3.axvline(main_mean, color='#1f77b4', linestyle='--', alpha=0.7)
    ax3.axvline(merge_mean, color='#ff7f0e', linestyle='--', alpha=0.7)
    
    # 添加均值标注
    ax3.text(main_mean + 0.2, ax3.get_ylim()[1]*0.9, f'Mean: {main_mean:.2f}', 
             color='#1f77b4', fontsize=10, ha='left')
    ax3.text(merge_mean + 0.2, ax3.get_ylim()[1]*0.8, f'Mean: {merge_mean:.2f}', 
             color='#ff7f0e', fontsize=10, ha='left')
    
    ax3.set_xlabel('Speed (m/s)')
    ax3.set_ylabel('Density')
    ax3.set_title('Speed Distribution Comparison')
    ax3.legend(title='Vehicle Type', frameon=False)  # 移除图例边框
    ax3.set_xlim(0, 12)  # 设置x轴范围从0到12
    sns.despine()
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f'{save_dir}/speed_distrub/speed_distribution_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.show()


########## 绘制所有轨迹 ##########
def plot_vehicle_trajectories(main_dict, merge_dict, save_path=None):

    # 设置科研绘图风格
    plt.figure(figsize=(12, 4))
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    ax = plt.gca()
    
    # 设置背景色为浅灰色
    ax.set_facecolor('#f8f8f8')
    
    # 生成充足的不同颜色
    import colorsys
    
    def get_distinct_colors(n):
        """生成n个视觉上区分度高的颜色"""
        colors = []
        for i in range(n):
            # 使用HSV色彩空间以获得更好的颜色区分
            hue = i / n
            saturation = 0.7 + 0.3 * (i % 2)  # 在0.7-1.0之间交替
            brightness = 0.6 + 0.2 * (i % 3) / 2  # 在0.6-0.8之间微调
            colors.append(colorsys.hsv_to_rgb(hue, saturation, brightness))
        return colors
    
    # 统计需要的颜色数量
    total_vehicles = 0
    for track_dict in [main_dict, merge_dict]:
        for vehicle_dict in track_dict.values():
            total_vehicles += len(vehicle_dict)
    
    # 获取颜色列表
    colors = get_distinct_colors(total_vehicles)
    color_index = 0
    
    # 绘制主道车辆轨迹
    for track_id, vehicle_dict in main_dict.items():
        for vehicle_id, vehicle in vehicle_dict.items():
            xs = vehicle.x_coords
            ys = vehicle.y_coords
            
            # 使用平滑曲线连接点
            plt.plot(xs, ys, '-', linewidth=1.2, alpha=0.8, 
                     color=colors[color_index], zorder=2)
            
            
            color_index += 1
    
    # 绘制匝道车辆轨迹
    for track_id, vehicle_dict in merge_dict.items():
        for vehicle_id, vehicle in vehicle_dict.items():
            xs = vehicle.x_coords
            ys = vehicle.y_coords
            
            # 使用平滑曲线连接点
            plt.plot(xs, ys, '-', linewidth=1.2, alpha=0.8, 
                     color=colors[color_index], zorder=2)
            
            # 可选：标记起点和终点
            # plt.scatter(xs[0], ys[0], s=15, color=colors[color_index], zorder=3)
            # plt.scatter(xs[-1], ys[-1], s=15, color=colors[color_index], zorder=3)
            
            color_index += 1
    
    # 设置坐标轴标签和标题
    plt.xlabel('X Coordinate (m)', fontweight='bold')
    plt.ylabel('Y Coordinate (m)', fontweight='bold')
    plt.title('Vehicle Trajectories', fontweight='bold')
    
    # 设置坐标轴范围（可根据实际数据调整）
    # plt.xlim(min_x, max_x)
    # plt.ylim(min_y, max_y)
    
    # 移除顶部和右侧边框
    sns.despine()
    plt.axis('equal')
    
    # 添加网格线（可选，如果需要可取消注释）
    # plt.grid(False)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
def get_trajectory_bounds(main_dict, merge_dict):
    """获取所有轨迹的坐标范围，用于设置绘图范围"""
    all_x = []
    all_y = []
    
    # 收集所有坐标
    for track_dict in [main_dict, merge_dict]:
        for vehicle_dict in track_dict.values():
            for vehicle in vehicle_dict.values():
                all_x.extend(vehicle.x_coords)
                all_y.extend(vehicle.y_coords)
    
    # 获取范围
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # 添加小边距
    margin_x = (max_x - min_x) * 0.05
    margin_y = (max_y - min_y) * 0.05
    
    return {
        'min_x': min_x - margin_x,
        'max_x': max_x + margin_x,
        'min_y': min_y - margin_y,
        'max_y': max_y + margin_y
    }


def plot_spacetime_heatmap(main_dict, merge_dict, save_path=None):
    """
    绘制车辆汇流和主道车辆的时空热力图
    
    参数:
    main_dict: 主道车辆数据字典
    merge_dict: 汇流车辆数据字典
    save_path: 图像保存路径
    """
    # 设置科研绘图风格
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    # 准备主道和汇流车辆的位置和时间数据
    main_x_all = []
    main_t_all = []
    merge_x_all = []
    merge_t_all = []
    
    # 提取主道车辆数据
    for track_id, vehicle_dict in main_dict.items():
        for vehicle_id, vehicle in vehicle_dict.items():
            # 使用x坐标表示位置(沿道路方向)
            x_coords = vehicle.x_coords
            # 时间数据
            timestamps = [t / 1000 for t in vehicle.timestamps]
            
            # 确保数据包含相同数量的点
            min_len = min(len(x_coords), len(timestamps))
            
            main_x_all.extend(x_coords[:min_len])
            main_t_all.extend(timestamps[:min_len])  # 转换为秒
    
    # 提取汇流车辆数据
    for track_id, vehicle_dict in merge_dict.items():
        for vehicle_id, vehicle in vehicle_dict.items():
            x_coords = vehicle.x_coords
            timestamps = vehicle.timestamps
            
            min_len = min(len(x_coords), len(timestamps))
            
            merge_x_all.extend(x_coords[:min_len])
            merge_t_all.extend(timestamps[:min_len])
    
    # 创建子图布局 (1行2列)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # 主道车辆热力图
    h1 = ax1.hist2d(main_x_all, main_t_all, bins=(50, 50), cmap='Blues', 
                   alpha=0.8, density=True)
    ax1.set_xlabel('Position (m)', fontweight='bold')
    ax1.set_ylabel('Time (s)', fontweight='bold')
    ax1.set_title('Main Road Vehicles', fontweight='bold')
    fig.colorbar(h1[3], ax=ax1, label='Density')
    
    # 汇流车辆热力图
    h2 = ax2.hist2d(merge_x_all, merge_t_all, bins=(50, 50), cmap='Oranges', 
                   alpha=0.8, density=True)
    ax2.set_xlabel('Position (m)', fontweight='bold')
    ax2.set_title('Merging Vehicles', fontweight='bold')
    fig.colorbar(h2[3], ax=ax2, label='Density')
    
    # 移除顶部和右侧边框
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 创建组合热力图（单图显示两种车辆）
    plt.figure(figsize=(12, 8))
    
    # 设置轴范围
    x_min = min(min(main_x_all), min(merge_x_all))
    x_max = max(max(main_x_all), max(merge_x_all))
    t_min = min(min(main_t_all), min(merge_t_all))
    t_max = max(max(main_t_all), max(merge_t_all))
    
    # 使用contourf创建平滑的热力图
    plt.hexbin(main_x_all, main_t_all, gridsize=50, cmap='Blues', 
              alpha=0.7, label='Main Road')
    plt.hexbin(merge_x_all, merge_t_all, gridsize=50, cmap='Oranges', 
              alpha=0.7, label='Merging Road')
    
    plt.xlabel('Position (m)', fontweight='bold')
    plt.ylabel('Time (s)', fontweight='bold')
    plt.title('Space-Time Heatmap of Vehicle Trajectories', fontweight='bold')
    
    # 自定义图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Main Road'),
        Patch(facecolor='orange', alpha=0.7, label='Merging Road')
    ]
    plt.legend(handles=legend_elements, frameon=False, loc='upper right')
    
    # 移除顶部和右侧边框
    sns.despine()
    
    plt.tight_layout()
    
    # 保存组合热力图
    if save_path:
        combined_path = save_path.replace('.png', '_combined.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    
    plt.show()
import os
    
def plot_file_spacetime_heatmaps(main_dict, merge_dict, file_id, save_dir=None):
    """
    为特定文件ID绘制车辆的时空热力图
    
    参数:
    main_dict: 主道车辆数据字典
    merge_dict: 汇流车辆数据字典
    file_id: 文件ID (例如 'vehicle_tracks_000')
    save_dir: 保存目录
    """
    # 确保目录存在
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 设置科研绘图风格
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    # 提取指定文件的数据
    main_vehicles = main_dict.get(file_id, {})
    merge_vehicles = merge_dict.get(file_id, {})
    
    if not main_vehicles and not merge_vehicles:
        print(f"No data found for file ID: {file_id}")
        return
    
    # 准备主道和汇流车辆的位置和时间数据
    main_x_all = []
    main_t_all = []
    merge_x_all = []
    merge_t_all = []
    
    # 提取主道车辆数据
    for vehicle_id, vehicle in main_vehicles.items():
        # 使用x坐标表示位置(沿道路方向)
        x_coords = vehicle.x_coords
        # 时间数据
        timestamps = np.array(vehicle.timestamps) / 1000  # 转换为秒
        
        # 确保数据包含相同数量的点
        min_len = min(len(x_coords), len(timestamps))
        
        main_x_all.extend(x_coords[:min_len])
        main_t_all.extend(timestamps[:min_len])
    
    # 提取汇流车辆数据
    for vehicle_id, vehicle in merge_vehicles.items():
        x_coords = vehicle.x_coords
        timestamps = np.array(vehicle.timestamps) / 1000  # 转换为秒
        
        min_len = min(len(x_coords), len(timestamps))
        
        merge_x_all.extend(x_coords[:min_len])
        merge_t_all.extend(timestamps[:min_len])
    
    # 1. 基本热力图 - 并排显示主道和汇流车道
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # 主道车辆热力图
    h1 = ax1.hist2d(main_x_all, main_t_all, bins=(50, 50), cmap='Blues', 
                   alpha=0.8, density=True)
    ax1.set_xlabel('Position (m)', fontweight='bold')
    ax1.set_ylabel('Time (s)', fontweight='bold')
    ax1.set_title(f'Main Road Vehicles - {file_id}', fontweight='bold')
    fig.colorbar(h1[3], ax=ax1, label='Density')
    
    # 汇流车辆热力图
    h2 = ax2.hist2d(merge_x_all, merge_t_all, bins=(50, 50), cmap='Oranges', 
                   alpha=0.8, density=True)
    ax2.set_xlabel('Position (m)', fontweight='bold')
    ax2.set_title(f'Merging Vehicles - {file_id}', fontweight='bold')
    fig.colorbar(h2[3], ax=ax2, label='Density')
    
    # 移除顶部和右侧边框
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    
    plt.tight_layout()
    
    # 保存图像
    if save_dir:
        save_path = os.path.join(save_dir, f"{file_id}_spacetime_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    # 2. 组合热力图（单图显示两种车辆）- 使用Hexbin
    plt.figure(figsize=(12, 8))
    
    # 设置轴范围
    x_min = min(min(main_x_all) if main_x_all else 0, min(merge_x_all) if merge_x_all else 0)
    x_max = max(max(main_x_all) if main_x_all else 0, max(merge_x_all) if merge_x_all else 0)
    t_min = min(min(main_t_all) if main_t_all else 0, min(merge_t_all) if merge_t_all else 0)
    t_max = max(max(main_t_all) if main_t_all else 0, max(merge_t_all) if merge_t_all else 0)
    
    # 使用hexbin创建平滑的热力图
    if main_x_all:
        plt.hexbin(main_x_all, main_t_all, gridsize=50, cmap='Blues', 
                alpha=0.7, label='Main Road')
    if merge_x_all:
        plt.hexbin(merge_x_all, merge_t_all, gridsize=50, cmap='Oranges', 
                alpha=0.7, label='Merging Road')
    
    plt.xlabel('Position (m)', fontweight='bold')
    plt.ylabel('Time (s)', fontweight='bold')
    plt.title(f'Space-Time Heatmap of Vehicle Trajectories - {file_id}', fontweight='bold')
    
    # 自定义图例
    legend_elements = []
    if main_x_all:
        legend_elements.append(Patch(facecolor='blue', alpha=0.7, label='Main Road'))
    if merge_x_all:
        legend_elements.append(Patch(facecolor='orange', alpha=0.7, label='Merging Road'))
    
    plt.legend(handles=legend_elements, frameon=False, loc='upper right')
    
    # 移除顶部和右侧边框
    sns.despine()
    
    plt.tight_layout()
    
    # 保存组合热力图
    if save_dir:
        save_path = os.path.join(save_dir, f"{file_id}_spacetime_heatmap_combined.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    # 3. 时空轨迹图 - 单独绘制每个车辆的轨迹
    plt.figure(figsize=(12, 8))
    
    # 获取不同的颜色
    def get_distinct_colors(n):
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + 0.3 * (i % 2)  
            brightness = 0.6 + 0.2 * (i % 3) / 2
            colors.append(colorsys.hsv_to_rgb(hue, saturation, brightness))
        return colors
    
    # 计算需要的颜色数量
    num_vehicles = len(main_vehicles) + len(merge_vehicles)
    colors = get_distinct_colors(max(num_vehicles, 1))
    color_idx = 0
    
    # 绘制主道车辆轨迹
    for vehicle_id, vehicle in main_vehicles.items():
        x_coords = vehicle.x_coords
        timestamps = np.array(vehicle.timestamps) / 1000  # 转换为秒
        
        min_len = min(len(x_coords), len(timestamps))
        plt.plot(x_coords[:min_len], timestamps[:min_len], '-', 
                 linewidth=1.2, color=colors[color_idx], alpha=0.7)
        color_idx = (color_idx + 1) % len(colors)
    
    # 绘制汇流车辆轨迹
    for vehicle_id, vehicle in merge_vehicles.items():
        x_coords = vehicle.x_coords
        timestamps = np.array(vehicle.timestamps) / 1000  # 转换为秒
        
        min_len = min(len(x_coords), len(timestamps))
        plt.plot(x_coords[:min_len], timestamps[:min_len], '--', 
                 linewidth=1.2, color=colors[color_idx], alpha=0.7)
        color_idx = (color_idx + 1) % len(colors)
    
    plt.xlabel('Position (m)', fontweight='bold')
    plt.ylabel('Time (s)', fontweight='bold')
    plt.title(f'Space-Time Trajectories - {file_id}', fontweight='bold')
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color='k', linestyle='-', label='Main Road'),
        plt.Line2D([0], [0], color='k', linestyle='--', label='Merging Road')
    ]
    plt.legend(handles=legend_elements, frameon=False, loc='upper right')
    
    sns.despine()
    plt.tight_layout()
    
    # 保存轨迹图
    if save_dir:
        save_path = os.path.join(save_dir, f"{file_id}_spacetime_trajectories.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    # 4. 密度等高线图
    plt.figure(figsize=(12, 8))
    
    if main_x_all and merge_x_all:
        # 创建网格
        x_range = np.linspace(x_min, x_max, 100)
        t_range = np.linspace(t_min, t_max, 100)
        X, T = np.meshgrid(x_range, t_range)
        
        # 使用KDE估计密度
        from scipy.stats import gaussian_kde
        
        if len(main_x_all) > 1:
            main_positions = np.vstack([main_x_all, main_t_all])
            main_kernel = gaussian_kde(main_positions)
            main_Z = main_kernel(np.vstack([X.flatten(), T.flatten()]))
            main_Z = main_Z.reshape(X.shape)
            
            # 主道等高线
            CS1 = plt.contour(X, T, main_Z, levels=10, colors='blue', linewidths=1.5, alpha=0.7)
            plt.contourf(X, T, main_Z, levels=10, cmap='Blues', alpha=0.3)
        
        if len(merge_x_all) > 1:
            merge_positions = np.vstack([merge_x_all, merge_t_all])
            merge_kernel = gaussian_kde(merge_positions)
            merge_Z = merge_kernel(np.vstack([X.flatten(), T.flatten()]))
            merge_Z = merge_Z.reshape(X.shape)
            
            # 汇流等高线
            CS2 = plt.contour(X, T, merge_Z, levels=10, colors='red', linewidths=1.5, alpha=0.7)
            plt.contourf(X, T, merge_Z, levels=10, cmap='Oranges', alpha=0.3)
        
        plt.xlabel('Position (m)', fontweight='bold')
        plt.ylabel('Time (s)', fontweight='bold')
        plt.title(f'Density Contour Plot - {file_id}', fontweight='bold')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], color='blue', label='Main Road'),
            plt.Line2D([0], [0], color='red', label='Merging Road')
        ]
        plt.legend(handles=legend_elements, frameon=False, loc='upper right')
        
        sns.despine()
        plt.tight_layout()
        
        # 保存等高线图
        if save_dir:
            save_path = os.path.join(save_dir, f"{file_id}_density_contour.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    # 5. 速度热力图 - 显示速度与位置的关系
    plt.figure(figsize=(12, 8))
    
    # 准备主道和汇流车辆的位置和速度数据
    main_x_speed = []
    main_speeds = []
    merge_x_speed = []
    merge_speeds = []
    
    # 计算车辆速度
    def compute_vehicle_speeds(vehicle):
        vx = np.array(vehicle.vx_values)
        vy = np.array(vehicle.vy_values)
        speeds = np.sqrt(vx**2 + vy**2)
        return speeds
    
    # 提取主道车辆速度数据
    for vehicle_id, vehicle in main_vehicles.items():
        x_coords = vehicle.x_coords
        speeds = compute_vehicle_speeds(vehicle)
        
        min_len = min(len(x_coords), len(speeds))
        
        main_x_speed.extend(x_coords[:min_len])
        main_speeds.extend(speeds[:min_len])
    
    # 提取汇流车辆速度数据
    for vehicle_id, vehicle in merge_vehicles.items():
        x_coords = vehicle.x_coords
        speeds = compute_vehicle_speeds(vehicle)
        
        min_len = min(len(x_coords), len(speeds))
        
        merge_x_speed.extend(x_coords[:min_len])
        merge_speeds.extend(speeds[:min_len])
    
    # 创建自定义色图（从白色渐变到蓝色）
    blue_cmap = LinearSegmentedColormap.from_list('BlueCmap', ['white', 'darkblue'])
    orange_cmap = LinearSegmentedColormap.from_list('OrangeCmap', ['white', 'darkorange'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 主道速度热力图
    if main_x_speed:
        sc1 = ax1.scatter(main_x_speed, main_speeds, c=main_speeds, cmap=blue_cmap, 
                         alpha=0.7, s=20, edgecolor='none')
        
        # 添加颜色条
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(sc1, cax=cax1, label='Speed (m/s)')
        
        ax1.set_ylabel('Speed (m/s)', fontweight='bold')
        ax1.set_title(f'Main Road Speed vs Position - {file_id}', fontweight='bold')
    
    # 汇流速度热力图
    if merge_x_speed:
        sc2 = ax2.scatter(merge_x_speed, merge_speeds, c=merge_speeds, cmap=orange_cmap, 
                         alpha=0.7, s=20, edgecolor='none')
        
        # 添加颜色条
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(sc2, cax=cax2, label='Speed (m/s)')
        
        ax2.set_xlabel('Position (m)', fontweight='bold')
        ax2.set_ylabel('Speed (m/s)', fontweight='bold')
        ax2.set_title(f'Merging Road Speed vs Position - {file_id}', fontweight='bold')
    
    # 移除顶部和右侧边框
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    
    plt.tight_layout()
    
    # 保存速度热力图
    if save_dir:
        save_path = os.path.join(save_dir, f"{file_id}_speed_position_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    print(f"已完成文件 {file_id} 的时空轨迹热图绘制")

if __name__ == "__main__":
    data_dir = r"data_process\behaviour\data"
    main_tra, merge_tra = load_vehicle_data(data_dir)
    
    save_dir = 'data_process/behaviour/results'
    
    # 绘制速度分布
    # plot_improved_speed_distribution(main_tra, merge_tra, save_dir)
    
    
    # 绘制所有轨迹（可以考虑绘制道路边界作为bound）
    trajectory_bounds = get_trajectory_bounds(main_tra, merge_tra)
     # 绘制所有车辆轨迹
    save_path_1 = 'data_process/behaviour/results/vehicle_trajectories.png'
    plot_vehicle_trajectories(main_tra, merge_tra, save_path_1)
    
    save_path_2 = 'data_process/behaviour/results/spacetime_heatmap.png'
    plot_spacetime_heatmap(main_tra, merge_tra, save_path_2)
    
    save_dir = 'data_process/behaviour/results/spacetime_heatmaps'
    plot_file_spacetime_heatmaps(main_tra, merge_tra, 'vehicle_tracks_009', save_dir)
    
    
    
    print("数据读取与绘图完成!")
  