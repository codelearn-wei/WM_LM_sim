import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent  # 根据你的文件层级调整
sys.path.append(str(project_root))
from data_process.train_raw_data import organize_by_frame,  classify_vehicles_by_frame_1
from data_process.LM_scene import LMScene
import json
import pandas as pd
from collections import OrderedDict 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import splprep, splev , splrep
from scipy.signal import savgol_filter
import os 
import pickle

def _get_frame_data(json_path: str, excel_path: str):
    """
    工具函数：处理Excel和JSON文件中的车辆数据，并返回分类后的结果。
    
    Args:
        json_path (str): JSON 文件路径（地图数据）
        excel_path (str): Excel 文件路径（车辆轨迹数据）
    
    Returns:
        dict: 按帧分类后的车辆数据
    
    Raises:
        FileNotFoundError: 如果输入的文件路径不存在
        ValueError: 如果文件内容无法正确解析
    """
    # 检查文件是否存在
    if not Path(json_path).exists():
        raise FileNotFoundError(f"JSON 文件未找到: {json_path}")
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Excel 文件未找到: {excel_path}")

    # 读取 JSON 文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"无法解析 JSON 文件: {json_path}")

    # 读取 Excel 文件（假设是 CSV 格式，也可以用 pd.read_excel 处理 .xlsx 文件）
    try:
        vehicle_data = pd.read_csv(excel_path)
    except Exception as e:
        raise ValueError(f"无法读取 Excel 文件: {excel_path}, 错误: {str(e)}")

    # 初始化 LMScene 对象
    scene = LMScene(json_path, excel_path)

    # 第一步：按帧组织数据
    x = scene.vehicles
    frame_data = organize_by_frame(scene.vehicles)

    # 第二步：获取边界数据
    upper_bd = scene.get_upper_boundary()
    auxiliary_bd = scene.get_auxiliary_dotted_line()

    # 第三步：分类车辆数据
    classified_data = classify_vehicles_by_frame_1(frame_data, upper_bd, auxiliary_bd)

    return classified_data



def _get_vehicle_tra(json_path: str, excel_path: str):
        """
        工具函数：处理Excel和JSON文件中的车辆数据，并返回分类后的结果。
        
        Args:
            json_path (str): JSON 文件路径（地图数据）
            excel_path (str): Excel 文件路径（车辆轨迹数据）
        
        Returns:
            dict: 按帧分类后的车辆数据
        
        Raises:
            FileNotFoundError: 如果输入的文件路径不存在
            ValueError: 如果文件内容无法正确解析
        """
        # 检查文件是否存在
        if not Path(json_path).exists():
            raise FileNotFoundError(f"JSON 文件未找到: {json_path}")
        if not Path(excel_path).exists():
            raise FileNotFoundError(f"Excel 文件未找到: {excel_path}")

        # 读取 JSON 文件
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"无法解析 JSON 文件: {json_path}")

        # 读取 Excel 文件（假设是 CSV 格式，也可以用 pd.read_excel 处理 .xlsx 文件）
        try:
            vehicle_data = pd.read_csv(excel_path)
        except Exception as e:
            raise ValueError(f"无法读取 Excel 文件: {excel_path}, 错误: {str(e)}")

        # 初始化 LMScene 对象
        scene = LMScene(json_path, excel_path)
        # 获取所有车辆的 ID 集合
        all_vehicle_ids = set()
        merge_vehicle_ids = set()
        for key in scene.vehicles.keys():
            all_vehicle_ids.add(scene.vehicles[key].track_id)     
        for i in range(len(scene.merge_vehicle)):
            merge_vehicle_ids.add(scene.merge_vehicle[i].track_id)      
        main_road_vehicle_ids = all_vehicle_ids - merge_vehicle_ids      
        merge_vehicles = scene.merge_vehicle
        main_road_vehicles = {vid: scene.vehicles[vid] for vid in main_road_vehicle_ids}
        map_dict = scene.map_dict
        # 滤除不在研究范围内的地图
        keys_except_last = list(map_dict.keys())[:-1]
        for key in keys_except_last:
            array = map_dict[key]
            filtered_array = array[array[:, 0] >= 1030]
            map_dict[key] = filtered_array   
            
        return main_road_vehicles , merge_vehicles , map_dict

def _get_mutli_vehicle_tra(json_path: str, dir_path: str):
    all_vehicle_tra = {}
    for csv_file in Path(dir_path).glob("*.csv"):
        file_name = csv_file.stem
        main_road_vehicles , merge_vehicles , map_dict = _get_vehicle_tra(json_path, str(csv_file))
        all_vehicle_tra[file_name] = {
                'main_road_vehicles': main_road_vehicles,
                'merge_vehicles': merge_vehicles,
                'map_dict': map_dict
            }
    return all_vehicle_tra
    
def _calculate_central_trajectory(vehicles_dict, x_threshold=1030):
    """
    计算中心轨迹，只使用x坐标大于阈值的轨迹点
    
    参数:
    vehicles_dict: 包含多辆车轨迹数据的字典
    x_threshold: x坐标筛选阈值，默认为1030
    
    返回:
    dict: 包含平滑处理后的中心轨迹数据
    """
    # 找出符合条件的最长的轨迹，用作参考骨架
    max_length = 0
    reference_vehicle = None
    filtered_vehicles = {}
    
    # 筛选车辆和轨迹点
    for vehicle_id, vehicle in vehicles_dict.items():
        # 筛选x坐标大于阈值的点
        valid_indices = [i for i, x in enumerate(vehicle.x_coords) if x > x_threshold]
        
        if len(valid_indices) > 0:
            # 创建一个只包含有效点的副本
            filtered_vehicle = type('', (), {})()
            filtered_vehicle.x_coords = [vehicle.x_coords[i] for i in valid_indices]
            filtered_vehicle.y_coords = [vehicle.y_coords[i] for i in valid_indices]
            
            if hasattr(vehicle, 'vx_values'):
                filtered_vehicle.vx_values = [vehicle.vx_values[i] for i in valid_indices]
            if hasattr(vehicle, 'vy_values'):
                filtered_vehicle.vy_values = [vehicle.vy_values[i] for i in valid_indices]
            if hasattr(vehicle, 'psi_rad_values'):
                filtered_vehicle.psi_rad_values = [vehicle.psi_rad_values[i] for i in valid_indices]
            if hasattr(vehicle, 'timestamps'):
                filtered_vehicle.timestamps = [vehicle.timestamps[i] for i in valid_indices]
            
            filtered_vehicles[vehicle_id] = filtered_vehicle
            
            # 更新最长轨迹
            if len(filtered_vehicle.x_coords) > max_length:
                max_length = len(filtered_vehicle.x_coords)
                reference_vehicle = filtered_vehicle
    
    if reference_vehicle is None or max_length == 0:
        print(f"没有找到x坐标大于{x_threshold}的有效轨迹点")
        return None
    
    # 使用最长轨迹作为参考，将其他轨迹映射到它上面
    ref_x = np.array(reference_vehicle.x_coords)
    ref_y = np.array(reference_vehicle.y_coords)
    
    # 初始化平均值数组
    avg_y = np.zeros_like(ref_x)
    avg_vx = np.zeros_like(ref_x)
    avg_vy = np.zeros_like(ref_x)
    avg_psi = np.zeros_like(ref_x)
    count = np.zeros_like(ref_x)
    
    # 将参考轨迹添加到平均值中
    avg_y += ref_y
    if hasattr(reference_vehicle, 'vx_values'):
        avg_vx += np.array(reference_vehicle.vx_values)
    if hasattr(reference_vehicle, 'vy_values'):
        avg_vy += np.array(reference_vehicle.vy_values)
    if hasattr(reference_vehicle, 'psi_rad_values'):
        avg_psi += np.array(reference_vehicle.psi_rad_values)
    count += 1
    
    # 对每个其他车辆，匹配最接近的点并添加到平均值中
    for vehicle_id, vehicle in filtered_vehicles.items():
        if vehicle == reference_vehicle:
            continue
            
        for i in range(len(vehicle.x_coords)):
            x = vehicle.x_coords[i]
            y = vehicle.y_coords[i]
            
            # 找到参考轨迹中最接近的点
            closest_idx = np.argmin(np.abs(ref_x - x))
            
            # 添加到平均值中
            avg_y[closest_idx] += y
            if hasattr(vehicle, 'vx_values'):
                avg_vx[closest_idx] += vehicle.vx_values[i]
            if hasattr(vehicle, 'vy_values'):
                avg_vy[closest_idx] += vehicle.vy_values[i]
            if hasattr(vehicle, 'psi_rad_values'):
                avg_psi[closest_idx] += vehicle.psi_rad_values[i]
            count[closest_idx] += 1
    
    # 计算平均值
    # 避免除以零
    count = np.maximum(count, 1)
    avg_y /= count
    avg_vx /= count
    avg_vy /= count
    avg_psi /= count
    
    # 平滑处理
    smooth_x, smooth_y = _smooth_trajectory(ref_x, avg_y)
    
    # 重新插值速度和方向
    smooth_vx = np.interp(smooth_x, ref_x, avg_vx)
    smooth_vy = np.interp(smooth_x, ref_x, avg_vy)
    smooth_psi = np.interp(smooth_x, ref_x, avg_psi)
    
    # 构建结果
    result = {
        'x_coords': smooth_x.tolist(),
        'y_coords': smooth_y.tolist(),
        'vx_values': smooth_vx.tolist(),
        'vy_values': smooth_vy.tolist(),
        'psi_rad_values': smooth_psi.tolist(),
        'raw_x': ref_x.tolist(),
        'raw_y': avg_y.tolist()
    }
    
    # 添加时间戳（如果存在）
    if hasattr(reference_vehicle, 'timestamps'):
        result['timestamps'] = reference_vehicle.timestamps
    
    return result

def _smooth_trajectory(x_coords, y_coords):
    """
    对轨迹进行平滑处理，使用样条插值和Savitzky-Golay滤波器
    
    参数:
    x_coords: x坐标数组
    y_coords: y坐标数组
    
    返回:
    smooth_x: 平滑后的x坐标数组
    smooth_y: 平滑后的y坐标数组
    """
    # 移除重复的x坐标点（否则样条插值会失败）
    points = np.array([x_coords, y_coords]).T
    _, unique_indices = np.unique(points[:, 0], return_index=True)
    unique_indices = np.sort(unique_indices)
    
    x_unique = x_coords[unique_indices]
    y_unique = y_coords[unique_indices]
    
    # 确保有足够的点进行插值
    if len(x_unique) < 4:
        print(f"警告: 轨迹点数不足，无法进行样条插值 (只有 {len(x_unique)} 个点)")
        return np.array(x_coords), np.array(y_coords)
    
    try:
        # 使用样条插值平滑轨迹
        # s参数控制平滑程度，越大越平滑
        tck, u = splprep([x_unique, y_unique], s=len(x_unique)*0.5, k=3)
        
        # 创建更密集的点进行插值
        u_new = np.linspace(0, 1, 5 * len(x_unique))
        smooth_x, smooth_y = splev(u_new, tck)
        
        # 应用Savitzky-Golay滤波器进一步平滑
        # window_length必须是奇数且大于polynomial_order
        window_length = min(51, len(smooth_x) - 2 if len(smooth_x) % 2 == 0 else len(smooth_x) - 1)
        if window_length > 3:
            smooth_y = savgol_filter(smooth_y, window_length, 3)
        
        return smooth_x, smooth_y
    except Exception as e:
        print(f"平滑处理失败: {e}")
        return np.array(x_coords), np.array(y_coords)
    
   
   
def _get_static_env(json_path: str, excel_path: str, visualize=False, save_path=None ,x_threshold=1030):
    """
    获取主道特征，计算并可视化平均轨迹
    
    参数:
    json_path: JSON文件路径
    excel_path: Excel文件路径
    visualize: 是否可视化，默认为True
    save_path: 可视化图像保存路径，默认为None（不保存）
    
    返回:
    main_road_avg_trajectory: 平滑后的主道平均轨迹
    reference_lanes: 基于平均轨迹生成的参考车道线
    """
    # 检查文件是否存在
    if not Path(json_path).exists():
        raise FileNotFoundError(f"JSON 文件未找到: {json_path}")
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"csv 文件未找到: {excel_path}")

    # 读取 JSON 文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"无法解析 JSON 文件: {json_path}")
        
    main_road_vehicles, merge_vehicles , map_dict= _get_vehicle_tra(json_path, excel_path)
    main_road_vehicles = OrderedDict(sorted(main_road_vehicles.items()))
   
    # 计算平均轨迹并进行平滑处理
    main_road_avg_trajectory = _calculate_central_trajectory(main_road_vehicles)
    # 可视化处理
    if visualize and main_road_avg_trajectory is not None:
        _visualize_trajectories(main_road_vehicles, main_road_avg_trajectory, x_threshold, save_path)  
    static_env = {}
    static_env['map_dict'] = map_dict
    static_env['main_road_avg_trajectory'] = main_road_avg_trajectory 
    return static_env

def _calc_static_map(all_static_env):
    static_map = {}
    
    # 如果有多个轨迹，需要先确定一个统一的采样点数量
    max_points = 0
    for key in all_static_env.keys():
        if 'main_road_avg_trajectory' in all_static_env[key]:
            # 假设轨迹数据结构中有 x_coords 和 y_coords
            traj_length = len(all_static_env[key]['main_road_avg_trajectory']['x_coords'])
            max_points = max(max_points, traj_length)
    
    if max_points == 0:
        return static_map
    
    # 存储所有轨迹数据
    all_trajectories = {}
    
    # 遍历所有轨迹并重新采样
    for key in all_static_env.keys():
        if 'main_road_avg_trajectory' in all_static_env[key]:
            traj = all_static_env[key]['main_road_avg_trajectory']
            
            # 获取原始 x 和 y 坐标
            x_orig = np.array(traj['x_coords'])
            y_orig = np.array(traj['y_coords'])
            
            # 使用样条插值进行重采样，确保所有轨迹有相同数量的点
            t_orig = np.linspace(0, 1, len(x_orig))
            t_new = np.linspace(0, 1, max_points)
            
            # 创建样条表示
            x_spl = splrep(t_orig, x_orig, s=0)
            y_spl = splrep(t_orig, y_orig, s=0)
            
            # 对样条进行求值
            x_resampled = splev(t_new, x_spl)
            y_resampled = splev(t_new, y_spl)
            
            # 存储重采样后的轨迹
            if 'x_coords' not in all_trajectories:
                all_trajectories['x_coords'] = []
                all_trajectories['y_coords'] = []
            
            all_trajectories['x_coords'].append(x_resampled)
            all_trajectories['y_coords'].append(y_resampled)
    
    # 计算平均轨迹
    if 'x_coords' in all_trajectories and len(all_trajectories['x_coords']) > 0:
        avg_x = np.mean(all_trajectories['x_coords'], axis=0)
        avg_y = np.mean(all_trajectories['y_coords'], axis=0)
        
        # 应用平滑处理以确保轨迹的平滑性
        # 使用样条平滑，s 参数控制平滑程度
        t_smooth = np.linspace(0, 1, max_points)
        smoothing_factor = 0.8 * max_points  # 调整此参数以控制平滑程度
        
        # 创建平滑的样条表示
        x_smooth_spl = splrep(t_smooth, avg_x, s=smoothing_factor)
        y_smooth_spl = splrep(t_smooth, avg_y, s=smoothing_factor)
        
        # 获取平滑后的轨迹
        smooth_x = splev(t_smooth, x_smooth_spl)
        smooth_y = splev(t_smooth, y_smooth_spl)
        
        # 存储到 static_map
        static_map['main_road_avg_trajectory'] = {
            'x_coords': smooth_x.tolist(),
            'y_coords': smooth_y.tolist()
        }
    
    return static_map, all_trajectories, avg_x, avg_y, smooth_x, smooth_y
        

def get_mutli_static_env(json_path: str, dir_path: str):
    all_static_env = {}
    for csv_file in Path(dir_path).glob("*.csv"):
        file_name = csv_file.stem
        static_env = _get_static_env(json_path, str(csv_file))
        all_static_env[file_name] = static_env
    static_map = _calc_static_map(all_static_env) 
    static_map[0]['map_dict'] = static_env['map_dict']
    return  all_static_env , static_map

   
# def get_dynamic_env(json_path: str, excel_path: str):
    
       
def _visualize_trajectories(vehicles_dict, central_trajectory, x_threshold=1030, save_path=None):
    """
    可视化所有车辆轨迹和平滑后的中心轨迹，标记x坐标阈值
    
    参数:
    vehicles_dict: 包含多辆车轨迹数据的字典
    central_trajectory: 计算得到的平均轨迹
    x_threshold: x坐标筛选阈值，默认为1030
    save_path: 保存图片的路径，默认为None
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制所有原始轨迹（灰色线）
    for vehicle_id, vehicle in vehicles_dict.items():
        plt.plot(vehicle.x_coords, vehicle.y_coords, color='lightgray', linewidth=0.5, alpha=0.3)
    
    # 绘制大于阈值的轨迹点（更深的灰色）
    for vehicle_id, vehicle in vehicles_dict.items():
        valid_points = [(x, y) for x, y in zip(vehicle.x_coords, vehicle.y_coords) if x > x_threshold]
        if valid_points:
            valid_x, valid_y = zip(*valid_points)
            plt.plot(valid_x, valid_y, color='gray', linewidth=0.5, alpha=0.5)
    
    # 绘制原始平均轨迹（如果存在）
    if central_trajectory and 'raw_x' in central_trajectory and 'raw_y' in central_trajectory:
        plt.plot(central_trajectory['raw_x'], central_trajectory['raw_y'], 
                 color='blue', linewidth=1, alpha=0.7, linestyle='--')
    
    # 绘制平滑后的平均轨迹（如果存在）
    if central_trajectory and 'x_coords' in central_trajectory and 'y_coords' in central_trajectory:
        plt.plot(central_trajectory['x_coords'], central_trajectory['y_coords'], 
                 color='red', linewidth=2, alpha=1.0)
    
    # 绘制垂直线表示x阈值
    plt.axvline(x=x_threshold, color='green', linestyle='--', alpha=0.7, 
                label=f'x坐标阈值 ({x_threshold})')
    
    # 设置图例
    legend_elements = [
        Line2D([0], [0], color='lightgray', lw=1, alpha=0.5, label='全部轨迹'),
        Line2D([0], [0], color='gray', lw=1, alpha=0.7, label=f'x > {x_threshold}的轨迹'),
        Line2D([0], [0], color='blue', lw=1, alpha=0.7, linestyle='--', label='原始平均轨迹'),
        Line2D([0], [0], color='red', lw=2, label='平滑后的平均轨迹'),
        Line2D([0], [0], color='green', lw=1, linestyle='--', alpha=0.7, label=f'x坐标阈值 ({x_threshold})')
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)
    
    # 设置坐标轴标签和标题
    plt.xlabel('X 坐标 (m)', fontsize=12)
    plt.ylabel('Y 坐标 (m)', fontsize=12)
    plt.title(f'车辆轨迹可视化 (仅使用 x > {x_threshold} 的轨迹计算平均)', fontsize=14)
    
    # 设置等比例坐标轴
    plt.axis('equal')
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 保存或显示图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
 
def save_static_map(static_map, output_path="static_map.json"):
    """
    保存静态地图数据到文件
    
    参数:
    static_map (dict): 要保存的静态地图数据
    output_path (str): 输出文件路径，默认为'static_map.json'
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 检查文件扩展名确定保存格式
    _, ext = os.path.splitext(output_path)
    

    if ext.lower() == '.pkl' or ext.lower() == '.pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(static_map, f)

    print(f"静态地图数据已保存到: {output_path}")
        
if __name__ == "__main__":
    # 获取环境的静态地图
    json_file = "LM_data/map/DR_CHN_Merging_ZS.json"
    dir_file = "LM_data/data/DR_CHN_Merging_ZS"
    all_static_env , static_map = get_mutli_static_env(json_file, dir_file)
    save_static_map(static_map, output_path="LM_env/LM_static_map.pkl")
    print("完成主道轨迹提取与平滑处理")
          
    
