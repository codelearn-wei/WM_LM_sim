
import numpy as np

# 判断可能和主车发生冲突的辅道车辆
def categorize_vehicles(vehicle_state, reference_line):
    """
    根据 ego vehicle 的位置和参考线，将邻近车辆分类到四个类别。

    参数：
    - vehicle_state: 字典，包含 环境主车的状态
    - reference_line: 形状为 (1000, 2) 的 numpy 数组，表示参考线的 (x, y) 点。
    返回：
    - 字典，包含四个键 ('辅道靠前', '辅道靠后', '主道靠前', '主道靠后')，每个键映射到一个车辆列表。
    """
    # 获取 环境主车的位置
    x_ego, y_ego = vehicle_state['position']
    neighbors = vehicle_state.get('neighbors', [])
    

    # 提取参考线的 x 和 y 坐标
    ref_x = reference_line[:, 0]
    ref_y = reference_line[:, 1]

    # 初始化分类字典
    categories = {
        '辅道靠前': [],  # 辅道，前方
        '辅道靠后': [],  # 辅道，后方
        '主道靠前': [],  # 主道，前方
        '主道靠后': []   # 主道，后方
    }

    # 处理每个邻近车辆
    for neighbor in neighbors:
        # 计算邻近车辆的绝对位置
        relative_position = neighbor['relative_position']
        x_neighbor = x_ego + relative_position[0]
        y_neighbor = y_ego + relative_position[1]

        # 找到参考线中 x 坐标最接近的点的索引
        idx = np.argmin(np.abs(ref_x - x_neighbor))
        ref_y_at_x = ref_y[idx]

        # 根据 y 坐标判断是在主道还是辅道
        if y_neighbor > ref_y_at_x + 1:
            road_type = '辅道'  # 在参考线之上 = 辅道
        else:
            road_type = '主道'  # 在参考线之下 = 主道

        # 根据 x 坐标判断是在前方还是后方
        if x_neighbor < x_ego:
            longitudinal_position = '靠前'  # 在 ego vehicle 前方
        else:
            longitudinal_position = '靠后'  # 在 ego vehicle 后方

        # 组合分类
        category = f"{road_type}{longitudinal_position}"
        categories[category].append(neighbor)

    return categories
    
        
def should_yield(aux_vehicles, vehicle_state):
    """
    判断是否让行给辅道车辆。
    
    参数:
        aux_vehicles: 相关辅道车辆列表
        vehicle_state: 当前车辆状态
    
    返回:
        bool: 是否让行
    """
    if not aux_vehicles:
        return False
    closest_aux = min(aux_vehicles, key=lambda veh: np.linalg.norm(np.array(veh['position']) - np.array(vehicle_state['position'])))
    if closest_aux['position'][0] > vehicle_state['position'][0] and \
       np.linalg.norm(np.array(closest_aux['position']) - np.array(vehicle_state['position'])) < 20:
        return True
    return False


def yield_to_aux(vehicle_state, aux_vehicle_state):
    """
    计算让行时的加速度。
    
    参数:
        vehicle_state: 当前车辆状态
        aux_vehicle_state: 辅道车辆状态
    
    返回:
        float: 加速度
    """
    safe_speed = 0  # 目标安全速度（停车）
    current_speed = np.linalg.norm(vehicle_state['velocity'])
    if current_speed > safe_speed:
        return -3.0  # 减速（假设最大减速度3 m/s²）
    return 0
    

def accelerate_past_aux(vehicle_state, aux_vehicle_state):
    """
    计算加速通过时的加速度。
    
    参数:
        vehicle_state: 当前车辆状态
        aux_vehicle_state: 辅道车辆状态
    
    返回:
        float: 加速度
    """
    return 2.0  # 加速（假设最大加速度2 m/s²）      