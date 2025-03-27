
import numpy as np

# 判断可能和主车发生冲突的辅道车辆
def detect_related_aux_vehicles(neighbors, simulator_state):
    """
    检测可能与主车发生冲突的辅道车辆。
    
    参数:
        neighbors: 邻居车辆列表，来自 simulator_state['active_vehicles'][vid]['neighbors']
        simulator_state: 仿真器当前状态
    
    返回:
        list: 相关辅道车辆状态列表
    """
    related_aux_vehicles = []
    for neighbor in neighbors:
        neighbor_id = neighbor['vehicle_id']
        neighbor_state = simulator_state['active_vehicles'][neighbor_id]
    return related_aux_vehicles


def follow_leader(vehicle_state, simulator_state):
    """
    使用IDM模型计算跟车加速度。
    
    参数:
        vehicle_state: 当前车辆状态
        simulator_state: 仿真器当前状态
    
    返回:
        float: 加速度
    """
    leader = find_leader(vehicle_state, simulator_state)
    if leader is None:
        return 2.0  # 无前车，加速到最大速度（假设2 m/s²）
    return idm_acceleration(vehicle_state, leader)

def find_leader(vehicle_state, simulator_state):
    """
    找到前方最近的车辆。
    """
    min_distance = float('inf')
    leader = None
    for other_id, other_state in simulator_state['active_vehicles'].items():
        if other_id != vehicle_state['vehicle_id']:
            other_pos = np.array(other_state['position'])
            pos = np.array(vehicle_state['position'])
            if other_pos[0] > pos[0]:  # x轴为前进方向
                distance = other_pos[0] - pos[0]
                if distance < min_distance:
                    min_distance = distance
                    leader = other_state
    return leader

def idm_acceleration(vehicle_state, leader):
    """
    IDM模型计算加速度。
    """
    desired_speed = 10.0  # 期望速度 (m/s)
    time_headway = 1.5  # 时间间距 (s)
    min_gap = 2.0  # 最小间距 (m)
    max_acceleration = 2.0  # 最大加速度 (m/s²)
    comfort_deceleration = 3.0  # 舒适减速度 (m/s²)

    v = np.linalg.norm(vehicle_state['velocity'])
    v_leader = np.linalg.norm(leader['velocity'])
    s = leader['position'][0] - vehicle_state['position'][0] - 5.0  # 假设车长5米

    s_star = min_gap + v * time_headway + (v * (v - v_leader)) / (2 * np.sqrt(max_acceleration * comfort_deceleration))
    a = max_acceleration * (1 - (v / desired_speed)**4 - (s_star / s)**2)
    return np.clip(a, -comfort_deceleration, max_acceleration)
    
    
        
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