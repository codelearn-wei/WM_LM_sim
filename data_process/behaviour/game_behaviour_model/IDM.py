import numpy as np
from typing import Dict, List, Optional, Tuple
from data_process.behaviour.game_behaviour_model.vehicle import Vehicle
import math


IDM_PARAMS = {
    'a_max': 2,  # Maximum acceleration (m/s^2)
    'v0': 6.0,    # Desired speed (m/s)
    'delta': 4,    # Acceleration exponent
    's0': 2.0,     # Minimum spacing (m)
    'T': 1,      # Safe time headway (s)
    'b': 2.0       # Comfortable deceleration (m/s^2)
}

def calculate_idm_acceleration(ego: Vehicle, leader: Optional[Vehicle], params: Dict) -> float:
    """Calculate acceleration using the IDM model."""
    if leader is None:
        # Free driving: accelerate to desired speed
        return params['a_max'] * (1 - (ego.v / params['v0']) ** params['delta'])
    
    # Calculate distance and relative speed to leader
    dx = leader.x - ego.x
    dy = leader.y - ego.y
    distance = np.sqrt(dx**2 + dy**2)
    delta_v = ego.v - leader.v  # Positive if ego is faster

    # Desired minimum gap
    s_star = params['s0'] + ego.v * params['T'] + (ego.v * delta_v) / (2 * np.sqrt(params['a_max'] * params['b']))
    s_star = max(s_star, params['s0'])  # Ensure s_star is at least s0

    # IDM acceleration
    a_idm = params['a_max'] * (1 - (ego.v / params['v0']) ** params['delta'] - (s_star / distance) ** 2)
    return a_idm


def predict_trajectory(ego: Vehicle, acceleration: float, dt: float = 0.1, horizon: float = 2.0) -> List[Tuple[float, float, float]]:
    """Predict vehicle trajectory based on constant acceleration (longitudinal motion only)."""
    trajectory = []
    t = 0
    current_v = ego.v
    current_x = ego.x
    current_y = ego.y

    while t < horizon:
        # Update speed and position
        current_v = max(0, current_v + acceleration * dt)  # Prevent negative speed
        current_x += current_v * dt * np.cos(ego.heading)
        current_y += current_v * dt * np.sin(ego.heading)
        trajectory.append((current_x, current_y, ego.heading , current_v))
        t += dt
    
    return trajectory

def add_time_to_trajectory(vehicle, trajectory, acceleration, dt=0.1, max_time=2.0):
    """
    给轨迹添加时间和速度信息，生成固定时间间隔的轨迹点
    
    参数:
    - vehicle: 当前车辆对象，包含初始速度信息
    - trajectory: 换道轨迹点列表，每个点为 (x, y, heading)
    - acceleration: 恒定加速度 a_fv
    - dt: 时间步长（固定为0.1秒）
    - max_time: 最大预测时间（秒）
    
    返回:
    - uniform_trajectory: 以固定时间间隔dt采样的轨迹，每个点为 (x, y, heading, v, t)
    """
    # 将原始轨迹与弧长参数化
    arc_lengths = [0]
    total_distance = 0
    
    # 计算累积弧长
    for i in range(1, len(trajectory)):
        prev_x, prev_y, _ = trajectory[i-1]
        curr_x, curr_y, _ = trajectory[i]
        
        segment_distance = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
        total_distance += segment_distance
        arc_lengths.append(total_distance)
    
    # 初始速度和位置
    initial_velocity = vehicle.v
    uniform_trajectory = []
    
    # 生成均匀时间间隔的轨迹点
    for t in np.arange(0, max_time + dt, dt):
        # 计算当前速度 (v = v0 + a*t)
        current_velocity = initial_velocity + acceleration * t
        current_velocity = max(0, current_velocity)  # 确保速度不为负
        
        # 计算从起点开始行驶的距离
        # 使用匀加速公式: s = v0*t + 0.5*a*t^2
        if abs(acceleration) < 1e-6:  # 接近匀速
            distance_traveled = initial_velocity * t
        else:
            distance_traveled = initial_velocity * t + 0.5 * acceleration * t * t
        
        # 如果计算的行驶距离超过轨迹总长度，则使用最后一点的heading并沿该方向延伸
        if distance_traveled >= total_distance and len(trajectory) > 0:
            last_x, last_y, last_heading = trajectory[-1]
            
            # 计算超出轨迹终点的额外距离
            extra_distance = distance_traveled - total_distance
            
            # 沿最后一点的heading延伸
            extended_x = last_x + extra_distance * math.cos(last_heading)
            extended_y = last_y + extra_distance * math.sin(last_heading)
            
            uniform_trajectory.append((extended_x, extended_y, last_heading, current_velocity, t))
        else:
            # 通过距离在原始轨迹中查找对应点并插值
            # 二分查找获取距离索引
            idx = np.searchsorted(arc_lengths, distance_traveled) - 1
            idx = max(0, min(idx, len(arc_lengths) - 2))  # 确保索引有效
            
            # 获取插值比例
            if arc_lengths[idx+1] - arc_lengths[idx] < 1e-6:
                ratio = 0  # 避免除以零
            else:
                ratio = (distance_traveled - arc_lengths[idx]) / (arc_lengths[idx+1] - arc_lengths[idx])
            
            # 线性插值获取位置和航向角
            x1, y1, heading1 = trajectory[idx]
            x2, y2, heading2 = trajectory[idx+1]
            
            # 插值位置
            interp_x = x1 + ratio * (x2 - x1)
            interp_y = y1 + ratio * (y2 - y1)
            
   
            heading_diff = heading2 - heading1
            if heading_diff > math.pi:
                heading_diff -= 2 * math.pi
            elif heading_diff < -math.pi:
                heading_diff += 2 * math.pi
                
            interp_heading = heading1 + ratio * heading_diff
            
            # 规范化航向角到[-π, π]
            while interp_heading > math.pi:
                interp_heading -= 2 * math.pi
            while interp_heading < -math.pi:
                interp_heading += 2 * math.pi
            
            uniform_trajectory.append((interp_x, interp_y, interp_heading, current_velocity, t))
    
    return uniform_trajectory




# Example usage
if __name__ == "__main__":
    # Define vehicles
    main_vehicle = Vehicle(x=85, y=0, v=4, a=0, heading=0, yaw_rate=0, length=4.5, width=1.8, lane='main')
    aux_vehicle = Vehicle(x=80, y=-5, v=5, a=0, heading=0, yaw_rate=0, length=4.5, width=1.8, lane='auxiliary')
    main_front = Vehicle(x=105, y=0, v=4, a=0, heading=0, yaw_rate=0, length=4.5, width=1.8, lane='main')
    aux_front = Vehicle(x=90, y=-5, v=2, a=0, heading=0, yaw_rate=0, length=4.5, width=1.8, lane='auxiliary')
    # 主道车辆跟踪主道前车
    a1 = calculate_idm_acceleration(main_vehicle, main_front, IDM_PARAMS)
    # 辅道车辆跟踪辅道前车
    a2 = calculate_idm_acceleration(main_vehicle, aux_vehicle, IDM_PARAMS)
    # 预测主道车辆轨迹
    main_trajectory1 = predict_trajectory(main_vehicle, a1, dt=0.1, horizon=2.0)
    
    main_trajectory2 = predict_trajectory(main_vehicle, a2, dt=0.1, horizon=2.0)
    
    print("跟踪主道前车主道车辆轨迹" , main_trajectory1)
    print("跟踪辅道前车主道车辆轨迹" , main_trajectory2)

    


