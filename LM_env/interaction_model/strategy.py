import numpy as np
from LM_env.interaction_model.decision_make import *
from LM_env.interaction_model.follower import idm_follow_leader
# TODO:建立主车道的博弈模型

# 该类只定义环境车辆的策略
class StrategyManager:
    """
    用于管理和创建不同类型的驾驶策略
    提供两层决策框架：感知层和决策层
    """
    
    def __init__(self):
        """初始化策略管理器"""
        self.available_strategies = {
            'simple': self.simple_strategy,
            'interactive': self.interactive_strategy,
        }
    
    def get_strategy(self, strategy_name):
        """
        获取指定名称的策略函数
        
        参数:
            strategy_name: 策略名称
        
        返回:
            strategy_func: 策略函数
        """
        if strategy_name not in self.available_strategies:
            raise ValueError(f"策略 '{strategy_name}' 不存在，可用的策略有: {list(self.available_strategies.keys())}")
        
        return self.available_strategies[strategy_name]
    
    def simple_strategy(self, simulator_state):
        """
        简单策略：主车跟随参考线，环境车辆保持恒定速度
        
        参数:
            simulator_state: 仿真器当前状态
        
        返回:
            actions: 字典，键为车辆ID，值为[加速度, 前轮转角]的动作
        """
        # 这里保留原始实现，未做改动
        reference_line = np.array(simulator_state['reference_line'])
        active_vehicles = simulator_state['active_vehicles']
        actions = {}
        
        for vehicle_id, vehicle_state in active_vehicles.items():
            pos = np.array(vehicle_state['position'])
            vel = np.array(vehicle_state['velocity'])
            speed = np.linalg.norm(vel)
            
            distances = np.sum((reference_line - pos) ** 2, axis=1)
            nearest_idx = np.argmin(distances)
            target_idx = min(nearest_idx + 10, len(reference_line) - 1)
            target_point = reference_line[target_idx]
            direction = target_point - pos
            angle_to_target = np.arctan2(direction[1], direction[0])
            current_heading = vehicle_state.get('heading', np.arctan2(vel[1], vel[0]) if speed > 0.1 else 0.0)
            angle_diff = (angle_to_target - current_heading + np.pi) % (2 * np.pi) - np.pi
            steering_angle = np.clip(angle_diff * 0.5, -0.5, 0.5)
            
            if vehicle_id == 0:  # 主车
                acceleration = 0.0
            else:  # 环境车辆
                desired_speed = 3.0 + (vehicle_id % 3)
                acceleration = np.clip((desired_speed - speed) * 0.3, -1.0, 1.0)
            
            actions[vehicle_id] = [acceleration, steering_angle]
        
        return actions
    
    def interactive_strategy(self, simulator_state, interactive_config=None):
        """
        交互策略：主道车辆感知辅道车辆并做出反应
        包含两层决策：感知层和决策层
        
        参数:
            simulator_state: 仿真器当前状态
            interactive_config: 交互配置参数（可选，用于后续扩展）
        
        返回:
            actions: 字典，键为车辆ID，值为[加速度, 前轮转角]的动作
        """
        reference_line = np.array(simulator_state['reference_line'])
        active_vehicles = simulator_state['active_vehicles']
        actions = {}
        
        for vehicle_id, vehicle_state in active_vehicles.items():
            if vehicle_state['is_ego'] == False: 
                steering_angle = compute_steering_angle(vehicle_state, reference_line)
                neighbors_vehicle_categories = categorize_vehicles(vehicle_state, reference_line)
                aux_vehicles = neighbors_vehicle_categories['辅道靠前'] + neighbors_vehicle_categories['辅道靠后']
                acceleration = idm_follow_leader(vehicle_state, neighbors_vehicle_categories)
                            
                # 决策层
                # if not aux_vehicles:
                #     # 没有影响，跟随前车
                #     acceleration = follow_leader(vehicle_state, simulator_state)
                # else:
                #     # 有影响，触发决策逻辑
                #     if should_yield(aux_vehicles, vehicle_state):
                #         # 让车逻辑
                #         acceleration = yield_to_aux(aux_vehicles, vehicle_state)
                #     else:
                #         # 不让车，加速逻辑
                #         acceleration = accelerate_past_aux(aux_vehicles, vehicle_state)
                
                actions[vehicle_id] = [acceleration, steering_angle]
                
                # TODO：主车暂时不提供策略，由强化学习外部输入（后续可以用这个接口测试搭建建立交互模型）
                # else:  # 主车,强化学习或其他算法控制车辆
                #     desired_speed = 3.0 + (vehicle_id % 3)
                #     acceleration = np.clip((desired_speed - speed) * 0.3, -1.0, 1.0)
                #     actions[vehicle_id] = [acceleration, steering_angle]
        
        return actions

def compute_steering_angle(vehicle_state, reference_line, look_ahead_time=2.0, wheelbase=2.5):
        """
        为环境车辆计算转向角
        
        参数:
        vehicle_state: 车辆状态字典
        reference_line: 参考线点数组 (N, 2)
        look_ahead_time: 前视时间（秒）
        wheelbase: 车辆轴距（米）
        
        返回:
        steering_angle: 转向角（弧度）
        """
        # 获取车辆状态
        pos = np.array(vehicle_state['position'])
        vel = np.array(vehicle_state['velocity']) if 'velocity' in vehicle_state else np.array([0.0, 0.0])
        speed = np.linalg.norm(vel)
        heading = vehicle_state.get('heading')
        
        # 计算动态前视距离
        min_speed = 1.0  # 最小速度（m/s）
        look_ahead_distance = max(min_speed, speed) * look_ahead_time
        
        # 找到最近点
        distances = np.sum((reference_line - pos) ** 2, axis=1)
        nearest_idx = np.argmin(distances)
        
        # 沿着参考线找到目标点
        accumulated_distance = 0.0
        for i in range(nearest_idx, len(reference_line) - 1):
            segment = reference_line[i + 1] - reference_line[i]
            segment_length = np.linalg.norm(segment)
            if accumulated_distance + segment_length >= look_ahead_distance:
                ratio = (look_ahead_distance - accumulated_distance) / segment_length
                target_point = reference_line[i] + ratio * segment
                break
            accumulated_distance += segment_length
        else:
            target_point = reference_line[-1]
        
        # 计算目标点相对车辆的局部坐标
        dx = target_point[0] - pos[0]
        dy = target_point[1] - pos[1]
        dx_local = dx * np.cos(heading) + dy * np.sin(heading)
        dy_local = -dx * np.sin(heading) + dy * np.cos(heading)
        
        # 使用纯追踪算法计算转向角
        if speed > 0.1:
            curvature = 2 * dy_local / (look_ahead_distance ** 2)
            steering_angle = np.arctan(curvature * wheelbase)
        else:
            steering_angle = 0.0
        
        # 限制转向角
        max_steering_angle = 0.5
        steering_angle = np.clip(steering_angle, -max_steering_angle, max_steering_angle)
        
        # 更新车辆状态中的航向角（可选）
        vehicle_state['last_heading'] = heading
        
        return steering_angle    