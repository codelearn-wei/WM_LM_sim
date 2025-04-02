import numpy as np
from interaction_model.obs_behavior.decision_make import *
from interaction_model.obs_behavior.follower import idm_follow_leader

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
            pos = np.array(vehicle_state['position'])
            vel = np.array(vehicle_state['velocity'])
            speed = np.linalg.norm(vel)
            
            # 计算转向角（与simple_strategy一致）
            distances = np.sum((reference_line - pos) ** 2, axis=1)
            nearest_idx = np.argmin(distances)
            target_idx = min(nearest_idx + 10, len(reference_line) - 1)
            target_point = reference_line[target_idx]
            direction = target_point - pos
            angle_to_target = np.arctan2(direction[1], direction[0])
            current_heading = vehicle_state.get('heading', np.arctan2(vel[1], vel[0]) if speed > 0.1 else 0.0)
            angle_diff = (angle_to_target - current_heading + np.pi) % (2 * np.pi) - np.pi
            steering_angle = np.clip(angle_diff * 0.5, -0.5, 0.5)
            
            #! 先考虑纵向决策，横向先跟踪参考轨迹，后续探究横向对让行策略的影响
            
            if vehicle_state['is_ego'] == False:  # 环境车辆
                # 其中0是主车，其他id是主道车辆          
                # 获得相对本车的车辆列表
                # 字典，包含四个键 ('辅道靠前', '辅道靠后', '主道靠前', '主道靠后')，每个键映射到一个车辆列表。
                neighbors_vehicle_categories = categorize_vehicles(vehicle_state, reference_line)
                aux_vehicles = neighbors_vehicle_categories['辅道靠前'] + neighbors_vehicle_categories['辅道靠后']
                if aux_vehicles == []:
                    # 没有辅道车辆，主道中的前车（先用简单的IDM，后面考虑决策个性化）
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