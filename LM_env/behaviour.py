import numpy as np

class StrategyManager:
    """
    用于管理和创建不同类型的驾驶策略
    可以根据需求扩展不同的策略函数
    """
    
    def __init__(self):
        """初始化策略管理器"""
        self.available_strategies = {
            'simple': self.simple_strategy,
            # 可在此添加更多策略
            # 'interactive': self.interactive_strategy,
            # 'data_driven': self.data_driven_strategy,
            # 'rule_based': self.rule_based_strategy,
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
        reference_line = np.array(simulator_state['reference_line'])
        active_vehicles = simulator_state['active_vehicles']
        actions = {}
        
        for vehicle_id, vehicle_state in active_vehicles.items():
            pos = np.array(vehicle_state['position'])
            vel = np.array(vehicle_state['velocity'])
            speed = np.linalg.norm(vel)
            
            # 计算距参考线最近的点
            distances = np.sum((reference_line - pos) ** 2, axis=1)
            nearest_idx = np.argmin(distances)
            
            # 选择前方的目标点
            target_idx = min(nearest_idx + 10, len(reference_line) - 1)
            target_point = reference_line[target_idx]
            
            # 计算朝向目标点的方向
            direction = target_point - pos
            angle_to_target = np.arctan2(direction[1], direction[0])
            
            # 计算当前航向角与目标方向之间的角度差
            if 'heading' in vehicle_state:
                current_heading = vehicle_state['heading']
            else:
                # 如果没有航向角信息，从速度推算
                if speed > 0.1:
                    current_heading = np.arctan2(vel[1], vel[0])
                else:
                    current_heading = 0.0
            
            # 计算转向角
            angle_diff = (angle_to_target - current_heading + np.pi) % (2 * np.pi) - np.pi
            steering_angle = np.clip(angle_diff * 0.5, -0.5, 0.5)  # 简单的P控制
            
            # 设置加速度
            if vehicle_id == 0:  # 主车
                # 主车目标速度为5m/s
                desired_speed = 5.0
                acceleration = np.clip((desired_speed - speed) * 0.5, -2.0, 2.0)
            else:  # 环境车辆
                # 环境车辆保持恒定速度
                desired_speed = 3.0 + (vehicle_id % 3)  # 不同环境车设置不同速度
                acceleration = np.clip((desired_speed - speed) * 0.3, -1.0, 1.0)
            
            # 返回加速度和转向角
            actions[vehicle_id] = [acceleration, steering_angle]
        
        return actions
    
    # 可以添加更多策略函数
    def create_rule_based_strategy(self, rules_config):
        """
        创建基于规则的策略
        
        参数:
        rules_config: 规则配置参数
        
        返回:
        strategy_func: 策略函数
        """
        def rule_based_strategy(simulator_state):
            # 基于规则的策略实现
            # ...
            return {}
        
        return rule_based_strategy
    
    def create_interactive_strategy(self, interactive_config=None):
        """
        创建交互式策略，允许用户输入或控制车辆行为
        
        参数:
        interactive_config: 交互配置参数
        
        返回:
        strategy_func: 策略函数
        """
        def interactive_strategy(simulator_state):
            # 交互式策略实现
            # 可以在这里读取用户输入或外部控制信号
            # ...
            return {}
        
        return interactive_strategy