import numpy as np

# 维护主道车辆初始化策略
# TODO：基于数据采样生成主道车辆初始状态
class SimulationInitializer:
    """专门用于构造仿真器重置时的初始状态"""
    
    def __init__(self, static_map):
        """
        初始化构造器
        
        Args:
            static_map: 静态地图数据
        """
        self.static_map = static_map
        self.reference_line = np.array(list(zip(
            static_map['main_road_avg_trajectory']['x_coords'],
            static_map['main_road_avg_trajectory']['y_coords']
        ))) if 'main_road_avg_trajectory' in static_map else None
    
    def create_ego_init_state(self, position_index=0, velocity=1.0):
        """
        创建主车初始状态
        
        Args:
            position_index: 参考线上的位置索引
            velocity: 初始速度大小
            
        Returns:
            ego_init_state: 主车初始状态字典
        """
        if self.reference_line is None or len(self.reference_line) == 0:
            raise ValueError("参考线数据不可用")
            
        return {
            'position': [self.reference_line[position_index][0], self.reference_line[position_index][1]],
            'velocity': [velocity, 0.0],
            'heading': 0.0,
            'length': 5.0,
            'width': 2.0
        }
    
    def create_env_vehicle_state(self, position_index, velocity, length=4.5, width=1.8):
        """
        创建单个环境车辆状态
        
        Args:
            position_index: 参考线上的位置索引
            velocity: 初始速度大小
            length: 车辆长度
            width: 车辆宽度
            
        Returns:
            vehicle_state: 环境车辆状态字典
        """
        if self.reference_line is None or len(self.reference_line) == 0 or position_index >= len(self.reference_line) - 2:
            raise ValueError("参考线数据不可用或索引越界")
        
        # 计算航向角
        heading = np.arctan2(
            self.reference_line[position_index+1][1] - self.reference_line[position_index-1][1], 
            self.reference_line[position_index+1][0] - self.reference_line[position_index-1][0]
        )
        
        return {
            'position': [self.reference_line[position_index][0], self.reference_line[position_index][1]],
            'velocity': [velocity, 0.0],
            'heading': heading,
            'length': length,
            'width': width
        }
    
    def create_env_vehicles_init_states(self, configs):
        """
        创建多个环境车辆初始状态
        
        Args:
            configs: 环境车辆配置列表，每个配置包含position_index, velocity, length, width
            
        Returns:
            env_vehicles_init_states: 环境车辆初始状态列表
        """
        states = []
        for config in configs:
            pos_idx = config.get('position_index', 0)
            vel = config.get('velocity', 0.0)
            length = config.get('length', 4.5)
            width = config.get('width', 1.8)
            
            state = self.create_env_vehicle_state(pos_idx, vel, length, width)
            states.append(state)
            
        return states
    
    def get_simulation_init_states(self, ego_config=None, env_vehicles_configs=None):
        """
        获取完整的仿真初始状态
        
        Args:
            ego_config: 主车配置，包含position_index和velocity
            env_vehicles_configs: 环境车辆配置列表
            
        Returns:
            tuple: (ego_init_state, env_vehicles_init_states)
        """
        # 默认主车配置
        if ego_config is None:
            ego_config = {'position_index': 0, 'velocity': 1.0}
        
        # 默认环境车辆配置
        if env_vehicles_configs is None:
            env_vehicles_configs = [
                {'position_index': 200, 'velocity': 2.0, 'length': 4.5, 'width': 1.8},
                {'position_index': 600, 'velocity': 3.0, 'length': 4.8, 'width': 1.9},
                {'position_index': 800, 'velocity': 4.0, 'length': 5.2, 'width': 2.1}
            ]
        
        # 创建状态
        ego_state = self.create_ego_init_state(ego_config['position_index'], ego_config['velocity'])
        env_states = self.create_env_vehicles_init_states(env_vehicles_configs)
        
        return ego_state, env_states