# import numpy as np

# # 维护主道车辆初始化策略

# # !需要方便调节以获得不同的训练状态，参数可调，符合自然驾驶数据分布，有方便的参数调节接口。

# TODO 初始化不要一开始就在主道上，应该一直都生成，运行到某个为止消失。
# ! 策略是先生成几辆主道车辆，然后再起始点位置，定时生成车辆，直到达到最大数量为止。（TODO）
import numpy as np
from LM_env.interaction_model.vehicle_model import Vehicle

class SimulationInitializer:
    def __init__(self, static_map):
        """
        初始化构造器
        
        Args:
            static_map: 静态地图数据，包含参考线信息
        """
        self.static_map = static_map
        self.reference_line = np.array(list(zip(
            static_map['main_road_avg_trajectory']['x_coords'],
            static_map['main_road_avg_trajectory']['y_coords']
        ))) if 'main_road_avg_trajectory' in static_map else None
    
    def create_vehicle(self, position_index, velocity, length, width, **kwargs):
        """
        创建单个车辆对象
        
        Args:
            position_index: 参考线上的位置索引
            velocity: 初始速度大小
            length: 车辆长度
            width: 车辆宽度
            **kwargs: 自定义属性
        
        Returns:
            Vehicle: 车辆对象
        """
        if self.reference_line is None or position_index >= len(self.reference_line) - 2:
            raise ValueError("参考线数据不可用或索引越界")
        
        position = self.reference_line[position_index]
        heading = np.arctan2(
            self.reference_line[position_index + 1][1] - self.reference_line[position_index - 1][1],
            self.reference_line[position_index + 1][0] - self.reference_line[position_index - 1][0]
        )
        
        return Vehicle(
            position=position,
            velocity=[velocity, 0.0],
            heading=heading,
            length=length,
            width=width,
            **kwargs
        )
    
    def generate_vehicles(self, num_vehicles, position_range, velocity_range, length_range, width_range, **common_attributes):
        """
        批量生成环境车辆
        
        Args:
            num_vehicles: 生成车辆数量
            position_range: 位置索引范围 (min_idx, max_idx)
            velocity_range: 速度范围 (min_vel, max_vel)
            length_range: 长度范围 (min_len, max_len)
            width_range: 宽度范围 (min_wid, max_wid)
            **common_attributes: 所有车辆共享的自定义属性
        
        Returns:
            list: 车辆对象列表
        """
        vehicles = []
        for _ in range(num_vehicles):
            pos_idx = np.random.randint(position_range[0], position_range[1])
            velocity = np.random.uniform(velocity_range[0], velocity_range[1])
            length = np.random.uniform(length_range[0], length_range[1])
            width = np.random.uniform(width_range[0], width_range[1])
            
            vehicle = self.create_vehicle(pos_idx, velocity, length, width, **common_attributes)
            vehicles.append(vehicle)
        return vehicles
    
    def set_strategy_params_for_vehicles(self, vehicles, strategy_params_list):
        """
        为车辆设置策略参数
        
        Args:
            vehicles: 车辆对象列表
            strategy_params_list: 策略参数列表，每个元素是一个字典
        """
        if len(vehicles) != len(strategy_params_list):
            raise ValueError("策略参数列表长度与车辆数量不匹配")
        
        for vehicle, params in zip(vehicles, strategy_params_list):
            vehicle.set_strategy_params(**params)
    
    def get_simulation_init_states(self, ego_config, env_vehicles_configs=None):
        """
        获取仿真初始状态
        
        Args:
            ego_config: 主车配置，包含position_index, velocity, length, width, attributes
            env_vehicles_configs: 环境车辆配置，可以是字典（批量生成）或列表（逐个指定）
        
        Returns:
            tuple: (ego_vehicle, env_vehicles)
        """
        # 创建主车
        ego_vehicle = []
        if ego_config:
            ego_vehicle = self.create_vehicle(
                position_index=ego_config.get('position_index', 0),
                velocity=ego_config.get('velocity', 1.0),
                length=ego_config.get('length', 5.0),
                width=ego_config.get('width', 2.0),
                **ego_config.get('attributes', {})
            )       
            
        # 创建环境车辆
        env_vehicles = []
        if isinstance(env_vehicles_configs, dict):
            # 批量生成
            num_vehicles = env_vehicles_configs.get('num_vehicles', 0)
            position_range = env_vehicles_configs.get('position_range', (0, len(self.reference_line) - 1))
            velocity_range = env_vehicles_configs.get('velocity_range', (0, 30))
            length_range = env_vehicles_configs.get('length_range', (4.0, 5.5))
            width_range = env_vehicles_configs.get('width_range', (1.8, 2.2))
            common_attributes = env_vehicles_configs.get('attributes', {})
            
            env_vehicles = self.generate_vehicles(
                num_vehicles, position_range, velocity_range, length_range, width_range, **common_attributes
            )
        elif isinstance(env_vehicles_configs, list):
            # 逐个指定
            for config in env_vehicles_configs:
                vehicle = self.create_vehicle(
                    position_index=config.get('position_index', 0),
                    velocity=config.get('velocity', 0.0),
                    length=config.get('length', 4.5),
                    width=config.get('width', 1.8),
                    **config.get('attributes', {})
                )
                env_vehicles.append(vehicle)
        
        return ego_vehicle, env_vehicles