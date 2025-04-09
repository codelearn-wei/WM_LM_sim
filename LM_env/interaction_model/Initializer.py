import numpy as np
from LM_env.interaction_model.vehicle_model import Vehicle

# TODO: 初始化策略需要大幅度的修改
class SimulationInitializer:
    def __init__(self, static_map, n_lanes=3, lane_width=3.5, min_point_spacing=8.0):
        """
        初始化构造器

        Args:
            static_map: 静态地图数据，包含参考线信息
            n_lanes: 车道数量，默认为3
            lane_width: 车道宽度（单位：米），默认为3.5
            min_point_spacing: 生成点之间的最小间距（单位：米），默认为8.0米
        """
        self.static_map = static_map
        self.n_lanes = n_lanes
        self.lane_width = lane_width
        self.min_point_spacing = min_point_spacing
        self.reference_line = np.array(list(zip(
            static_map['main_road_avg_trajectory']['x_coords'],
            static_map['main_road_avg_trajectory']['y_coords']
        ))) if 'main_road_avg_trajectory' in static_map else None
        if self.reference_line is not None:
            self.s_values = self.compute_arc_lengths()

    def compute_arc_lengths(self):
        """计算参考线的累积弧长"""
        diffs = np.diff(self.reference_line, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        s_values = np.cumsum(distances)
        s_values = np.insert(s_values, 0, 0)
        return s_values

    def select_spawn_points(self, num_points, min_spacing):
        """
        在参考线上选择满足间距条件的生成点

        Args:
            num_points: 需要选择的生成点数量
            min_spacing: 生成点之间的最小间距（单位：米）

        Returns:
            list: 选定的位置索引列表
        """
        if num_points == 0:
            return []

        selected_points = [2]  # 从参考线起点开始
        last_s = self.s_values[1]

        for i in range(1, len(self.s_values)):
            if len(selected_points) >= num_points:
                break
            if self.s_values[i] - last_s >= min_spacing:
                selected_points.append(i)
                last_s = self.s_values[i]

        if len(selected_points) < num_points:
            print(f"警告：无法找到足够的生成点来放置{num_points}辆车，最多可放置{len(selected_points)}辆车，建议减少车辆数量")

        return selected_points[:num_points]

    def create_vehicle(self, position_index, velocity, length, width, lane ,**attributes):
        """
        创建单个车辆对象

        Args:
            position_index: 参考线上的位置索引
            velocity: 初始速度大小
            length: 车辆长度
            width: 车辆宽度
            lane: 车道索引（0到n_lanes-1）

        Returns:
            Vehicle: 车辆对象
        """
        if self.reference_line is None or position_index >= len(self.reference_line) - 2:
            raise ValueError("参考线数据不可用或索引越界")

        ref_point = self.reference_line[position_index]
        heading = np.arctan2(
            self.reference_line[position_index + 1][1] - self.reference_line[position_index - 1][1],
            self.reference_line[position_index + 1][0] - self.reference_line[position_index - 1][0]
        )
        position = ref_point

        return Vehicle(
            position=position,
            velocity=[velocity * np.cos(heading), velocity * np.sin(heading)],
            heading=heading,
            length=length,
            width=width,
            lane=lane,
            s=self.s_values[position_index],
            **attributes
            
        )


    def get_simulation_init_states(self, ego_config, env_vehicles_configs=None):
        """
        获取仿真初始状态
        
        Args:
            ego_config: 主车配置，包含position_index, velocity, length, width, lane, attributes
            env_vehicles_configs: 环境车辆配置，可以是字典（批量生成）或列表（逐个指定）
        
        Returns:
            tuple: (ego_vehicle, env_vehicles)
        """
        occupied_intervals = [[] for _ in range(self.n_lanes)]
        
        # 创建主车
        ego_vehicle = None
        if ego_config:
            pos_idx = ego_config.get('position_index', 0)
            lane = ego_config.get('lane', 0)
            velocity = ego_config.get('velocity', 1.0)
            length = ego_config.get('length', 5.0)
            width = ego_config.get('width', 2.0)
            attributes = ego_config.get('attributes', {'isego': True})
            s = self.s_values[pos_idx]
            interval = [s - length / 2, s + length / 2]
            occupied_intervals[lane].append(interval)
            ego_vehicle = self.create_vehicle(pos_idx, velocity, length, width, lane, **attributes)
        
        # 创建环境车辆
        env_vehicles = []
        
        if isinstance(env_vehicles_configs, dict):
            num_vehicles = env_vehicles_configs.get('num_vehicles', 0)
            velocity_range = env_vehicles_configs.get('velocity_range', (0, 30))
            length_range = env_vehicles_configs.get('length_range', (4.0, 5.5))
            width_range = env_vehicles_configs.get('width_range', (2, 2.1))
            common_attributes = env_vehicles_configs.get('attributes', {'isego': False})
            
            # 新增参数：车辆间距配置
            vehicle_spacing = env_vehicles_configs.get('vehicle_spacing', self.min_point_spacing)
            
            # 自动选择生成点
            max_length = length_range[1]
            spawn_spacing = max(vehicle_spacing, max_length + 0.5)  # 确保车辆不重叠
            spawn_points = self.select_spawn_points(num_vehicles, spawn_spacing)
            
            for pos_idx in spawn_points:
                lane = np.random.randint(0, self.n_lanes)
                velocity = np.random.uniform(velocity_range[0], velocity_range[1])
                length = np.random.uniform(length_range[0], length_range[1])
                width = np.random.uniform(width_range[0], width_range[1])
                s = self.s_values[pos_idx]
                interval = [s - length / 2, s + length / 2]
                
                # 使用配置的车辆间距检查安全距离
                if all(
                    interval[1] + vehicle_spacing < existing[0] or interval[0] - vehicle_spacing > existing[1]
                    for existing in occupied_intervals[lane]
                ):
                    occupied_intervals[lane].append(interval)
                    vehicle = self.create_vehicle(pos_idx, velocity, length, width, lane, **common_attributes)
                    env_vehicles.append(vehicle)
                else:
                    print(f"警告：位置 {pos_idx} 处车辆间距不足，跳过")
        
        elif isinstance(env_vehicles_configs, list):
            # 对于逐个指定的车辆配置，允许每个车辆单独指定间距
            for config in env_vehicles_configs:
                pos_idx = config.get('position_index', 0)
                lane = config.get('lane', 0)
                velocity = config.get('velocity', 0.0)
                length = config.get('length', 4.5)
                width = config.get('width', 1.8)
                attributes = config.get('attributes', {'isego': False})
                
                # 为每个单独的车辆配置添加间距参数
                vehicle_spacing = config.get('vehicle_spacing', self.min_point_spacing)
                
                s = self.s_values[pos_idx]
                interval = [s - length / 2, s + length / 2]
                if all(
                    interval[1] + vehicle_spacing < existing[0] or interval[0] - vehicle_spacing > existing[1]
                    for existing in occupied_intervals[lane]
                ):
                    occupied_intervals[lane].append(interval)
                    vehicle = self.create_vehicle(pos_idx, velocity, length, width, lane, **attributes)
                    env_vehicles.append(vehicle)
                else:
                    print(f"警告：位置 {pos_idx} 处车辆间距不足，跳过")
        
        return ego_vehicle, env_vehicles

    def generate_new_vehicle(self, current_vehicles, velocity_range, length_range, width_range, vehicle_spacing=None, **common_attributes):
        """
        在仿真过程中生成新车辆
        
        Args:
            current_vehicles: 当前场景中的车辆列表
            velocity_range: 速度范围
            length_range: 长度范围
            width_range: 宽度范围
            vehicle_spacing: 车辆间距，如果为None则使用默认值
            **common_attributes: 自定义属性
        
        Returns:
            Vehicle or None: 新车辆对象，若无法生成则返回None
        """
        if vehicle_spacing is None:
            vehicle_spacing = self.min_point_spacing
            
        occupied_intervals = [[] for _ in range(self.n_lanes)]
        for vehicle in current_vehicles:
            s = vehicle.s
            lane = vehicle.lane
            interval = [s - vehicle.length / 2, s + vehicle.length / 2]
            occupied_intervals[lane].append(interval)
        
        lane = np.random.randint(0, self.n_lanes)
        length = np.random.uniform(length_range[0], length_range[1])
        pos_idx = self.find_suitable_position(lane, length, occupied_intervals[lane], vehicle_spacing)
        if pos_idx is None:
            print(f"警告：车道 {lane} 上无法找到合适位置生成新车辆")
            return None
        
        s = self.s_values[pos_idx]
        interval = [s - length / 2, s + length / 2]
        occupied_intervals[lane].append(interval)
        velocity = np.random.uniform(velocity_range[0], velocity_range[1])
        width = np.random.uniform(width_range[0], width_range[1])
        return self.create_vehicle(pos_idx, velocity, length, width, lane, **common_attributes)

    def find_suitable_position(self, lane, length, occupied_intervals, min_spacing=None):
        """
        找到一个不与现有车辆重叠且满足间距条件的位置
        
        Args:
            lane: 车道索引
            length: 新车辆长度
            occupied_intervals: 当前车道的占用区间列表
            min_spacing: 最小间距，默认为类的 min_point_spacing
        
        Returns:
            int or None: 位置索引，若无合适位置则返回None
        """
        if min_spacing is None:
            min_spacing = self.min_point_spacing
        for i, s in enumerate(self.s_values):
            interval = [s - length / 2, s + length / 2]
            if all(
                interval[1] + min_spacing < existing[0] or interval[0] - min_spacing > existing[1]
                for existing in occupied_intervals
            ):
                return i
        return None