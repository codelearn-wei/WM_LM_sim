import numpy as np
from LM_env.utils.Vehicle_model import Vehicle


# 现在可用于初始化的对象：
#! 单车智能汇入模式
# 1、主道参考线（生成环境车辆）
# 2、辅道参考线（生成主车）
#! 多车智能汇入模式
# 1、主道参考线（生成主道环境车辆）
# 2、辅道参考线（生成多辆主车）
#! 随机汇流汇入模式
# 1、主道参考线（生成环境车辆和主车）
# 2、辅道参考线（生成多辆主车）

# TODO：初始化一定要数据样本多样
# TODO：初始化的车辆数量要可调
# TODO：初始化车的行为要符合实际的数据
# TODO：初始化主道车辆要能一直生成，一直添加
# TODO：初始化主车的行为要为强化学习服务
# TODO：要定义好什么车大概率不让，大概率让（和当前情况和驾驶风格，决策价值观都有关系）

class SimulationInitializer:
    """仿真初始化基类，提供通用功能"""
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
        
        # 主道参考线
        self.reference_line = np.array(list(zip(
            static_map['main_road_avg_trajectory']['x_coords'],
            static_map['main_road_avg_trajectory']['y_coords']
        ))) if 'main_road_avg_trajectory' in static_map else None
        

        self.reference_xy = self.reference_line[::20]  # 每20个点取一个，减少点数
        if self.reference_xy is not None:
            self.s_values = self.compute_arc_lengths(self.reference_xy)
        
        # 辅道参考线
        self.aux_reference_lines = np.array(list(zip(
            static_map['aux_reference_lanes']['x_coords'],
            static_map['aux_reference_lanes']['y_coords']
        ))) if 'aux_reference_lanes' in static_map else None
        self.aux_reference_xy = self.aux_reference_lines[::-5]  # 每5个点取一个，减少点数,反转一下
        if self.aux_reference_xy is not None:
            self.aux_s_values = self.compute_arc_lengths(self.aux_reference_xy)

    def compute_arc_lengths(self, reference_xy):
        """计算参考线的累积弧长"""
        diffs = np.diff(reference_xy, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        s_values = np.cumsum(distances)
        s_values = np.insert(s_values, 0, 0)
        return s_values

    def select_spawn_points(self, reference_xy, s_values, num_points, min_spacing):
        """在指定参考线上选择满足间距条件的生成点"""
        if num_points == 0:
            return []

        selected_points = [2]  # 从起点开始
        last_s = s_values[1]

        for i in range(1, len(s_values)):
            if len(selected_points) >= num_points:
                break
            if s_values[i] - last_s >= min_spacing:
                selected_points.append(i)
                last_s = s_values[i]

        if len(selected_points) < num_points:
            print(f"警告：无法找到足够的生成点来放置{num_points}辆车，最多可放置{len(selected_points)}辆车")
        return selected_points[:num_points]

    def create_vehicle(self, reference_xy, s_values, position_index, velocity, length, width, lane, **attributes):
        """创建单个车辆对象"""
        if reference_xy is None or position_index >= len(reference_xy) - 2:
            raise ValueError("参考线数据不可用或索引越界")

        # 获取参考点的 x, y 坐标
        ref_point = reference_xy[position_index]
        x, y = ref_point[0], ref_point[1]

        # 计算车辆朝向（heading），基于参考线前后点的斜率
        heading = np.arctan2(
            reference_xy[position_index + 1][1] - reference_xy[position_index - 1][1],
            reference_xy[position_index + 1][0] - reference_xy[position_index - 1][0]
        )

        # 设置速度、加速度和偏航率
        v = velocity  # 速度大小
        a = 0.0       # 初始加速度设为0
        yaw_rate = 0.0  # 初始偏航率设为0

        # 创建并返回 Vehicle 对象
        return Vehicle(
            x=x,
            y=y,
            v=v,
            a=a,
            heading=heading,
            yaw_rate=yaw_rate,
            length=length,
            width=width,
            **attributes
        )

    def get_simulation_init_states(self, ego_config, env_vehicles_configs=None):
        """抽象方法，子类需实现"""
        raise NotImplementedError("子类必须实现此方法")

class SingleEgoMergeInitializer(SimulationInitializer):
    """单车智能汇入模式：主车在辅道，环境车辆在主道"""
    def get_simulation_init_states(self, ego_config, env_vehicles_configs=None):
        # 创建主车（在辅道上）
        ego_vehicle = None
        if ego_config:
            pos_idx = ego_config.get('position_index', 0)
            lane = ego_config.get('lane', 0)
            velocity = ego_config.get('velocity', 1.0)
            length = ego_config.get('length', 5.0)
            width = ego_config.get('width', 2.0)
            attributes = ego_config.get('attributes', {'isego': True})
            ego_vehicle = self.create_vehicle(
                self.aux_reference_xy, self.aux_s_values, pos_idx, velocity, length, width, lane, **attributes
            )

        # 创建环境车辆（在主道上）
        env_vehicles = []
        if isinstance(env_vehicles_configs, dict):
            num_vehicles = env_vehicles_configs.get('num_vehicles', 0)
            velocity_range = env_vehicles_configs.get('velocity_range', (0, 30))
            length_range = env_vehicles_configs.get('length_range', (4.0, 5.5))
            width_range = env_vehicles_configs.get('width_range', (2, 2.1))
            common_attributes = env_vehicles_configs.get('attributes', {'isego': False})
            spawn_points = self.select_spawn_points(self.reference_xy, self.s_values, num_vehicles, self.min_point_spacing)

            for pos_idx in spawn_points:
                lane = np.random.randint(0, self.n_lanes)
                velocity = np.random.uniform(velocity_range[0], velocity_range[1])
                length = np.random.uniform(length_range[0], length_range[1])
                width = np.random.uniform(width_range[0], width_range[1])
                vehicle = self.create_vehicle(
                    self.reference_xy, self.s_values, pos_idx, velocity, length, width, lane, **common_attributes
                )
                env_vehicles.append(vehicle)

        return ego_vehicle, env_vehicles

class MultiEgoMergeInitializer(SimulationInitializer):
    """多车智能汇入模式：多辆主车在辅道，环境车辆在主道"""
    def get_simulation_init_states(self, multi_ego_configs, env_vehicles_configs=None):
        # 创建多辆主车（在辅道上）
        ego_vehicles = []
        if isinstance(multi_ego_configs, list):
            spawn_points = self.select_spawn_points(self.aux_reference_xy, self.aux_s_values, len(multi_ego_configs), self.min_point_spacing)
            for i, config in enumerate(multi_ego_configs):
                pos_idx = spawn_points[i] if i < len(spawn_points) else config.get('position_index', 0)
                lane = config.get('lane', 0)
                velocity = config.get('velocity', 1.0)
                length = config.get('length', 5.0)
                width = config.get('width', 2.0)
                attributes = config.get('attributes', {'isego': True})
                vehicle = self.create_vehicle(
                    self.aux_reference_xy, self.aux_s_values, pos_idx, velocity, length, width, lane, **attributes
                )
                ego_vehicles.append(vehicle)

        # 创建环境车辆（在主道上）
        env_vehicles = []
        if isinstance(env_vehicles_configs, dict):
            num_vehicles = env_vehicles_configs.get('num_vehicles', 0)
            velocity_range = env_vehicles_configs.get('velocity_range', (0, 30))
            length_range = env_vehicles_configs.get('length_range', (4.0, 5.5))
            width_range = env_vehicles_configs.get('width_range', (2, 2.1))
            common_attributes = env_vehicles_configs.get('attributes', {'isego': False})
            spawn_points = self.select_spawn_points(self.reference_xy, self.s_values, num_vehicles, self.min_point_spacing)

            for pos_idx in spawn_points:
                lane = np.random.randint(0, self.n_lanes)
                velocity = np.random.uniform(velocity_range[0], velocity_range[1])
                length = np.random.uniform(length_range[0], length_range[1])
                width = np.random.uniform(width_range[0], width_range[1])
                vehicle = self.create_vehicle(
                    self.reference_xy, self.s_values, pos_idx, velocity, length, width, lane, **common_attributes
                )
                env_vehicles.append(vehicle)

        return ego_vehicles, env_vehicles

class RandomMergeInitializer(SimulationInitializer):
    """随机汇流汇入模式：主车在主道和辅道，环境车辆在主道"""
    def get_simulation_init_states(self, ego_config, env_vehicles_configs=None):
        # 创建主车（主道和辅道）
        ego_vehicles = []
        if isinstance(ego_config, dict):
            if 'main' in ego_config:
                for config in ego_config['main']:
                    pos_idx = config.get('position_index', 0)
                    lane = config.get('lane', 0)
                    velocity = config.get('velocity', 1.0)
                    length = config.get('length', 5.0)
                    width = config.get('width', 2.0)
                    attributes = config.get('attributes', {'isego': True})
                    vehicle = self.create_vehicle(
                        self.reference_xy, self.s_values, pos_idx, velocity, length, width, lane, **attributes
                    )
                    ego_vehicles.append(vehicle)
            if 'aux' in ego_config:
                for config in ego_config['aux']:
                    pos_idx = config.get('position_index', 0)
                    lane = config.get('lane', 0)
                    velocity = config.get('velocity', 1.0)
                    length = config.get('length', 5.0)
                    width = config.get('width', 2.0)
                    attributes = config.get('attributes', {'isego': True})
                    vehicle = self.create_vehicle(
                        self.aux_reference_xy, self.aux_s_values, pos_idx, velocity, length, width, lane, **attributes
                    )
                    ego_vehicles.append(vehicle)

        # 创建环境车辆（在主道上）
        env_vehicles = []
        if isinstance(env_vehicles_configs, dict):
            num_vehicles = env_vehicles_configs.get('num_vehicles', 0)
            velocity_range = env_vehicles_configs.get('velocity_range', (0, 30))
            length_range = env_vehicles_configs.get('length_range', (4.0, 5.5))
            width_range = env_vehicles_configs.get('width_range', (2, 2.1))
            common_attributes = env_vehicles_configs.get('attributes', {'isego': False})
            spawn_points = self.select_spawn_points(self.reference_xy, self.s_values, num_vehicles, self.min_point_spacing)

            for pos_idx in spawn_points:
                lane = np.random.randint(0, self.n_lanes)
                velocity = np.random.uniform(velocity_range[0], velocity_range[1])
                length = np.random.uniform(length_range[0], length_range[1])
                width = np.random.uniform(width_range[0], width_range[1])
                vehicle = self.create_vehicle(
                    self.reference_xy, self.s_values, pos_idx, velocity, length, width, lane, **common_attributes
                )
                env_vehicles.append(vehicle)

        return ego_vehicles, env_vehicles