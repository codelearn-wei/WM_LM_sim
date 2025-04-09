
#! 搭建merge的强化学习训练环境
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pickle
from scipy.interpolate import splprep, splev
from LM_env.interaction_model.vehicle_model import VehicleKinematicsModel , Vehicle
from LM_env.interaction_model.Initializer import SimulationInitializer
from LM_env.interaction_model.strategy import StrategyManager

# TODO:仿真器搭建还有以下工作（已完成主体函数框架）：
# 1、StrategyManager类，现在是简单的IDM策略，需要建模汇入场景的环境，实现基于交互的策略。（重点，难点。考虑数据分布，决策价值观，世界模型的建立和群体收益等）

# 2、仿真器的初始化函数，需要根据场景初始化主车和环境车辆的状态，包括位置，速度，航向等。（主要结合数据分布去构建，考虑初始位置生成的合理性）

# 3、在建模前面两个函数模型的时候，需要同时修改Merge_env中的接口，使得仿真器和环境接口一致，方便后续的训练和测试，同时要不断优化仿真器的关键函数。

# 4、考虑变道主车的的行为学习，基于规则或基于强化学习方法实现，探索多车联合决策的可能性。


# Global constants
WINDOW_WIDTH = 1300
WINDOW_HEIGHT = 200
ROAD_COLOR = (180, 180, 180)      # Road color: light gray
BOUNDARY_COLOR = (255, 255, 255)  # Boundary color: white
BACKGROUND_COLOR = (30, 30, 30)   # Background color: dark gray


class MergeEnv(gym.Env):
    """
    Traffic merge environment for reinforcement learning using Gymnasium interface
    """
    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 60,
    }
    
    def __init__(self, map_path="LM_map/LM_static_map.pkl", render_mode=None, dt=0.1, 
                 max_episode_steps=500, other_vehicle_strategy='interactive'):
        """
        Initialize the merge environment
        
        Args:
            map_path: Path to the static map data file
            render_mode: 'human' for pygame visualization, 'rgb_array' for numpy array, None for no rendering
            dt: Time step for simulation
            max_episode_steps: Maximum number of steps per episode
            other_vehicle_strategy: Strategy for controlling other vehicles
        """
        super().__init__()
        
        # Load map data
        try:
            with open(map_path, 'rb') as file:
                self.static_map_data = pickle.load(file)[0]  # Assuming first item contains map data
        except Exception as e:
            raise FileNotFoundError(f"Failed to load map data: {e}")
        
        # Set environment parameters
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.current_step = 0
        
        # Initialize map data
        self.map_dict = self.static_map_data['map_dict']
        self.reference_line = np.array(list(zip(
            self.static_map_data['main_road_avg_trajectory']['x_coords'],
            self.static_map_data['main_road_avg_trajectory']['y_coords']
        )))
        self.aux_reference_line = np.array(list(zip(
            self.static_map_data['aux_reference_lanes']['x_coords'],
            self.static_map_data['aux_reference_lanes']['y_coords']
        )))
        self._fit_reference_spline()
        
        # 车辆状态更新和管理
        self.vehicles = {}
        self.next_vehicle_id = 0
        self.ego_vehicle_id = None
        self.vehicle_model = VehicleKinematicsModel()
        
        # TODO：为建立更真实的仿真环境，需要设计车辆的初始化分布和交互模型
        # 环境车辆初始化
        self.initializer = SimulationInitializer(self.static_map_data)
        
        # 车辆参数初始化,需要更复杂的符合驾驶人行为建模的参数
        
        self.ego_config = {
             'position_index': 1000,
                'velocity': 10.0,
                'length': 5.0,
                'width': 2.0,
                'lane': 1,
                'attributes': {'is_ego': True}  
        }

        self.env_vehicles_configs = {
            'num_vehicles': 5,
            'velocity_range': (5, 6),
            'length_range': (4.0, 5),
            'width_range': (1.8, 2.2),
            'vehicle_spacing': 1.0,  # 数字越大表示生成车辆越稀疏
            'attributes': {'is_ego': False}
        }
        self.strategy_manager = StrategyManager()
        self.strategy_func = self.strategy_manager.get_strategy(other_vehicle_strategy)
        
        
        # TODO：后续可能需要考虑更改动作空间维度
        # Action: [acceleration, steering_angle]
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),  # Min acceleration and steering angle
            high=np.array([1, 1]),   # Max acceleration and steering angle
            dtype=np.float32
        )
        
        # TODO:后续可能需要考虑更改观测空间维度
        max_vehicles_observed = 5  # Maximum number of surrounding vehicles to observe
        obs_dim = 4 + max_vehicles_observed * 4  # 4 for ego vehicle + 4 features per surrounding vehicle
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # For rendering
        if render_mode is not None:
            self._setup_rendering()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for environment reset
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        
        # Clear existing vehicles
        self.vehicles = {}
        self.next_vehicle_id = 0
        
        # Get initial states from initializer
        ego_init_state, env_vehicles_init_states = self.initializer.get_simulation_init_states(self.ego_config ,self.env_vehicles_configs)
        
        # Add ego vehicle
        # TODO:如何生成多辆主车？？？
        if ego_init_state:
            ego_id = self.next_vehicle_id
            self.vehicles[ego_id] = Vehicle(
                    position = ego_init_state.position,
                    velocity = ego_init_state.velocity,
                    heading = ego_init_state.heading,
                    length = ego_init_state.length,
                    width = ego_init_state.width,
                    attributes = ego_init_state.attributes    
            )
            self.ego_vehicle_id = ego_id
            self.next_vehicle_id += 1
        
        # Add environment vehicles
        if env_vehicles_init_states:
            for env_vehicle_state in env_vehicles_init_states:
                env_id = self.next_vehicle_id
                self.vehicles[env_id] = Vehicle(
                        position = env_vehicle_state.position,
                        velocity = env_vehicle_state.velocity,
                        heading = env_vehicle_state.heading,
                        length = env_vehicle_state.length,
                        width = env_vehicle_state.width,
                        attributes = env_vehicle_state.attributes           
                )
                self.next_vehicle_id += 1
        
        # Calculate initial observation
        observation = self._get_observation()
        info = {}
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def delete_env_vehicle(self):
        to_delete = []
        for vid, vehicle in self.vehicles.items():
            if vid != self.ego_vehicle_id and vehicle.position[0] < 1030:
                to_delete.append(vid)
        for vid in to_delete:
            del self.vehicles[vid]
    
    def step(self, ego_action):
        self.current_step += 1
        # 分发环境车辆获得的观测
        obs_for_other_vehicles = self.distrub_env_obs()
        # 利用定义的World_Model选择环境车辆的动作
        env_actions = self.strategy_func(obs_for_other_vehicles)
        for vid, action in env_actions.items():
            if vid in self.vehicles:
                vehicle = self.vehicles[vid]
                acceleration = action[0]
                steering_angle = action[1]
                self.vehicle_model.update(vehicle, acceleration, steering_angle, self.dt)
        self.delete_env_vehicle()  # 删除已经到达终点的车辆
        
        # 默认返回值
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)  # 填充零值
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # 如果有主车，更新其状态并计算观测和奖励
        if self.ego_vehicle_id is not None:
            ego_vehicle = self.vehicles[self.ego_vehicle_id]
            acceleration = float(ego_action[0])
            steering_angle = float(ego_action[1])
            self.vehicle_model.update(ego_vehicle, acceleration, steering_angle, self.dt)
            
            # 获得主车观测
            observation = self._get_observation()
            # 计算主车奖励
            reward = self._calculate_reward()
            # 检查终止条件
            terminated = self._check_termination()
            
            # 回合阶段条件
            truncated = self.current_step >= self.max_episode_steps
            # 额外信息汇总
            info = {
                'current_step': self.current_step,
                'ego_position': ego_vehicle.position.tolist(),
                'ego_velocity': ego_vehicle.velocity.tolist(),
                'ego_heading': ego_vehicle.heading
            }
            
            # 监测环境状态
            env_status = self._check_environment_status()
            # print(f" Environment Status: {env_status}")
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            If render_mode is 'rgb_array', returns a numpy array of the rendered frame.
            Otherwise, returns None.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def close(self):
        """Clean up resources"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
    

    def distrub_env_obs(self):
        """获取当前仿真状态，供外部算法使用，包含周围车辆的相对信息"""
        active_vehicles = {}
        distance_threshold = 50.0  # 周围车辆的距离阈值（米）

        # 遍历所有车辆
        for vid, vehicle in self.vehicles.items():
            # 基本信息
            vehicle_info = {
                'position': vehicle.position.tolist(),
                'velocity': vehicle.velocity.tolist(),
                'acceleration': vehicle.acceleration.tolist(),
                'heading': vehicle.heading , # 航向角
                 'is_ego': vehicle.attributes.get('isego', False)
            }

            # 计算周围车辆的相对信息
            neighbors = []
            for other_vid, other_vehicle in self.vehicles.items():
                if vid == other_vid:  # 跳过自身
                    continue
                # 计算两车距离
                distance = np.linalg.norm(vehicle.position - other_vehicle.position)
                if distance < distance_threshold:
                    # 绝对位置
                    position = other_vehicle.position.tolist()
                    # 相对位置
                    relative_position = (other_vehicle.position - vehicle.position).tolist()
                    # 绝对速度
                    velocity = other_vehicle.velocity.tolist()                 
                    # 相对速度
                    relative_velocity = (other_vehicle.velocity - vehicle.velocity).tolist()                  
                    # 绝对航向角
                    heading = other_vehicle.heading
                    # 相对航向角
                    relative_heading = other_vehicle.heading - vehicle.heading
                    # 规范化到 [-π, π]
                    relative_heading = (relative_heading + np.pi) % (2 * np.pi) - np.pi
                    # 车长
                    length = other_vehicle.length
                    # 车宽
                    width = other_vehicle.width
                    # 是否是主车
                    is_ego = other_vehicle.attributes.get('isego', False)

                    neighbors.append({
                        'vehicle_id': other_vid,
                        'position': position,
                        'heading': heading,
                        'velocity': velocity,
                        'relative_position': relative_position,
                        'relative_velocity': relative_velocity,
                        'relative_heading': relative_heading,
                        'length': length,
                        'withd': width,
                        'is_ego': is_ego
                    })

            # 添加邻居信息
            vehicle_info['neighbors'] = neighbors
            active_vehicles[vid] = vehicle_info

        # 返回仿真状态
        state = {
            'active_vehicles': active_vehicles,
            'reference_line': self.smooth_reference_line.tolist()
        }
        return state
    
    def _get_observation(self):
        """
        Convert the current state into an observation for the RL agent
        """
        # Initialize empty observation
        obs = []
        
        # Check if ego vehicle exists
        if hasattr(self, 'ego_vehicle_id') and self.ego_vehicle_id in self.vehicles:
            ego_vehicle = self.vehicles[self.ego_vehicle_id]
            ego_x, ego_y = ego_vehicle.position
            ego_heading = ego_vehicle.heading
            ego_speed = np.linalg.norm(ego_vehicle.velocity)
            
            # Basic ego vehicle features
            obs = [ego_x, ego_y, ego_heading, ego_speed]
            
            # Get surrounding vehicles' relative positions and velocities
            surrounding_vehicles = []
            for vid, vehicle in self.vehicles.items():
                if vid != self.ego_vehicle_id:
                    # Calculate relative position and velocity
                    rel_pos_x = vehicle.position[0] - ego_vehicle.position[0]
                    rel_pos_y = vehicle.position[1] - ego_vehicle.position[1]
                    rel_vel_x = vehicle.velocity[0] - ego_vehicle.velocity[0]
                    rel_vel_y = vehicle.velocity[1] - ego_vehicle.velocity[1]
                    
                    # Calculate distance
                    distance = np.sqrt(rel_pos_x**2 + rel_pos_y**2)
                    
                    surrounding_vehicles.append((distance, rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y))
            
            # Sort by distance and take closest vehicles
            surrounding_vehicles.sort()
            max_vehicles_observed = (self.observation_space.shape[0] - 4) // 4
            
            # Add surrounding vehicles to observation
            for i in range(max_vehicles_observed):
                if i < len(surrounding_vehicles):
                    # Add relative x, y position and vx, vy velocity
                    obs.extend([surrounding_vehicles[i][1], surrounding_vehicles[i][2],
                            surrounding_vehicles[i][3], surrounding_vehicles[i][4]])
                else:
                    # Pad with zeros if we have fewer vehicles than the maximum
                    obs.extend([0.0, 0.0, 0.0, 0.0])
        else:
            # If there's no ego vehicle, return a zero-filled observation
            obs_size = self.observation_space.shape[0]
            obs = [0.0] * obs_size
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self):
        """
        Calculate reward based on current state
        """
        reward = 0
        return reward
        
    
    # 检测终止条件
    def _check_termination(self):
        """Check if episode should terminate (collision or off-road)"""
        ego_is_collision , _ = self._check_ego_collision()
        ego_off_road = self._check_ego_off_road()
        ego_reach_end = self._check_reach_end()
        return ego_is_collision or ego_reach_end or ego_off_road
    
    
    # 检测主车是否到达终点
    def _check_reach_end(self):
        """Check if ego vehicle has reached the end of the road"""
        if self.ego_vehicle_id is not None:
            ego_vehicle = self.vehicles[self.ego_vehicle_id]
            # 设计目标终点（应该是一个x，y的范围）
            if 1032 < ego_vehicle.position[0] < 1035 & 0 < ego_vehicle.position[1] < 2:
                return True
        return False
    

    ##############################################
    #############碰撞检测与环境监测################
    #############################################
    
    def _get_distance_to_reference_line(self, position):
        """Calculate distance from a position to the reference line"""
        # Calculate distances to all points on the reference line
        distances = np.sqrt(np.sum((self.smooth_reference_line - position) ** 2, axis=1))
        # Return the minimum distance
        return np.min(distances) 
    
    # 检查主车是否与他车发生碰撞
    def _check_env_collision(self):
        """
        检查环境车辆之间是否发生碰撞，并返回发生碰撞的车辆ID对
        
        Returns:
            tuple: (bool, list) - 第一个元素表示是否发生碰撞，第二个元素是发生碰撞的车辆ID对列表
        """
        collision_occurred = False
        colliding_pairs = []
        
        # 检查所有车辆对
        for vid1, vehicle1 in self.vehicles.items():
            for vid2, vehicle2 in self.vehicles.items():
                if vid1 != vid2 and vid1 < vid2:  # 避免重复检查相同的车辆对
                    distance = np.linalg.norm(vehicle1.position - vehicle2.position)
                    # 使用车辆长度和宽度的平均值作为碰撞阈值
                    collision_threshold = (vehicle1.length + vehicle2.length + vehicle1.width + vehicle2.width) / 4.0
                    
                    if distance < collision_threshold:
                        collision_occurred = True
                        colliding_pairs.append((vid1, vid2))
        
        return collision_occurred, colliding_pairs

    def _check_ego_collision(self):
        """
        检查主车是否与环境车辆发生碰撞，并返回与主车发生碰撞的环境车辆ID
        
        Returns:
            tuple: (bool, list) - 第一个元素表示主车是否发生碰撞，第二个元素是与主车发生碰撞的环境车辆ID列表
        """
        if not hasattr(self, 'ego_vehicle_id') or self.ego_vehicle_id not in self.vehicles:
            return False, []
            
        ego_collision = False
        colliding_with_ego = []
        
        ego_vehicle = self.vehicles[self.ego_vehicle_id]
        
        for vid, vehicle in self.vehicles.items():
            if vid != self.ego_vehicle_id:
                distance = np.linalg.norm(ego_vehicle.position - vehicle.position)
                # 使用车辆长度和宽度的平均值作为碰撞阈值
                collision_threshold = (ego_vehicle.length + vehicle.length + ego_vehicle.width + vehicle.width) / 4.0
                
                if distance < collision_threshold:
                    ego_collision = True
                    colliding_with_ego.append(vid)
        
        return ego_collision, colliding_with_ego

    def _check_env_off_road(self):
        """
        检查环境车辆是否超出道路边界约束，并返回超出边界的车辆ID
        
        Returns:
            tuple: (bool, list) - 第一个元素表示是否有车辆超出边界，第二个元素是超出边界的车辆ID列表
        """
        off_road_occurred = False
        off_road_vehicles = []
        
        for vid, vehicle in self.vehicles.items():
            # 跳过主车的检测，主车会在_check_ego_off_road中检测
            if hasattr(self, 'ego_vehicle_id') and vid == self.ego_vehicle_id:
                continue
                
            position = vehicle.position
            distance_to_ref = self._get_distance_to_reference_line(position)
            
            # 超出参考线一定距离视为超出道路边界
            road_width_threshold = 3.0
            if distance_to_ref > road_width_threshold:
                off_road_occurred = True
                off_road_vehicles.append(vid)
        
        return off_road_occurred, off_road_vehicles

    def _check_ego_off_road(self):
        """
        检查主车是否超出道路边界约束
        
        Returns:
            bool: 主车是否超出道路边界
        """
        if not hasattr(self, 'ego_vehicle_id') or self.ego_vehicle_id not in self.vehicles:
            return False
            
        ego_vehicle = self.vehicles[self.ego_vehicle_id]
        position = ego_vehicle.position
        distance_to_ref = self._get_distance_to_reference_line(position)
        
        # 超出参考线一定距离视为超出道路边界
        road_width_threshold = 3.0
        return distance_to_ref > road_width_threshold

    # 添加一个综合检测函数，在仿真中监测环境状态
    def _check_environment_status(self):
        """
        综合检测环境状态，包括车辆碰撞和道路边界情况，同时包含主车的状态
        
        Returns:
            dict: 包含环境状态的字典，包括碰撞和超出边界信息
        """
        # 检测环境车辆之间的碰撞
        env_collision_status, env_collision_pairs = self._check_env_collision()
        
        # 检测主车与环境车辆的碰撞
        ego_collision_status, ego_collision_vehicles = self._check_ego_collision()
        
        # 检测环境车辆是否超出道路边界
        env_off_road_status, env_off_road_ids = self._check_env_off_road()
        
        # 检测主车是否超出道路边界
        ego_off_road_status = self._check_ego_off_road()
        
        status = {
            # 环境车辆碰撞状态
            "env_collision": env_collision_status,
            "env_colliding_pairs": env_collision_pairs,
            
            # 主车碰撞状态
            "ego_collision": ego_collision_status,
            "ego_colliding_with": ego_collision_vehicles,
            
            # 环境车辆超出道路边界状态
            "env_off_road": env_off_road_status, 
            "env_off_road_ids": env_off_road_ids,
            
            # 主车超出道路边界状态
            "ego_off_road": ego_off_road_status
        }
        
        return status
    
    ##############################################
    ##################渲染与绘图###################
    ##############################################
    
    def _fit_reference_spline(self):
        """Fit a spline curve to the reference line"""
        x = self.reference_line[:, 0]
        y = self.reference_line[:, 1]
        tck, _ = splprep([x, y], s=0)
        u_fine = np.linspace(0, 1, 1000)
        self.smooth_reference_line = np.array(splev(u_fine, tck)).T
    
    def _setup_rendering(self):
            """Set up rendering environment"""
            pygame.init()
            
            # Calculate coordinate ranges and scaling
            all_points = np.vstack((self.map_dict['upper_boundary'], self.map_dict['main_lower_boundary'], self.reference_line))
            if 'auxiliary_dotted_line' in self.map_dict and len(self.map_dict['auxiliary_dotted_line']) > 0:
                all_points = np.vstack((all_points, self.map_dict['auxiliary_dotted_line']))
            self.min_x, self.min_y = np.min(all_points, axis=0)
            self.max_x, self.max_y = np.max(all_points, axis=0)
            
            # Add padding
            padding = 1.0
            self.min_x -= padding
            self.max_x += padding
            self.min_y -= padding
            self.max_y += padding
            
            # Calculate scale
            range_x = self.max_x - self.min_x
            range_y = self.max_y - self.min_y
            self.scale = min(WINDOW_WIDTH / range_x, WINDOW_HEIGHT / range_y)
            self.scale_x = self.scale
            self.scale_y = self.scale
            
            # Prepare pixel points for rendering
            self._prepare_pixel_points()
            
            # Initialize pygame screen and clock
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Traffic Merge RL Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 16)
    
    def _prepare_pixel_points(self):
        """Prepare pixel coordinates for map elements"""
        upper_points = self.map_dict['upper_boundary']
        lower_points = self.map_dict['main_lower_boundary'][::-1]
        road_points = np.vstack((upper_points, lower_points))
        
        self.road_pixel_points = [self.map_to_pixel(x, y) for x, y in road_points]
        self.upper_pixel_points = [self.map_to_pixel(x, y) for x, y in self.map_dict['upper_boundary']]
        self.lower_pixel_points = [self.map_to_pixel(x, y) for x, y in self.map_dict['main_lower_boundary']]
        self.reference_pixel_points = [self.map_to_pixel(x, y) for x, y in self.smooth_reference_line]
        self.aux_reference_pixel_points = [self.map_to_pixel(x, y) for x, y in self.aux_reference_line]
        
        if 'auxiliary_dotted_line' in self.map_dict:
            self.auxiliary_dotted_line_pixel_points = [self.map_to_pixel(x, y) for x, y in self.map_dict['auxiliary_dotted_line']]
        else:
            self.auxiliary_dotted_line_pixel_points = []
    
    def map_to_pixel(self, x, y):
        """Convert map coordinates to pixel coordinates"""
        pixel_x = (x - self.min_x) * self.scale_x
        pixel_y = WINDOW_HEIGHT - (y - self.min_y) * self.scale_y
        return int(pixel_x), int(pixel_y)
    
    def _render_frame(self):
        """Render the current state of the environment"""
        if self.render_mode is None:
            return None
            
        if self.screen is None:
            self._setup_rendering()
        
        # Fill background
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw road
        pygame.draw.polygon(self.screen, ROAD_COLOR, self.road_pixel_points)
        pygame.draw.lines(self.screen, BOUNDARY_COLOR, False, self.upper_pixel_points, 3)
        pygame.draw.lines(self.screen, BOUNDARY_COLOR, False, self.lower_pixel_points, 3)
        
        # Draw reference line
        pygame.draw.lines(self.screen, (0, 0, 0), False, self.reference_pixel_points, 1)
        pygame.draw.lines(self.screen, (0, 0, 0), False, self.aux_reference_pixel_points, 1)
        
        # Draw dotted line if exists
        if self.auxiliary_dotted_line_pixel_points:
            for i in range(0, len(self.auxiliary_dotted_line_pixel_points) - 1, 2):
                if i + 1 < len(self.auxiliary_dotted_line_pixel_points):
                    start_pos = self.auxiliary_dotted_line_pixel_points[i]
                    end_pos = self.auxiliary_dotted_line_pixel_points[i + 1]
                    pygame.draw.line(self.screen, BOUNDARY_COLOR, start_pos, end_pos, 2)
        
        # Draw vehicles
        for vid, vehicle in self.vehicles.items():
            # Get vehicle center position and heading
            center_x, center_y = vehicle.position
            heading = vehicle.heading
            length = vehicle.length
            width = vehicle.width
            
            # Calculate local coordinates of corners (unrotated)
            half_length = length / 2
            half_width = width / 2
            local_points = [
                [-half_length, -half_width],  # bottom-left
                [-half_length, half_width],   # top-left
                [half_length, half_width],    # top-right
                [half_length, -half_width]    # bottom-right
            ]
            
            # Apply rotation matrix
            cos_theta = np.cos(heading)
            sin_theta = np.sin(heading)
            rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            
            # Rotate and translate to global coordinates
            global_points = []
            for local_x, local_y in local_points:
                rotated_point = np.dot(rotation_matrix, [local_x, local_y])
                global_x = center_x + rotated_point[0]
                global_y = center_y + rotated_point[1]
                global_points.append([global_x, global_y])
            
            # Convert to pixel coordinates
            pixel_points = [self.map_to_pixel(x, y) for x, y in global_points]
            
            # Draw vehicle with different colors for ego and environment vehicles
            if vid == self.ego_vehicle_id:
                color = (0, 0, 255)  # Blue for ego vehicle
                
                # Draw direction indicator for ego vehicle
                front_center = [center_x + np.cos(heading) * half_length, 
                                center_y + np.sin(heading) * half_length]
                direction_end = [front_center[0] + np.cos(heading) * (length / 2), 
                                front_center[1] + np.sin(heading) * (length / 2)]
                pygame.draw.line(self.screen, (255, 255, 0), 
                                self.map_to_pixel(front_center[0], front_center[1]),
                                self.map_to_pixel(direction_end[0], direction_end[1]), 2)
            else:
                color = (0, 255, 0)  # green for environment vehicles
            
            # Draw vehicle polygon
            pygame.draw.polygon(self.screen, color, pixel_points)
        
        # Draw information text
        info_text = f"Frame: {self.current_step} | Vehicles: {len(self.vehicles)}"
        info_surface = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(info_surface, (10, 10))
        
        # If ego vehicle exists, display its information
        if self.ego_vehicle_id in self.vehicles:
            ego_vehicle = self.vehicles[self.ego_vehicle_id]
            ego_speed = np.linalg.norm(ego_vehicle.velocity)
            ego_heading_deg = np.degrees(ego_vehicle.heading) % 360
            ego_info = f"Ego Speed: {ego_speed:.2f} m/s | Heading: {ego_heading_deg:.1f}°"
            ego_surface = self.font.render(ego_info, True, (0, 255, 255))
            self.screen.blit(ego_surface, (10, 30))
        
        # Update the display
        pygame.display.flip()
        
        # Control the frame rate
        self.clock.tick(self.metadata["render_fps"])
        
        # Return the rendered frame if needed
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

def make_env(map_path="LM_env/LM_map/LM_static_map.pkl", render_mode=None, max_episode_steps=500):
    """Factory function to create the environment with specified parameters"""
    env = MergeEnv(
        map_path=map_path,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps
    )
    return env



if __name__ == "__main__":

    render_mode = 'human' 
    episodes = 100
    max_steps = 500 
    env = make_env(render_mode=render_mode)
    for i in range(episodes):

        
        # 定义reset函数的输入
        # TODO：交通状态（主道车辆的状态）
        # TODO：社会化函数（主道车辆的策略参数）
        # TODO：进一步对主车进行设置
        

        observation, info = env.reset()  
            
        for i in range(max_steps):   
            #TODO： 接入强化学习主车动作选择      
            acceleration = 0
            steering = 0.0                       
            # 动作更新于运行
            action = np.array([acceleration, steering], dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step: {env.current_step}, Reward: {reward:.2f}")
            
            # End episode if terminated or truncated
            if terminated or truncated:
                print("Episode ended")
                break
        
