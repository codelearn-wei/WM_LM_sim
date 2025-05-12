#! 搭建merge的强化学习训练环境
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
from scipy.interpolate import splprep, splev

# 主车道策略交互函数
from LM_env.interaction_model.initial import SingleEgoMergeInitializer
from LM_env.interaction_model.strategy import StrategyManager

# 工具函数
from LM_env.utils.Vehicle_model import VehicleKinematicsModel , Vehicle
from LM_env.utils.Frenet_Trans import *
from LM_env.utils.Render import Renderer
from LM_env.utils.Monitor import Monitor


# TODO:仿真器搭建还有以下工作（已完成主体函数框架）：
# 1、StrategyManager类，现在是简单的IDM策略，需要建模汇入场景的环境，实现基于交互的策略。（重点，难点。考虑数据分布，决策价值观，世界模型的建立和群体收益等）

# 2、仿真器的初始化函数，需要根据场景初始化主车和环境车辆的状态，包括位置，速度，航向等。（主要结合数据分布去构建，考虑初始位置生成的合理性）

# 3、在建模前面两个函数模型的时候，需要同时修改Merge_env中的接口，使得仿真器和环境接口一致，方便后续的训练和测试，同时要不断优化仿真器的关键函数。

# 4、考虑变道主车的的行为学习，基于规则或基于强化学习方法实现，探索多车联合决策的可能性。


def load_map_data(map_path):
    """Load and process map data from a pickle file."""
    try:
        with open(map_path, 'rb') as file:
            static_map_data = pickle.load(file)[0]
        return static_map_data
    except Exception as e:
        raise FileNotFoundError(f"Failed to load map data: {e}")

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
        初始化Merge环境

        参数说明:
            map_path: 静态地图数据的路径
            render_mode: 渲染模式，可选 'human'（pygame窗口）、'rgb_array'（返回图像数组）或 None（不渲染）
            dt: 仿真时间步长
            max_episode_steps: 每个episode的最大步数
            other_vehicle_strategy: 环境中非ego车辆的控制策略
        """
        super().__init__()

        # ======================== 加载静态地图数据 ========================
        self.static_map_data = load_map_data(map_path)

        # ======================== 设置环境基本参数 ========================
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.current_step = 0

        # ======================== 参考线及地图处理 ========================
        self.map_dict = self.static_map_data['map_dict']

        # 主车道平均轨迹参考线（主参考线）
        self.reference_line = np.array(list(zip(
            self.static_map_data['main_road_avg_trajectory']['x_coords'],
            self.static_map_data['main_road_avg_trajectory']['y_coords']
        )))

        # 辅助车道参考线（如匝道）
        self.aux_reference_line = np.array(list(zip(
            self.static_map_data['aux_reference_lanes']['x_coords'],
            self.static_map_data['aux_reference_lanes']['y_coords']
        )))

        # 拟合主参考线用于曲线插值
        self._fit_reference_spline()

        # =========== Frenet坐标系转换（用于坐标系转换） =============
        self.reference_xy = self.reference_line[::20]  # 每20个点取一个，降低计算量
        self.xy2Frenet = Frenet_trans(self.reference_xy)

        # ======================== 车辆管理相关 ========================
        self.vehicles = {}                # 所有车辆对象的字典
        self.next_vehicle_id = 0         # 分配车辆ID的计数器
        self.ego_vehicle_id = None       # ego车辆ID
        self.vehicle_model = VehicleKinematicsModel()  # 车辆动力学模型

        # ego车辆初始化配置
        self.initializer = SingleEgoMergeInitializer(self.static_map_data)
        self.ego_config = {
            'position_index': 50,       # 初始位置在线上的第50个点
            'velocity': 2,              # 初始速度
            'length': 5.0,              # 车长
            'width': 2.0,               # 车宽
            'lane': 2,                  # 初始车道号
            'attributes': {'is_ego': True}
        }

        # 环境其他车辆初始化配置
        # 环境中的车辆数量也需要不一样
        self.env_vehicles_configs = {
            'num_vehicles': 5,                   # 环境中车辆数量
            'velocity_range': (5, 6),            # 速度范围
            'length_range': (4.0, 5.0),          # 车长范围
            'width_range': (1.8, 2.2),           # 车宽范围
            'vehicle_spacing': 1.0,              # 间隔控制因子，越大越稀疏
            'attributes': {'is_ego': False}
        }

        # ======================== 策略管理 ========================
        self.strategy_manager = StrategyManager()
        self.strategy_func = self.strategy_manager.get_strategy(other_vehicle_strategy)

        # ======================== 动作空间定义 ========================
        # 动作为 [加速度, 转角]，范围均为 [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

        # ======================== 观测空间定义 ========================
        # 观测为 [ego_x, ego_y, ego_heading, ego_speed, ...] 
        # 外加最多4辆周围车辆，每辆车用4个特征表示
        max_vehicles_observed = 4
        obs_dim = 4 + max_vehicles_observed * 4

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # ======================== 监控与记录 ========================
        self.monitor = Monitor(self)

        # ======================== 可视化渲染 ========================
        self.renderer = Renderer(self) if render_mode is not None else None
        if self.renderer is not None:
            self.renderer._setup_rendering()

    
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
                    x = ego_init_state.x,
                    y = ego_init_state.y,
                    v = ego_init_state.v,
                    a = ego_init_state.a,
                    yaw = ego_init_state.yaw,
                    heading = ego_init_state.heading,
                    yaw_rate= ego_init_state.yaw_rate,
                    width = ego_init_state.width,
                    length = ego_init_state.length,
                    attributes = ego_init_state.attributes    
            )
            self.ego_vehicle_id = ego_id
            self.next_vehicle_id += 1
        
        # 添加环境车辆
        if env_vehicles_init_states:
            for env_vehicle_state in env_vehicles_init_states:
                env_id = self.next_vehicle_id
                self.vehicles[env_id] = Vehicle(
                    x = env_vehicle_state.x,
                    y = env_vehicle_state.y,
                    v = env_vehicle_state.v,
                    a = env_vehicle_state.a,
                    yaw = env_vehicle_state.yaw,
                    heading = env_vehicle_state.heading,
                    width = env_vehicle_state.width,
                    yaw_rate= env_vehicle_state.yaw_rate,
                    length = env_vehicle_state.length,
                    attributes = env_vehicle_state.attributes          
                )
                self.next_vehicle_id += 1
        
        # Calculate initial observation
        observation = self._get_observation()
        info = {}
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return observation, info
    
    def delete_env_vehicle(self):
        to_delete = []
        for vid, vehicle in self.vehicles.items():
            if vid != self.ego_vehicle_id and vehicle.x < 1030:
                to_delete.append(vid)
        for vid in to_delete:
            del self.vehicles[vid]
    
    def step(self, ego_action):
        self.current_step += 1
        # 分发环境车辆获得的观测
        obs_for_other_vehicles = self._distrub_env_obs()
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
            # 转化到s-l坐标系
            ego_s , ego_l = self.xy2Frenet.trans_sl(ego_vehicle.x, ego_vehicle.y, ego_vehicle.v, ego_vehicle.a, ego_vehicle.yaw)
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
                'ego_x': ego_vehicle.x,
                'ego_y': ego_vehicle.y,
                'ego_velocity': ego_vehicle.v,
                'ego_heading': ego_vehicle.heading
            }
            
            # 监测环境状态
            env_status =self.monitor.check_environment_status()
            print(f" Environment Status: {env_status}")
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.renderer:
            return self.renderer.render_frame()
        return None
    

    def close(self):
        if self.renderer:
            self.renderer.close()
    

    def _distrub_env_obs(self):
        """获取当前仿真状态，供外部算法使用，包含周围车辆的相对信息"""
        active_vehicles = {}
        distance_threshold = 50.0  # 周围车辆的距离阈值（米）

        # 遍历所有车辆
        for vid, vehicle in self.vehicles.items():
            # 本车基本信息
            vehicle_info = {
            'position': [vehicle.x, vehicle.y],
            'velocity': vehicle.v,  # 标量速度
            'acceleration': vehicle.a,
            'heading': vehicle.heading,
            'is_ego': vehicle.attributes['attributes'].get('is_ego', False)
            }

            # 计算周围车辆的相对信息
            neighbors = []
            for other_vid, other_vehicle in self.vehicles.items():
                if vid == other_vid:  # 跳过自身
                    continue
                 # 计算两车中心点距离
                position = np.array([vehicle.x, vehicle.y])
                other_position = np.array([other_vehicle.x, other_vehicle.y])
                distance = np.linalg.norm(position - other_position)
                if distance < distance_threshold:
                    # 相对位置
                    relative_position = (other_position - position).tolist()
                    # 绝对速度（标量）
                    velocity = other_vehicle.v
                    # 相对速度（标量差值，近似处理）
                    relative_velocity = other_vehicle.v - vehicle.v
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
                        'is_ego': vehicle.attributes['attributes'].get('is_ego', False)
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
            ego_x = ego_vehicle.x
            ego_y = ego_vehicle.y
            ego_heading = ego_vehicle.heading
            ego_speed = np.linalg.norm(ego_vehicle.v)
            
            # Basic ego vehicle features
            obs = [ego_x, ego_y, ego_heading, ego_speed]
            
            # Get surrounding vehicles' relative positions and velocities
            surrounding_vehicles = []
            for vid, vehicle in self.vehicles.items():
                if vid != self.ego_vehicle_id:
                    # Calculate relative position and velocity
                    rel_pos_x = vehicle.x - ego_vehicle.x
                    rel_pos_y = vehicle.y - ego_vehicle.y
                    rel_vel = vehicle.v - ego_vehicle.v

                    
                    # Calculate distance
                    distance = np.sqrt(rel_pos_x**2 + rel_pos_y**2)
                    
                    surrounding_vehicles.append((distance, rel_pos_x, rel_pos_y, rel_vel))
            
            # Sort by distance and take closest vehicles
            surrounding_vehicles.sort()
            max_vehicles_observed = (self.observation_space.shape[0] - 4) // 4
            
            # Add surrounding vehicles to observation
            for i in range(max_vehicles_observed):
                if i < len(surrounding_vehicles):
                    # Add relative x, y position and v
                    obs.extend([surrounding_vehicles[i][0], surrounding_vehicles[i][1],
                            surrounding_vehicles[i][2], surrounding_vehicles[i][3]])
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
        ego_is_collision , _ = self.monitor.check_ego_collision()
        ego_off_road = self.monitor.check_ego_off_road()
        ego_reach_end = self.monitor.check_reach_end()
        termination = ego_is_collision or ego_off_road or ego_reach_end
        return termination 
    

    # 拟合参考线
    def _fit_reference_spline(self):
        """Fit a spline curve to the reference line"""
        x = self.reference_line[:, 0]
        y = self.reference_line[:, 1]
        tck, _ = splprep([x, y], s=0)
        u_fine = np.linspace(0, 1, 1000)
        self.smooth_reference_line = np.array(splev(u_fine, tck)).T
    


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