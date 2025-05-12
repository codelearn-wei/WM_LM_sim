
#! 用于交通环境的搭建和测试，接口和Merge_env中一样
import pygame
import numpy as np
import pickle
from scipy.interpolate import splprep, splev
from LM_env.utils.Vehicle_model import VehicleKinematicsModel , Vehicle
from LM_env.interaction_model.initial import SingleEgoMergeInitializer
from LM_env.interaction_model.strategy import StrategyManager
from LM_env.utils.Frenet_Trans import *
import time


# TODO:仿真器搭建还有以下工作（已完成主体函数框架）：
# 1、StrategyManager类，现在是简单的策略，需要建模汇入场景的环境，实现基于交互的策略。（重点，难点。考虑数据分布，决策价值观，世界模型的建立和群体收益等）

# 2、仿真器的初始化函数，需要根据场景初始化主车和环境车辆的状态，包括位置，速度，航向等。（主要结合数据分布去构建，考虑初始位置生成的合理性）

# 3、在建模前面两个函数模型的时候，需要同时修改Merge_env中的接口，使得仿真器和环境接口一致，方便后续的训练和测试，同时要不断优化仿真器的关键函数。

# 4、考虑变道主车的的行为学习，基于规则或基于强化学习方法实现，探索多车联合决策的可能性。

# 全局常量
WINDOW_WIDTH = 1300
WINDOW_HEIGHT = 200
ROAD_COLOR = (180, 180, 180)      # 道路颜色：浅灰色
BOUNDARY_COLOR = (255, 255, 255)  # 边界颜色：白色
BACKGROUND_COLOR = (30, 30, 30)   # 背景颜色：深灰色

class Simulator:
    """仿真器类，用于管理仿真环境"""
    def __init__(self, static_map_data, strategy, dt=0.1 , window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT ,real_time_mode=False):
        # 初始化地图数据
        self.map_dict = static_map_data['map_dict']
        # 主要参考线
        self.reference_line = np.array(list(zip(
            static_map_data['main_road_avg_trajectory']['x_coords'],
            static_map_data['main_road_avg_trajectory']['y_coords']
        )))
        self._fit_reference_spline()
        self.aux_reference_line = np.array(list(zip(
            static_map_data['aux_reference_lanes']['x_coords'],
            static_map_data['aux_reference_lanes']['y_coords']
        )))
        
        # 主道参考线
        self.reference_xy = self.reference_line[::20]  # 每20个点取一个，减少点数
        xy2Frenet = Frenet_trans(self.reference_xy)
               
        # 初始化 Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Traffic Simulator - Ramp Merge")


        # 计算坐标范围和缩放比例
        all_points = np.vstack((self.map_dict['upper_boundary'], self.map_dict['main_lower_boundary'], self.reference_line))
        if 'auxiliary_dotted_line' in self.map_dict and len(self.map_dict['auxiliary_dotted_line']) > 0:
            all_points = np.vstack((all_points, self.map_dict['auxiliary_dotted_line']))
        self.min_x, self.min_y = np.min(all_points, axis=0)
        self.max_x, self.max_y = np.max(all_points, axis=0)
        padding = 1.0
        self.min_x -= padding
        self.max_x += padding
        self.min_y -= padding
        self.max_y += padding
        range_x = self.max_x - self.min_x
        range_y = self.max_y - self.min_y
        self.scale = min(window_width / range_x, window_height / range_y)
        self.scale_x = self.scale
        self.scale_y = self.scale
        self._prepare_pixel_points()
        
        # 仿真状态
        self.paused = False
        self.current_simulation_frame = 0
        self.simulation_speed = 1
        self.dt = 0.1
        self.min_frame = 0
        self.max_frame = 0
        self.font = pygame.font.SysFont('Arial', 16)
        
        # 时间模式设置
        self.real_time_mode = real_time_mode
        self.last_step_time = 0  # 上次步进的时间
        self.time_accumulator = 0  # 累积时间

        # 车辆管理
        self.vehicles = {}          # 存储车辆，键为ID
        self.next_vehicle_id = 0    # 车辆ID计数器
        self.strategy = strategy    # 策略函数
        self.dt = dt                # 时间步长
        self.current_frame = 0      # 当前帧
        self.vehicle_model = VehicleKinematicsModel() # 车辆运动学模型
        
         
        # 初始化 Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Merge Traffic Simulator")
    
    
    def reset(self, ego_init_state, env_vehicles_init_states=None):
        
        # 清空现有车辆
        self.vehicles = {}
        self.next_vehicle_id = 0
        self.current_frame = 0
        self.current_simulation_frame = 0
        
        # 添加主车(ego vehicle)
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
        
    def delete_vehicle(self):
        """如果车辆已经到达终点，删除该车辆"""
        to_delete = []  # 用于存储需要删除的车辆键
        # 遍历字典，找出满足条件的车辆
        for vid, vehicle in self.vehicles.items():
            if vehicle.x < 1030:  # 判断条件
                to_delete.append(vid)  # 将满足条件的车辆键添加到列表
        # 根据收集的键逐个删除
        for vid in to_delete:
            del self.vehicles[vid]  # 从字典中删除对应的车辆
    
    def step(self):
        obs = self.distrub_env_obs()
        actions = self.strategy(obs)
        for vid, action in actions.items():
            if vid in self.vehicles:
                vehicle = self.vehicles[vid]
                
                # 解析动作
                acceleration = action[0]  # 纵向加速度
                steering_angle = action[1]  # 前轮转角
                
                # 使用运动学模型更新车辆状态
                self.vehicle_model.update(vehicle, acceleration, steering_angle, self.dt)
        # 更新仿真帧
        self.delete_vehicle()
        self.current_frame += 1
        self.current_simulation_frame += 1
        env_status = self.check_environment_status()
        print(f"Frame: {self.current_frame} | Environment Status: {env_status}")
        return env_status
        
        
    # TODO ：策略函数调用这里获取的状态，用于计算环境中车辆的动作

    def distrub_env_obs(self):
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
            'current_frame': self.current_simulation_frame,
            'active_vehicles': active_vehicles,
            'reference_line': self.smooth_reference_line.tolist()
        }
        return state
    
    # 仿真平台的可视化绘制
    def draw(self):
        """绘制地图和动态元素，区分主车和环境车辆，并显示车辆ID"""
        # 绘制背景和道路
        self.screen.fill(BACKGROUND_COLOR)
        pygame.draw.polygon(self.screen, ROAD_COLOR, self.road_pixel_points)
        pygame.draw.lines(self.screen, BOUNDARY_COLOR, False, self.upper_pixel_points, 3)
        pygame.draw.lines(self.screen, BOUNDARY_COLOR, False, self.lower_pixel_points, 3)
        
        # 绘制参考轨迹
        pygame.draw.lines(self.screen, (0, 0, 0), False, self.reference_pixel_points, 1)
        pygame.draw.lines(self.screen, (0, 0, 0), False, self.aux_reference_pixel_points, 1)
        
        # 绘制虚线
        if self.auxiliary_dotted_line_pixel_points:
            for i in range(0, len(self.auxiliary_dotted_line_pixel_points) - 1, 2):
                start_pos = self.auxiliary_dotted_line_pixel_points[i]
                end_pos = self.auxiliary_dotted_line_pixel_points[i + 1]
                pygame.draw.line(self.screen, BOUNDARY_COLOR, start_pos, end_pos, 2)
        
        # 绘制车辆
        for vid, vehicle in self.vehicles.items():
            # 获取车辆的中心位置和航向角
            center_x = vehicle.x
            center_y = vehicle.y
            heading = vehicle.heading
            length = vehicle.length
            width = vehicle.width

            # 计算局部坐标系中的四个顶点（未旋转）
            half_length = length / 2
            half_width = width / 2
            local_points = [
                [-half_length, -half_width],  # 左下角
                [-half_length, half_width],   # 左上角
                [half_length, half_width],    # 右上角
                [half_length, -half_width]    # 右下角
            ]

            # 应用旋转矩阵
            cos_theta = np.cos(heading)
            sin_theta = np.sin(heading)
            rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

            # 旋转并平移到全局坐标
            global_points = []
            for local_x, local_y in local_points:
                rotated_point = np.dot(rotation_matrix, [local_x, local_y])
                global_x = center_x + rotated_point[0]
                global_y = center_y + rotated_point[1]
                global_points.append([global_x, global_y])

            # 转换为屏幕像素坐标
            pixel_points = [self.map_to_pixel(x, y) for x, y in global_points]

            # 根据车辆类型使用不同颜色
            if hasattr(self, 'ego_vehicle_id') and vid == self.ego_vehicle_id:
                # 主车用蓝色表示
                color = (0, 0, 255)  # 蓝色
            else:
                # 环境车辆用绿色表示
                color = (0, 255, 0)  # 绿色

            # 绘制车辆多边形
            pygame.draw.polygon(self.screen, color, pixel_points)
            
            # 为主车添加方向指示线（可选）
            if hasattr(self, 'ego_vehicle_id') and vid == self.ego_vehicle_id:
                front_center = [center_x + np.cos(heading) * half_length, 
                            center_y + np.sin(heading) * half_length]
                direction_end = [front_center[0] + np.cos(heading) * (length / 2), 
                                front_center[1] + np.sin(heading) * (length / 2)]
                pygame.draw.line(self.screen, (255, 255, 0), 
                            self.map_to_pixel(front_center[0], front_center[1]),
                            self.map_to_pixel(direction_end[0], direction_end[1]), 2)

            # 在车辆上绘制ID文本
            # 选择在车辆顶部或中心绘制ID
            text_color = (255, 255, 255)  # 白色文字
            text_surface = self.font.render(str(vid), True, text_color)
            
            # 计算文字绘制位置（车辆中心）
            text_center_x, text_center_y = np.mean(pixel_points, axis=0)
            text_rect = text_surface.get_rect(center=(text_center_x, text_center_y))
            
            # 绘制文字
            self.screen.blit(text_surface, text_rect)

        # 绘制信息文字
        info_text = f"Frame: {self.current_simulation_frame} | Active Vehicles: {len(self.vehicles)} | Speed: {self.simulation_speed}x"
        info_surface = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(info_surface, (10, 10))
        
        # 如果存在主车，显示主车信息
        if hasattr(self, 'ego_vehicle_id') and self.ego_vehicle_id in self.vehicles:
            ego_vehicle = self.vehicles[self.ego_vehicle_id]
            ego_speed = np.linalg.norm(ego_vehicle.v)
            ego_heading_deg = np.degrees(ego_vehicle.heading) % 360
            ego_info = f"Ego Vehicle | Speed: {ego_speed:.2f} m/s | Heading: {ego_heading_deg:.1f}°"
            ego_surface = self.font.render(ego_info, True, (0, 255, 255))
            self.screen.blit(ego_surface, (10, 30))
            
    def toggle_time_mode(self):
        """切换实时模式和固定帧率模式"""
        self.real_time_mode = not self.real_time_mode
        self.last_step_time = time.time()  # 重置时间计数器
        self.time_accumulator = 0        

    def run(self):
        """主仿真循环"""
        running = True
        clock = pygame.time.Clock()
        self.last_step_time = time.time()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_s and self.paused:
                        self.step()
                    elif event.key == pygame.K_UP:
                        self.simulation_speed = min(10, self.simulation_speed + 1)
                    elif event.key == pygame.K_DOWN:
                        self.simulation_speed = max(1, self.simulation_speed - 1)
                    elif event.key == pygame.K_r:
                        self.current_simulation_frame = self.min_frame
                    elif event.key == pygame.K_t:
                        self.toggle_time_mode()
            
            if not self.paused:
                current_time = time.time()
                elapsed_time = current_time - self.last_step_time
                self.last_step_time = current_time
                
                if self.real_time_mode:
                    # 真实时间模式：积累时间直到达到步长要求
                    self.time_accumulator += elapsed_time * self.simulation_speed
                    
                    # 当累积时间达到步长时执行步进
                    while self.time_accumulator >= self.dt:
                        self.step()
                        self.time_accumulator -= self.dt
                else:
                    # 固定帧率模式：按照模拟速度执行指定数量的步进
                    for _ in range(self.simulation_speed):
                        self.step()
            
            # 绘制场景
            self.draw()
            pygame.display.flip()
            
            # 控制帧率
            clock.tick(60)
        
        pygame.quit()
        
    #############################################
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
                    distance = np.linalg.norm(vehicle1.x - vehicle2.x)
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
                distance = np.linalg.norm(ego_vehicle.x - vehicle.x)
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
                
            position = [vehicle.x , vehicle.y]
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
        position = [ego_vehicle.x , ego_vehicle.y]
        distance_to_ref = self._get_distance_to_reference_line(position)
        
        # 超出参考线一定距离视为超出道路边界
        road_width_threshold = 3.0
        return distance_to_ref > road_width_threshold

    # 添加一个综合检测函数，在仿真中监测环境状态
    def check_environment_status(self):
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
            "env_collision_detected": env_collision_status,
            "env_colliding_pairs": env_collision_pairs,
            
            # 主车碰撞状态
            "ego_collision_detected": ego_collision_status,
            "ego_colliding_with": ego_collision_vehicles,
            
            # 环境车辆超出道路边界状态
            "env_off_road_detected": env_off_road_status, 
            "env_off_road_ids": env_off_road_ids,
            
            # 主车超出道路边界状态
            "ego_off_road_detected": ego_off_road_status
        }
        
        return status
          
    def _fit_reference_spline(self):
        """拟合参考线的样条曲线"""
        x = self.reference_line[:, 0]
        y = self.reference_line[:, 1]
        tck, _ = splprep([x, y], s=0)
        u_fine = np.linspace(0, 1, 1000)
        self.smooth_reference_line = np.array(splev(u_fine, tck)).T
 
        
    def _prepare_pixel_points(self):
        """预计算地图元素的像素坐标"""
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
        """将地图坐标转换为像素坐标"""
        pixel_x = (x - self.min_x) * self.scale_x
        pixel_y = WINDOW_HEIGHT - (y - self.min_y) * self.scale_y
        return int(pixel_x), int(pixel_y)

def main():
    """测试仿真器功能"""
    print("正在加载地图数据...")
    try:
        file_path = "LM_env/LM_map/LM_static_map.pkl"
        with open(file_path, 'rb') as file:
            static_map = pickle.load(file)
        print(f"成功加载地图数据: {file_path}")
    except Exception as e:
        print(f"加载地图数据失败: {e}")
        return

    # 1、 主车生成策略函数
    strategy_manager = StrategyManager()
    strategy_func = strategy_manager.get_strategy('interactive')  # 获取简单的策略

    
    # 2、 创建仿真器
    simulator = Simulator(static_map[0], strategy_func, dt=0.1, real_time_mode=True)
    
    # 3、 创建仿真器初始化函数
    ego_config = {
    'position_index': 10,
    'velocity': 5.0,
    'length': 4.0,
    'width': 2.0,
    'lane': 1,
    'attributes': {'is_ego': True}
    }

    env_vehicles_configs = {
        'num_vehicles': 8,
        'velocity_range': (5, 6),
        'length_range': (4.5, 5),
        'width_range': (1.6, 2.0),
        'vehicle_spacing': 1.0,  # 数字越大表示生成车辆越稀疏
        'attributes': {'is_ego': False}
    }
    
    #! 初始化的方式需要贴合实际情况
    initializer = SingleEgoMergeInitializer(static_map[0])
    # 配置
    
    ego_init_state, env_vehicles_init_states = initializer.get_simulation_init_states(ego_config,env_vehicles_configs)
    simulator.reset(ego_init_state, env_vehicles_init_states)
      
    # 4、 启动仿真
    print("启动模拟器...")
    print("控制指南:")
    print("- 空格键: 暂停/继续")
    print("- S键: 在暂停时单步执行")
    print("- 上箭头: 增加速度")
    print("- 下箭头: 减少速度")
    print("- R键: 重置到起始帧")
    simulator.run()
    print("模拟结束")
    
if __name__ == "__main__":
    main()