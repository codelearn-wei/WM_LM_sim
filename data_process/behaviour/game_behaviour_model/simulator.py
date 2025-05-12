import pygame
import math
import time
from data_process.behaviour.game_behaviour_model.vehicle import *  # Assuming this imports constants like RED, GREEN, etc.
from config import LANE_CHANGE_PARAMS # 将每一条可能的采样轨迹画出来
from data_process.behaviour.game_behaviour_model.game.game_action import *
pygame.init()
pygame.display.set_caption("车道变道仿真")

class MergingSimulation:
    def __init__(self):
        self.vehicles = {}
        self.time = 0
        self.dt = 0.1
        self.running = True
        self.pause = False
        self.real_time = True
        self.last_update = time.time()
        self.road_bounds = []
        
        # More professional and distinctive color palette
        self.vehicle_colors = {
            'FV': (44, 62, 80),      # Dark Blue-Gray
            'LFV': (39, 174, 96),    # Emerald Green
            'ZV': (52, 152, 219),    # Bright Blue
            'BZV': (241, 196, 15),   # Vivid Yellow
            'LZV': (211, 84, 0)      # Deep Orange
        }
        
        # Road configuration with enhanced lane positioning
        self.road_y = SCREEN_HEIGHT / 2 / SCALE
        self.lane_positions = {
            "main": self.road_y + LANE_WIDTH,   # Main lane center line
            "aux": self.road_y                  # Auxiliary lane center line
        }
        
        self.behavior_planners = {}
    
    def add_vehicle(self, name, x, y, v, a, heading, yaw_rate, length=5, width=2, lane=None):
        """Add a vehicle to the simulation with specified parameters"""
        vehicle = Vehicle(x, y, v, a, heading, yaw_rate, length, width, lane)
        vehicle.name = name
        vehicle.color = self.vehicle_colors.get(name, (128, 128, 128))
        self.vehicles[name] = vehicle
        return vehicle
    
    def add_vehicle_in_lane(self, name, x, lane, v, a=0, heading=0, yaw_rate=0, length=4.8, width=1.8):
        """Add a vehicle positioned in a specific lane"""
        if lane not in self.lane_positions:
            raise ValueError(f"Lane '{lane}' not defined")
        
        y = self.lane_positions[lane]
        return self.add_vehicle(name, x, y, v, a, heading, yaw_rate, length, width, lane)
    
    def register_behavior_planner(self, vehicle_name, planner_function):
        """Register a behavior planning function for a specific vehicle"""
        if vehicle_name not in self.vehicles:
            raise ValueError(f"Vehicle '{vehicle_name}' not found")
        
        self.behavior_planners[vehicle_name] = planner_function
    
    def update(self):
        if not self.pause:
            if self.real_time:
                current_time = time.time()
                dt = current_time - self.last_update
                self.last_update = current_time
                
                # Limit dt to prevent large jumps
                dt = min(dt, 0.1)
            else:
                dt = self.dt
            
            # Execute behavior planners
            self.execute_behavior_planners()
            
            # Update vehicle states
            all_out_of_bounds = True
            for vehicle in self.vehicles.values():
                vehicle.update(dt)
                
                # Check if vehicle is within bounds
                if not vehicle.is_out_of_bounds(self.road_bounds):
                    all_out_of_bounds = False
                    
                # Check if vehicle reached end of main lane
                if vehicle.x > MAIN_LANE_LENGTH:
                    print(f"Vehicle {vehicle.name} reached the end of main lane!")
            
            # Pause simulation if all vehicles are out of bounds
            if all_out_of_bounds and self.vehicles:
                print("All vehicles are out of bounds! Stopping simulation.")
                self.pause = True
            
            self.time += dt
    
    def execute_behavior_planners(self):
        """Execute all registered behavior planners"""
        for vehicle_name, planner in self.behavior_planners.items():
            if vehicle_name in self.vehicles:
                planner(self, self.vehicles[vehicle_name])
        
    def draw(self, screen):
        # Clean background
        screen.fill((245, 245, 245))
        
        # Enhanced road drawing
        self.draw_road(screen)
        
        # Pre-planned and current lane change trajectories
        self.draw_preplanned_trajectories(screen)
        
        # Current lane change trajectory for FV
        if 'FV' in self.vehicles and hasattr(self.vehicles['FV'], 'lane_change_trajectory'):
            self.draw_trajectory(screen, self.vehicles['FV'].lane_change_trajectory, 
                                color=(142, 68, 173), width=3)
        
        # Draw all vehicles without shadows
        for vehicle in self.vehicles.values():
            vehicle.draw(screen)
        
        # Time display with clean style
        font = pygame.font.SysFont('Arial', 24)
        time_text = font.render(f"Simulation Time: {self.time:.1f} s", True, (50, 50, 50))
        screen.blit(time_text, (10, 10))

        # Vehicle information display
        y_pos = 40
        info_font = pygame.font.SysFont('Arial', 18)
        for name, vehicle in self.vehicles.items():
            info_text = f"{name}: x={vehicle.x:.1f}m, y={vehicle.y:.1f}m, v={vehicle.v:.1f}m/s"
            vehicle_info = info_font.render(info_text, True, vehicle.color)
            screen.blit(vehicle_info, (10, y_pos))
            y_pos += 25
    
    def draw_road(self, screen):
        # Clean light background
        screen.fill((245, 245, 245))
        
        # Road surface color
        road_color = (220, 220, 220)
        road_line_color = (100, 100, 100)
        
        # Main lane bottom boundary
        pygame.draw.line(screen, road_line_color, 
                        world_to_screen(0, self.road_y + LANE_WIDTH/2 + LANE_WIDTH), 
                        world_to_screen(MAIN_LANE_LENGTH, self.road_y + LANE_WIDTH/2 + LANE_WIDTH), 
                        3)

        # Dashed center line
        x1, y1 = world_to_screen(0, self.road_y + LANE_WIDTH/2)
        x2, y2 = world_to_screen(MAIN_LANE_LENGTH, self.road_y + LANE_WIDTH/2)

        dash_length = 20
        gap_length = 15
        width = 2

        distance = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
        dx = (x2 - x1) / distance
        dy = (y2 - y1) / distance

        d = 0
        while d < distance:
            start = (x1 + dx * d, y1 + dy * d)
            d += dash_length
            end = (x1 + dx * min(d, distance), y1 + dy * min(d, distance))
            
            pygame.draw.line(screen, road_line_color, start, end, width)
            d += gap_length

        # Auxiliary lane top boundary
        pygame.draw.line(screen, road_line_color, 
                        world_to_screen(0, self.road_y - LANE_WIDTH/2), 
                        world_to_screen(AUX_LANE_LENGTH, self.road_y - LANE_WIDTH/2), 
                        3)

        # Merging area visualization
        merge_start = 50  # meters
        merge_color = (231, 76, 60, 50)  # Semi-transparent red
        merge_points = [
            world_to_screen(merge_start, self.road_y - LANE_WIDTH/2),
            world_to_screen(merge_start, self.road_y + LANE_WIDTH + LANE_WIDTH/2),
            world_to_screen(merge_start + 20, self.road_y + LANE_WIDTH + LANE_WIDTH/2),
            world_to_screen(merge_start + 20, self.road_y - LANE_WIDTH/2)
        ]
        
        # Create a surface for blending
        merge_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(merge_surface, merge_color, merge_points)
        screen.blit(merge_surface, (0, 0))

        # 50m marker with enhanced visibility
        font = pygame.font.SysFont('Arial', 18)
        mark_x = 50
        pygame.draw.line(screen, (231, 76, 60), 
                         world_to_screen(mark_x, self.road_y - LANE_WIDTH/2 - 1), 
                         world_to_screen(mark_x, self.road_y + LANE_WIDTH + LANE_WIDTH/2 + 1), 
                         2)
        mark_text = font.render("Merge Zone", True, (231, 76, 60))
        screen.blit(mark_text, world_to_screen(mark_x, self.road_y + LANE_WIDTH + LANE_WIDTH/2 + 2))

        # Road boundaries for collision detection
        self.road_bounds = [
            [0, self.road_y, MAIN_LANE_LENGTH, self.road_y + LANE_WIDTH + LANE_WIDTH/2],
            [0, self.road_y - LANE_WIDTH/2, AUX_LANE_LENGTH, self.road_y]
        ]
    
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.pause = not self.pause
            elif event.key == pygame.K_r:
                self.reset()
            elif event.key == pygame.K_t:
                self.real_time = not self.real_time
                self.last_update = time.time()
    
    def reset(self):
        """Reset the simulation"""
        self.vehicles.clear()
        self.behavior_planners.clear()
        self.time = 0
        self.pause = False
        self.last_update = time.time()
    
    def get_nearby_vehicles(self, vehicle, max_distance=50):
        """Get all vehicles within a certain distance of the given vehicle"""
        nearby = {}
        for name, other in self.vehicles.items():
            if other is not vehicle:
                dx = abs(vehicle.x - other.x)
                dy = abs(vehicle.y - other.y)
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < max_distance:
                    nearby[name] = {
                        'vehicle': other,
                        'distance': distance,
                        'dx': other.x - vehicle.x,  # positive if other is ahead
                        'dy': other.y - vehicle.y   # positive if other is to the right
                    }
        return nearby

    def get_lane_vehicles(self, lane, x_min=0, x_max=float('inf')):
        """Get all vehicles in a specific lane within x-range"""
        vehicles_in_lane = {}
        for name, vehicle in self.vehicles.items():
            if vehicle.lane == lane and x_min <= vehicle.x <= x_max:
                vehicles_in_lane[name] = vehicle
        return vehicles_in_lane
    
    def plan_lane_change_trajectory(self, vehicle, target_lane, distance=5, points=20):
        """
        规划车辆的换道轨迹
        
        参数:
        vehicle - 执行换道的车辆
        target_lane - 目标车道名称
        distance - 换道轨迹的总长度(m)
        points - 轨迹点的数量
        
        返回:
        包含轨迹信息的字典
        """
        if target_lane not in self.lane_positions:
            raise ValueError(f"目标车道 '{target_lane}' 未定义")
        
        target_y = self.lane_positions[target_lane]
        current_y = vehicle.y
        
        dy = target_y - current_y
        target_heading = math.atan2(dy, distance)
        
        # 生成完整轨迹点
        trajectory_points = self.generate_lane_change_trajectory(vehicle, target_lane, distance, points)
        
        return {
            'target_y': target_y,
            'target_heading': target_heading,
            'completion_distance': distance,
            'trajectory': trajectory_points
        }
        
    def generate_lane_change_trajectory(self, vehicle, target_lane, trajectory_length=20, points=20):
        """
        生成车辆的换道轨迹点序列
        
        参数:
        vehicle - 执行换道的车辆
        target_lane - 目标车道
        trajectory_length - 换道轨迹的总长度(m)
        points - 轨迹点的数量
        
        返回:
        轨迹点列表 [(x1,y1), (x2,y2), ...]
        """
        if target_lane not in self.lane_positions:
            raise ValueError(f"目标车道 '{target_lane}' 未定义")
        
        start_x = vehicle.x
        start_y = vehicle.y
        target_y = self.lane_positions[target_lane]
        
        # 使用S形曲线平滑过渡
        trajectory = []
        for i in range(points):
            # 沿x轴的位置
            progress = i / (points - 1)
            x = start_x + progress * trajectory_length
            
            # 使用正弦函数创建S形轨迹，使其在始末平滑
            transition = 0.5 - 0.5 * math.cos(math.pi * progress)
            y = start_y + (target_y - start_y) * transition
            
            # 计算当前点的航向角(与x轴夹角)
            heading = 0
            if i < points-1:
                next_progress = (i+1) / (points - 1)
                next_x = start_x + next_progress * trajectory_length
                next_y = start_y + (target_y - start_y) * (0.5 - 0.5 * math.cos(math.pi * next_progress))
                
                dx = next_x - x
                dy = next_y - y
                heading = math.atan2(dy, dx)
            
            trajectory.append((x, y, heading))
        
        return trajectory
    
    def draw_trajectory(self, screen, trajectory, color=(255, 0, 255), width=2):
        """绘制轨迹路径"""
        if not trajectory:
            return
            
        # 绘制轨迹线
        points = [world_to_screen(x, y) for x, y, _ in trajectory]
        if len(points) > 1:
            pygame.draw.lines(screen, color, False, points, width)
        
        # 在轨迹起点和终点绘制标记
        if points:
            pygame.draw.circle(screen, color, points[0], 5)
            pygame.draw.circle(screen, color, points[-1], 5)
        
        # 每隔几个点绘制方向箭头
        arrow_step = max(1, len(trajectory) // 5)
        for i in range(0, len(trajectory), arrow_step):
            x, y, heading = trajectory[i]
            screen_x, screen_y = world_to_screen(x, y)
            
            # 绘制方向箭头
            arrow_length = 10
            end_x = screen_x + arrow_length * math.cos(heading)
            end_y = screen_y + arrow_length * math.sin(heading)
            pygame.draw.line(screen, color, (screen_x, screen_y), (end_x, end_y), 2)
    
    def draw_preplanned_trajectories(self, screen):
        """
        绘制FV车辆的所有预规划换道轨迹
        使用较淡的颜色以区分当前实际执行的轨迹
        """
        fv = self.vehicles.get("FV")
        if not fv or not hasattr(fv, 'preplanned_trajectories'):
            return
        
        # 定义每种换道策略对应的淡色
        light_colors = {
            ActionType.IDM_FV_MERGE_1: (255, 200, 200),  # 淡红色
            ActionType.IDM_FV_MERGE_2: (200, 255, 200),  # 淡绿色
            ActionType.IDM_FV_MERGE_3: (200, 200, 255),  # 淡蓝色
            ActionType.IDM_FV_MERGE_4: (255, 255, 200),  # 淡黄色
            ActionType.IDM_FV_MERGE_5: (255, 200, 255)   # 淡紫色
        }
        
        # 绘制所有预规划的轨迹
        for strategy, trajectory_data in fv.preplanned_trajectories.items():
            trajectory = trajectory_data['trajectory']
            color = light_colors.get(strategy, (220, 220, 220))  # 默认淡灰色
            
            # 绘制轨迹线（较细的线条）
            points = [self.world_to_screen(x, y) for x, y, _ in trajectory]
            if len(points) > 1:
                pygame.draw.lines(screen, color, False, points, 1)  # 线条宽度设为2，更淡
            
            # 在轨迹终点添加小文本标签
            if points:
                # 获取策略名称的简短表示
                strategy_name = str(strategy).split('.')[-1].replace('IDM_FV_MERGE_', 'M')
                font = pygame.font.SysFont('Arial', 14)  # 小字体
                text = font.render(strategy_name, True, color)
                screen.blit(text, (points[-1][0] + 5, points[-1][1] - 5))
        
        # 如果FV正在执行换道，用更明显的颜色高亮显示当前执行的轨迹
        if hasattr(fv, 'is_executing_lane_change') and fv.is_executing_lane_change and hasattr(fv, 'game_action'):
            if 'original_action' in fv.game_action:
                current_strategy = fv.game_action['original_action']
                if current_strategy in fv.preplanned_trajectories:
                    current_trajectory = fv.preplanned_trajectories[current_strategy]['trajectory']
                    self.draw_trajectory(screen, current_trajectory, color=(255, 0, 255), width=2)
                    
    def world_to_screen(self, x, y):
        screen_x = int(x * SCALE )
        screen_y = int(SCREEN_HEIGHT - (y * SCALE))
        return (screen_x, screen_y)
    
    




