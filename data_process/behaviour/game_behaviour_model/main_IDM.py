import pygame
import sys
import time
from vehicle import * 
from simulator import MergingSimulation
from IDM import *
import math


def get_env_info(sim):
    """获取环境信息"""
    env_state = sim.get_environment_state()
    env_info = {}
    env_info['env_state'] = env_state
    return env_info

def fv_lane_change_behavior(sim, vehicle):
    """FV车辆的换道行为规划"""

    if not hasattr(vehicle, 'lane_change_flag'):
        vehicle.lane_change_flag = 1  
    
    if not hasattr(vehicle, 'lane_change_trajectory'):
        vehicle.lane_change_trajectory = None
        vehicle.lane_change_index = 0
        vehicle.lane_change_target_lane = None
    
    if vehicle.lane_change_flag == 0:
        trajectory_info = sim.plan_lane_change_trajectory(vehicle, 'main', distance=10, points=100)
        vehicle.lane_change_trajectory = trajectory_info['trajectory']
        vehicle.lane_change_index = 0
        vehicle.lane_change_target_lane = 'main'
        return
    
    # 当flag为1时，执行换道
    elif vehicle.lane_change_flag == 1:
        if vehicle.lane_change_trajectory is None:
            trajectory_info = sim.plan_lane_change_trajectory(vehicle, 'main', distance=10, points=100)
            vehicle.lane_change_trajectory = trajectory_info['trajectory']
            vehicle.lane_change_index = 0
            vehicle.lane_change_target_lane = 'main'
            print(f"已为{vehicle.name}规划换道轨迹，从{vehicle.lane}到{vehicle.lane_change_target_lane}")
        
        # 如果车辆已完成换道
        if vehicle.lane_change_index >= len(vehicle.lane_change_trajectory):
            if vehicle.lane != vehicle.lane_change_target_lane:
                vehicle.lane = vehicle.lane_change_target_lane
                print(f"{vehicle.name}已完成换道到{vehicle.lane}")
                vehicle.lane_change_flag = 0  # 重置换道标志
            return
        
        # 获取当前轨迹点
        current_x, current_y, current_heading = vehicle.lane_change_trajectory[vehicle.lane_change_index]
        
        # 计算当前位置与目标点的距离
        dx = current_x - vehicle.x
        dy = current_y - vehicle.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # 寻找目标车道的前车
        leader = find_target_lane_leader(sim, vehicle)
        
        # 使用IDM计算加速度
        acceleration = calculate_idm_acceleration(vehicle, leader, IDM_PARAMS)
        
        vehicle.v += acceleration * sim.dt  # 更新速度
        
        if distance < 1:  # 容差1米
            vehicle.lane_change_index += 1
      
        
        vehicle.heading = current_heading  # 更新车辆朝向

# 基于距离寻找前车的逻辑，不符合实际情况，需要替换为博弈策略。
def find_leading_vehicle(sim, ego_vehicle):
    """寻找换道过程中的潜在前车"""
    # 在当前车道寻找前车
    current_lane_vehicles = sim.get_lane_vehicles('aux')
    # 在目标车道寻找潜在前车（如果正在换道）
 
    target_lane_vehicles = sim.get_lane_vehicles('main')
    
    # 合并所有可能的前车
    all_vehicles = list(current_lane_vehicles.values()) + list(target_lane_vehicles.values())
    vehicles_ahead = [v for v in all_vehicles if v.x > ego_vehicle.x and v is not ego_vehicle]
    
    # 找到距离最近的前车
    if vehicles_ahead:
        return min(vehicles_ahead, key=lambda v: abs(v.x - ego_vehicle.x))
    else:
        return None

def find_target_lane_leader(sim, ego_vehicle):
    """仅寻找目标车道中的前车"""
    if not hasattr(ego_vehicle, 'lane_change_target_lane') or ego_vehicle.lane_change_target_lane is None:
        return None
    
    # 获取目标车道的所有车辆
    target_lane_vehicles = sim.get_lane_vehicles(ego_vehicle.lane_change_target_lane)
    
    # 找出目标车道中所有在ego车辆前方的车辆
    vehicles_ahead = [v for v in target_lane_vehicles.values() 
                      if v.x > ego_vehicle.x and v is not ego_vehicle]
    
    # 找到距离最近的前车
    if vehicles_ahead:
        return min(vehicles_ahead, key=lambda v: abs(v.x - ego_vehicle.x))
    else:
        return None

def car_following_behavior(sim, vehicle):
    """跟驰行为"""
    # 寻找前车
    leader = find_leading_vehicle(sim, vehicle)
    
    # 使用IDM计算加速度
    acceleration = calculate_idm_acceleration(vehicle, leader, IDM_PARAMS)
    
    # 更新车辆状态
    vehicle.a = acceleration


# 仿真中五辆车的行为定义：
# 1、ZV：主道交互车辆（和辅道车博弈————博弈动作选择跟车对象）
# 2、LZV：主道前车（简单跟车策略，后续依据具体的环境信息可切换成博弈策略）
# 3、BZV：主道前车（简单跟车策略，后续依据具体的环境信息可切换成博弈策略）
# 4、FV：辅道交互车辆（和主道车博弈————博弈动作选择跟车对象）
# 5、LFV：辅道前车（简单跟车策略，后续依据具体的环境信息可切换成博弈策略）

def setup_default_scenario(sim):
    """Set up the default scenario with five vehicles"""
    # Main lane vehicles
    sim.add_vehicle_in_lane("ZV", x=15, lane="main", v=6)
    sim.add_vehicle_in_lane("LZV", x=30, lane="main", v=5)
    sim.add_vehicle_in_lane("BZV", x=5, lane="main", v=4)
    
    # Auxiliary lane vehicles
    sim.add_vehicle_in_lane("FV", x=20, lane="aux", v=4)
    sim.add_vehicle_in_lane("LFV", x=35, lane="aux", v=4)
    
    # Register behaviors
    sim.register_behavior_planner("FV", fv_lane_change_behavior)
    sim.register_behavior_planner("ZV", car_following_behavior)
    sim.register_behavior_planner("BZV", car_following_behavior)
    sim.register_behavior_planner("LZV", car_following_behavior)
    sim.register_behavior_planner("LFV", car_following_behavior)



def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    sim = MergingSimulation()
    
    # Set up initial scenario
    setup_default_scenario(sim)
    
    clock = pygame.time.Clock()
    sim.last_update = time.time()
    
    while sim.running:
        for event in pygame.event.get():
            sim.handle_event(event)
        
        sim.update()
        sim.draw(screen)
        
        pygame.display.flip()
        
        if not sim.real_time:
            clock.tick(60)
        else:
            clock.tick(120)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()