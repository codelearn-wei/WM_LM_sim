import pygame
import sys
import time
import math
from vehicle import * 
from simulator import MergingSimulation
from IDM import *
from game.main_game import GameModel
from data_process.behaviour.game_behaviour_model.game.game_action import *
from config import *

# ================= Utility Functions =================
def calculate_thw(distance, speed):
    if speed <= 0:
        return float('inf')
    if distance < 0:
        raise ValueError("Distance cannot be negative")
    
    return distance / speed

def calculate_ttc(distance, relative_speed):
    if relative_speed <= 0:
        return float('inf')
    if distance <= 0:
        return 0.0
    if distance < 0:
        raise ValueError("Distance cannot be negative")
    
    return distance / relative_speed

def find_leading_vehicle(sim, ego_vehicle):
    current_lane_vehicles = sim.get_lane_vehicles(ego_vehicle.lane)
    
    vehicles_ahead = [v for v in current_lane_vehicles.values() 
                     if v.x > ego_vehicle.x and v is not ego_vehicle]
    
    if vehicles_ahead:
        return min(vehicles_ahead, key=lambda v: abs(v.x - ego_vehicle.x))
    else:
        return None

def find_vehicle_by_name(sim, name):
    return sim.vehicles.get(name)

def adjust_idm_params_by_mode(following_mode):
    idm_params = IDM_PARAMS.copy()
    
    if following_mode == 'yield':
        idm_params['s0'] *= 1.5
        idm_params['a_max'] *= 0.8
        idm_params['b'] *= 1.2
    
    elif following_mode == 'aggressive':
        idm_params['s0'] *= 0.7
        idm_params['a_max'] *= 1.3
        idm_params['b'] *= 0.8
    
    return idm_params

def map_vehicle_to_role(vehicle_name):
    if vehicle_name == "FV":
        return VehicleRole.FV
    elif vehicle_name == "ZV":
        return VehicleRole.ZV
    else:
        return None

def preplan_all_lane_change_trajectories(sim, fv_vehicle):
    """
    预规划所有可能的换道轨迹并将它们存储在FV车辆对象中
    
    Args:
        sim: 模拟器对象
        fv_vehicle: FV车辆对象
    
    Returns:
        预规划的所有轨迹字典
    """
    if not hasattr(fv_vehicle, 'preplanned_trajectories'):
        fv_vehicle.preplanned_trajectories = {}
    
    # 定义所有可能的换道策略
    lane_change_strategies = [
        ActionType.IDM_FV_MERGE_1,
        ActionType.IDM_FV_MERGE_2,
        ActionType.IDM_FV_MERGE_3,
        ActionType.IDM_FV_MERGE_4,
        ActionType.IDM_FV_MERGE_5
    ]
    
    target_lane = 'main' if fv_vehicle.lane == 'aux' else 'aux'
    
    # 遍历所有可能的换道策略，为每个策略计算轨迹
    for strategy in lane_change_strategies:
        # 获取该策略的换道参数
        params = LANE_CHANGE_PARAMS.get(strategy, {'distance': 20, 'points': 100})
        
        # 计算轨迹
        trajectory_info = sim.plan_lane_change_trajectory(
            fv_vehicle, 
            target_lane=target_lane, 
            distance=params['distance'], 
            points=params['points']
        )
        
        # 存储轨迹
        fv_vehicle.preplanned_trajectories[strategy] = {
            'trajectory': trajectory_info['trajectory'],
            'params': params,
            'target_lane': target_lane
        }
    
    return fv_vehicle.preplanned_trajectories

# ================= Game Model and Prediction Functions =================
def get_pay_off_info(sim):
    pay_off_info = {}
    vehicles = sim.vehicles
    ZV = vehicles.get("ZV")
    FV = vehicles.get("FV")
    LFV = vehicles.get("LFV")
    BZV = vehicles.get("BZV")
    LZV = vehicles.get("LZV")
    
    FV_dis_2_LFV = abs(FV.x - LFV.x)
    FV_dis_2_LZV = abs(FV.x - LZV.x)
    FV_dis_2_merge = abs(FV.x - GAME_PARAMS['merge_point'][0])
    ZV_dis_2_LZV = abs(ZV.x - LZV.x)
    ZV_dis_2_BZV = abs(ZV.x - BZV.x)
    ZV_dis_2_FV = abs(ZV.x - FV.x)
    ZV_dis_2_merge = abs(ZV.x - GAME_PARAMS['merge_point'][0])
    
    v_ZV = ZV.v
    a_ZV = ZV.a
    v_FV = FV.v
    a_FV = FV.a
    v_LFV = LFV.v
    v_BZV = BZV.v
    
    thw_FV_2_LFV = calculate_thw(FV_dis_2_LFV, v_FV)
    thw_ZV_2_FV = calculate_thw(ZV_dis_2_FV, v_ZV)
    thw_ZV_2_LZV = calculate_thw(ZV_dis_2_LZV, v_ZV)
    thw_BZV_2_ZV = calculate_thw(ZV_dis_2_BZV, v_BZV)
    
    pay_off_info["FV"] = {
        "a_FV": a_FV,
        "v_FV": v_FV,
        "v_LFV": v_LFV,
        "FV_dis_2_merge": FV_dis_2_merge,
        "thw_ZV_2_FV": thw_ZV_2_FV,
        "thw_FV_2_LFV": thw_FV_2_LFV,
        "thw_BZV_2_ZV": thw_BZV_2_ZV,
        "FV_dis_2_LZV": FV_dis_2_LZV,
        "length": FV.length,
        "width": FV.width,
    }
    
    pay_off_info["ZV"] = {
        "a_ZV": a_ZV,
        "v_ZV": v_ZV,
        "ZV_dis_2_LZV": ZV_dis_2_LZV,
        "ZV_dis_2_merge": ZV_dis_2_merge,
        "thw_ZV_2_FV": thw_ZV_2_FV,
        "thw_ZV_2_LZV": thw_ZV_2_LZV,
        "thw_BZV_2_ZV": thw_BZV_2_ZV,
        "length": ZV.length,
        "width": ZV.width,
    }
    
    return pay_off_info

def predict_zv_state(ZV, vehicles, action):
    LZV = vehicles.get("LZV")
    FV = vehicles.get("FV")
    
    if action == ActionType.IDM_ZV_2_LZV:
        a_zv = calculate_idm_acceleration(ZV, LZV, IDM_PARAMS)
        new_state = predict_trajectory(ZV, a_zv, dt=0.1, horizon=2.0)
    elif action == ActionType.IDM_ZV_2_FV:
        a_zv = calculate_idm_acceleration(ZV, FV, IDM_PARAMS)
        new_state = predict_trajectory(ZV, a_zv, dt=0.1, horizon=2.0)
    
    return new_state

def predict_fv_state(FV, vehicles, action, sim):
    LFV = vehicles.get("LFV")
    ZFV = vehicles.get("ZFV")
    
    if action == ActionType.IDM_FV_2_LFV:
        a_fv = calculate_idm_acceleration(FV, LFV, IDM_PARAMS)
        new_state = predict_trajectory(FV, a_fv, dt=0.1, horizon=2.0)
    elif action in [ActionType.IDM_FV_MERGE_1, ActionType.IDM_FV_MERGE_2, 
                   ActionType.IDM_FV_MERGE_3, ActionType.IDM_FV_MERGE_4, 
                   ActionType.IDM_FV_MERGE_5]:
        a_fv = calculate_idm_acceleration(FV, ZFV, IDM_PARAMS)
        
        params = LANE_CHANGE_PARAMS.get(action, {'distance': 20, 'points': 100})
        
        trajectory_info = sim.plan_lane_change_trajectory(
            FV, 
            target_lane='main', 
            distance=params['distance'], 
            points=params['points']
        )
        raw_trajectory = trajectory_info['trajectory'] 
        new_state = add_time_to_trajectory(FV, raw_trajectory, a_fv, dt=0.1, max_time=2.0)
    
    return new_state

def predict_future_states(sim):
    future_states = {}
    
    action_combinations = [
        (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)
    ]
    
    zv_actions = {
        0: ActionType.IDM_ZV_2_LZV,
        1: ActionType.IDM_ZV_2_FV
    }
    
    fv_actions = {
        2: ActionType.IDM_FV_2_LFV,
        3: ActionType.IDM_FV_MERGE_1,
        4: ActionType.IDM_FV_MERGE_2,
        5: ActionType.IDM_FV_MERGE_3,
        6: ActionType.IDM_FV_MERGE_4,
        7: ActionType.IDM_FV_MERGE_5,
    }
    
    vehicles = sim.vehicles
    ZV = vehicles.get("ZV")
    FV = vehicles.get("FV")
 
    for zv_num, fv_num in action_combinations:
        zv_action = zv_actions[zv_num]
        fv_action = fv_actions[fv_num]
        
        zv_state = predict_zv_state(ZV, vehicles, zv_action)
        fv_state = predict_fv_state(FV, vehicles, fv_action, sim)
        
        future_states[(zv_num, fv_num)] = {
            'zv': zv_state,
            'fv': fv_state
        }
    
    return future_states

# ================= Game Controller Functions =================
def game_leader(sim, vehicle):
    if hasattr(vehicle, 'is_executing_lane_change') and vehicle.is_executing_lane_change:
        return vehicle.game_action
    
    vehicles = {}
    fv = sim.vehicles.get("FV")
    zv = sim.vehicles.get("ZV")
    
    predict_state = predict_future_states(sim)
    pay_off_info = get_pay_off_info(sim)
    game_model = GameModel(GAME_PARAMS, predict_state, pay_off_info)
    
    if fv and zv:
        vehicles[VehicleRole.FV] = fv
        vehicles[VehicleRole.ZV] = zv
        
        P_FV, P_main = game_model.generate_payoff_matrices(vehicles)
        
        equilibrium = game_model.stackelberg_equilibrium(P_FV, P_main)
        
        vehicle_role = map_vehicle_to_role(vehicle.name)
        if vehicle_role:
            if vehicle_role == VehicleRole.FV:
                strategy = equilibrium['fv_strategy']
            else:
                strategy = equilibrium['zv_strategy']
                 
            selected_action = game_model.select_action(vehicle_role, strategy)
            action_dict = convert_game_action_to_sim_action(selected_action, vehicle, sim)
            
            vehicle.game_action = action_dict
            return action_dict
    
    default_action = {
        'action_type': 'car_following',
        'following_mode': 'normal',
        'target_vehicle': find_leading_vehicle(sim, vehicle),
        'target_lane': vehicle.lane
    }
    
    vehicle.game_action = default_action
    return default_action

def convert_game_action_to_sim_action(game_action, vehicle, sim):
    action_dict = {
        'action_type': 'car_following',
        'following_mode': 'normal',
        'target_vehicle': None, 
        'target_lane': vehicle.lane,
        'original_action': game_action['type']
    }
    
    if game_action['type'] == ActionType.IDM_ZV_2_LZV:
        action_dict['action_type'] = 'car_following'
        action_dict['target_vehicle'] = find_leading_vehicle(sim, vehicle)
        
    elif game_action['type'] == ActionType.IDM_ZV_2_FV:
        action_dict['action_type'] = 'car_following'
        action_dict['following_mode'] = 'yield'
        action_dict['target_vehicle'] = find_vehicle_by_name(sim, "FV" if vehicle.name == "ZV" else "ZV")
        
    elif game_action['type'] == ActionType.IDM_FV_2_LFV:
        action_dict['action_type'] = 'car_following'
        action_dict['target_vehicle'] = find_leading_vehicle(sim, vehicle)
        
    elif game_action['type'] in [
        ActionType.IDM_FV_MERGE_1, 
        ActionType.IDM_FV_MERGE_2, 
        ActionType.IDM_FV_MERGE_3,
        ActionType.IDM_FV_MERGE_4,
        ActionType.IDM_FV_MERGE_5
    ]:
        action_dict['action_type'] = 'lane_change'
        action_dict['target_lane'] = 'main' if vehicle.lane == 'aux' else 'aux'
        action_dict['target_vehicle'] = find_vehicle_by_name(sim, "LZV" if vehicle.name == "FV" else "FV")
        
    return action_dict

def get_lane_change_params(vehicle, game_action):
    default_params = {
        'distance': 20,
        'points': 100
    }
    
    original_action_type = None
    if hasattr(vehicle, 'game_action') and vehicle.game_action and 'original_action' in vehicle.game_action:
        original_action_type = vehicle.game_action['original_action']
    
    if not original_action_type and 'original_action' in game_action:
        original_action_type = game_action['original_action']
    
    return LANE_CHANGE_PARAMS.get(original_action_type, default_params)

# ================= Vehicle Behavior Functions =================
def car_following_behavior(sim, vehicle):
    leader = find_leading_vehicle(sim, vehicle)
    acceleration = calculate_idm_acceleration(vehicle, leader, IDM_PARAMS)
    vehicle.a = acceleration

def fv_game_lane_change_behavior(sim, vehicle):
    if not hasattr(vehicle, 'game_action'):
        vehicle.game_action = None
        
    if not hasattr(vehicle, 'lane_change_trajectory'):
        vehicle.lane_change_trajectory = None
        vehicle.lane_change_index = 0
        vehicle.lane_change_target_lane = None
        
    if not hasattr(vehicle, 'lane_change_completed'):
        vehicle.lane_change_completed = False
        
    if not hasattr(vehicle, 'is_executing_lane_change'):
        vehicle.is_executing_lane_change = False
    
    if vehicle.lane_change_completed:
        leader = find_leading_vehicle(sim, vehicle)
        acceleration = calculate_idm_acceleration(vehicle, leader, IDM_PARAMS)
        vehicle.a = acceleration
        return
    preplan_all_lane_change_trajectories(sim, vehicle)
    
    if vehicle.is_executing_lane_change and vehicle.lane_change_trajectory:
        if vehicle.lane_change_index < len(vehicle.lane_change_trajectory):
            current_x, current_y, current_heading = vehicle.lane_change_trajectory[vehicle.lane_change_index]
            
            dx = current_x - vehicle.x
            dy = current_y - vehicle.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if vehicle.lane_change_target_lane == 'main':
                target_vehicle = find_vehicle_by_name(sim, "LZV")
            else:
                target_vehicle = find_vehicle_by_name(sim, "LFV")
            
            if target_vehicle is None:
                target_vehicle = find_leading_vehicle(sim, vehicle)
                
            acceleration = calculate_idm_acceleration(vehicle, target_vehicle, IDM_PARAMS)
            vehicle.a = acceleration
            vehicle.v += acceleration * sim.dt
            
            if distance < 1:
                vehicle.lane_change_index += 1
            
            vehicle.heading = current_heading
            return
        else:
            if vehicle.lane != vehicle.lane_change_target_lane:
                vehicle.lane = vehicle.lane_change_target_lane
                print(f"{vehicle.name} completed lane change to {vehicle.lane}")
                
            vehicle.lane_change_trajectory = None
            vehicle.lane_change_index = 0
            vehicle.lane_change_completed = True
            vehicle.is_executing_lane_change = False
            print(f"{vehicle.name} lane change complete, switching to car following behavior")
            return
    
    game_action = game_leader(sim, vehicle)
    
    if game_action['action_type'] == 'lane_change' and not vehicle.is_executing_lane_change:
        target_lane = game_action['target_lane']
        target_vehicle = game_action['target_vehicle']
        
        lane_change_params = get_lane_change_params(vehicle, game_action)
        
        trajectory_info = sim.plan_lane_change_trajectory(
            vehicle, 
            target_lane, 
            distance=lane_change_params['distance'], 
            points=lane_change_params['points']
        )
        vehicle.lane_change_trajectory = trajectory_info['trajectory']
        vehicle.lane_change_index = 0
        vehicle.lane_change_target_lane = target_lane
        vehicle.is_executing_lane_change = True
        print(f"Planning {vehicle.name} lane change from {vehicle.lane} to {vehicle.lane_change_target_lane} with {lane_change_params}")
        
    else:
        following_mode = game_action['following_mode']
        target_vehicle = game_action['target_vehicle']
        
        idm_params = adjust_idm_params_by_mode(following_mode)
        acceleration = calculate_idm_acceleration(vehicle, target_vehicle, idm_params)
        vehicle.a = acceleration

def zv_game_following_behavior(sim, vehicle):
    if not hasattr(vehicle, 'game_action'):
        vehicle.game_action = None
    
    fv = sim.vehicles.get("FV")
    if fv and hasattr(fv, 'is_executing_lane_change') and fv.is_executing_lane_change:
        if vehicle.game_action:
            following_mode = vehicle.game_action['following_mode']
            target_vehicle = vehicle.game_action['target_vehicle']
            
            idm_params = adjust_idm_params_by_mode(following_mode)
            acceleration = calculate_idm_acceleration(vehicle, target_vehicle, idm_params)
            vehicle.a = acceleration
            return
    
    if fv and hasattr(fv, 'lane_change_completed') and fv.lane_change_completed:
        leader = find_leading_vehicle(sim, vehicle)
        acceleration = calculate_idm_acceleration(vehicle, leader, IDM_PARAMS)
        vehicle.a = acceleration
        return
    
    game_action = game_leader(sim, vehicle)
    
    following_mode = game_action['following_mode']
    target_vehicle = game_action['target_vehicle']
    
    idm_params = adjust_idm_params_by_mode(following_mode)
    acceleration = calculate_idm_acceleration(vehicle, target_vehicle, idm_params)
    
    vehicle.a = acceleration

# ================= Simulation Setup and Main Function =================
def setup_default_scenario(sim):
    # Main lane vehicles
    sim.add_vehicle_in_lane("ZV", x=24, lane="main", v=3)
    sim.add_vehicle_in_lane("LZV", x=30, lane="main", v=5)
    sim.add_vehicle_in_lane("BZV", x=5, lane="main", v=4)
    
    # Auxiliary lane vehicles
    sim.add_vehicle_in_lane("FV", x=24, lane="aux", v=6)
    sim.add_vehicle_in_lane("LFV", x=35, lane="aux", v=4)
    
    # Register behaviors
    sim.register_behavior_planner("FV", fv_game_lane_change_behavior)
    sim.register_behavior_planner("ZV", zv_game_following_behavior)
    sim.register_behavior_planner("BZV", car_following_behavior)
    sim.register_behavior_planner("LZV", car_following_behavior)
    sim.register_behavior_planner("LFV", car_following_behavior)

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    sim = MergingSimulation()
    
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