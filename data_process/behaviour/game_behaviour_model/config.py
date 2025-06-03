# Configuration file for merging simulation parameters

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# IDM (Intelligent Driver Model) Parameters
IDM_PARAMS = {
    'v0': 6.0,    # Desired velocity
    's0': 2.0,     # Minimum spacing
    'T': 1.5,      # Time headway
    'a_max': 1.5,  # Maximum acceleration
    'b': 2.0,      # Comfortable deceleration
    'delta': 4.0,  # Acceleration exponent
}

# Game theory model parameters
GAME_PARAMS = {
    # ZV (主路车辆) 参数
    'target_zv_v': 4.0,                # 目标速度 (m/s)
    'target_zv_2_lzv_d': 10.0,          # 与前车目标距离 (m)
    'target_thw_zv_2_fv': 1.0,          # 与FV的目标车头时距 (s)
    'comfort_weight': 0.15,             # 舒适度权重
    'speed_weight': 0.20,               # 速度偏差权重
    'distance_weight': 0.15,            # 距离偏差权重
    'thw_weight': 0.10,                 # 车头时距权重
    'collision_weight': 0.2,            # 避免碰撞权重
    'zv_payoff_min': -3.0,              # ZV最小收益
    'zv_payoff_max': 3.0,               # ZV最大收益
    
    # FV (匝道车辆) 参数
    'target_fv_v': 6.0,                # 目标速度 (m/s)
    'target_thw_fv_2_zv': 1.0,          # 与ZV的目标车头时距 (s)
    'merge_critical_distance': 50.0,    # 关键合流距离 (m)
    'comfort_weight_fv': 0.10,          # 舒适度权重
    'speed_weight_fv': 0.15,            # 速度偏差权重
    'thw_weight_fv': 0.10,              # 车头时距权重
    'merge_pressure_weight': 0.20,      # 合流压力权重
    'collision_weight_fv': 2,        # 避免碰撞权重
    'fv_payoff_min': -4.0,              # FV最小收益
    'fv_payoff_max': 4.0,               # FV最大收益
    
    # 碰撞风险评估参数
    'collision_distance_threshold': 2.0,  # 碰撞距离阈值 (m)
    'collision_time_horizon': 3.0,        # 碰撞时间视野 (s)
    'collision_risk_max': 1.0,            # 最大碰撞风险
    'merge_point': (50.0 , 0)            # 合流点位置 (m)
}

# Lane change behavior parameters for different action types
from data_process.behaviour.game_behaviour_model.game.game_action import ActionType

LANE_CHANGE_PARAMS = {
    ActionType.IDM_FV_MERGE_1: {'distance': 30, 'points': 120},  # Conservative
    ActionType.IDM_FV_MERGE_2: {'distance': 20, 'points': 100},  # Normal
    ActionType.IDM_FV_MERGE_3: {'distance': 15, 'points': 80},   # Aggressive
    ActionType.IDM_FV_MERGE_4: {'distance': 10, 'points': 60},   # Very aggressive
    ActionType.IDM_FV_MERGE_5: {'distance': 8, 'points': 50}     # Emergency
}

# Vehicle initial positions and velocities
INITIAL_VEHICLE_CONFIG = {
    "ZV":  {"x": 24, "lane": "main", "v": 3},
    "LZV": {"x": 30, "lane": "main", "v": 5},
    "BZV": {"x": 5,  "lane": "main", "v": 4},
    "FV":  {"x": 24, "lane": "aux", "v": 6},
    "LFV": {"x": 35, "lane": "aux", "v": 4}
}