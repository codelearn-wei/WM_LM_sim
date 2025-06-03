import numpy as np

#! 1、主道车辆认知环境认知参数
MERGE_POINT = 15   ### 换道终端位置
MERGE_TRA_INTERVAL = 3  ### 采样轨迹位置间隔（m）
MERGE_DURATION = 3  ### 换道持续时间（s）
MERGE_DURATION_INTERVAL = 0.5
 
### 主车跟驰IDM参数
MAINA_IDM_PARAMS = {'desired_velocity': 6.0, 'time_headway': 1.0, 'minimum_spacing': 1.0}
### 辅道变道IDM参数
AUXA_MERGE_IDM_PARAMS = {'desired_velocity': 5.0, 'time_headway': 1.2, 'minimum_spacing': 2.0}
### 辅道跟驰IDM参数
AUXA_IDM_PARAMS = {'desired_velocity': 5.0, 'time_headway': 1.2, 'minimum_spacing': 2.0}


#! 2、主道辅道车辆博弈收益参数
class TrajectoryConfig:
    VEHICLE_RADIUS = 1
    MIN_SAFE_DISTANCE = 5.0
    SAFE_ACC_LIMIT = 3.0
    MAIN_TARGET_SPEED = 6.0
    AUX_TARGET_SPEED = 5.0
    MERGE_POINT_X = 1060.0
    MERGE_PRESSURE_DISTANCE = 200.0
    Y_MERGE_THRESHOLD = 963.0
    
    LANE_WIDTH = 4  # 车道宽度
    COLLISION_ZONE_THRESHOLD = 5.0  # 碰撞风险区域阈值
    
    # class MainWeights:
    #     COMFORT = 0.2
    #     EFFICIENCY = 0.4
    #     SAFETY = 0.00
    
    # class AuxWeights:
    #     COMFORT = 0.2
    #     EFFICIENCY = 0.6
    #     SAFETY = 1
    #     MERGE_PRESSURE = 0


class IDMParams:
    """
    智能驾驶模型（IDM）参数类，用于模拟车辆的跟车行为。
    """
    def __init__(self, desired_velocity=5.0, time_headway=1.0, minimum_spacing=1.0, max_acceleration=3.0, comfortable_deceleration=4.0):
        """
        初始化 IDM 参数。
        
        :param desired_velocity: 期望速度 (m/s)
        :param time_headway: 时间间隙 (s)
        :param minimum_spacing: 最小间距 (m)
        :param max_acceleration: 最大加速度 (m/s²)
        :param comfortable_deceleration: 舒适减速度 (m/s²)
        """
        self.desired_velocity = desired_velocity
        self.time_headway = time_headway
        self.minimum_spacing = minimum_spacing
        self.max_acceleration = max_acceleration
        self.comfortable_deceleration = comfortable_deceleration

    def as_dict(self):
        """
        将参数转换为字典形式。
        
        :return: 参数字典
        """
        return self.__dict__

class PathFollowingParams:
    """
    路径跟随参数类，用于控制车辆如何跟随预定义路径。
    """
    def __init__(self, lookahead_factor=2.0, max_lookahead=10.0, max_heading_change=0.1, lateral_correction_threshold=0.5, lateral_correction_factor=0.1):
        """
        初始化路径跟随参数。
        
        :param lookahead_factor: 前视因子
        :param max_lookahead: 最大前视距离 (m)
        :param max_heading_change: 最大航向变化 (rad)
        :param lateral_correction_threshold: 横向校正阈值 (m)
        :param lateral_correction_factor: 横向校正因子
        """
        self.lookahead_factor = lookahead_factor
        self.max_lookahead = max_lookahead
        self.max_heading_change = max_heading_change
        self.lateral_correction_threshold = lateral_correction_threshold
        self.lateral_correction_factor = lateral_correction_factor

class MergeParams:
    """
    合并参数类，用于控制车辆的合并动作。
    """
    def __init__(self, merge_point_distance=10.0, merge_duration=3.0, control_point_factor=3.0):
        """
        初始化合并参数。
        
        :param merge_point_distance: 合并点距离 (m)
        :param merge_duration: 合并持续时间 (s)
        :param control_point_factor: 控制点因子
        """
        self.merge_point_distance = MERGE_POINT
        self.merge_duration = MERGE_DURATION
        self.control_point_factor = control_point_factor
        self.tra_interval = MERGE_TRA_INTERVAL
        self.merge_duration_interval = MERGE_DURATION_INTERVAL


# 体现为环境车辆对主车车辆的行为认知
class AwarenessParams:
    """
    感知参数类，汇总所有参数以配置不同场景或车辆类型的行为。
    """
    def __init__(self):
        """
        初始化感知参数，包含主车和辅道车辆的 IDM 参数、自由驾驶参数、路径跟随参数、合并参数等。
        """
        self.main_idm_params = IDMParams(desired_velocity=MAINA_IDM_PARAMS['desired_velocity'], time_headway=MAINA_IDM_PARAMS['time_headway'], minimum_spacing=MAINA_IDM_PARAMS['minimum_spacing'])
        self.auxiliary_idm_params = IDMParams(desired_velocity= AUXA_MERGE_IDM_PARAMS['desired_velocity'], time_headway= AUXA_MERGE_IDM_PARAMS['time_headway'], minimum_spacing=AUXA_MERGE_IDM_PARAMS['minimum_spacing'])
        self.free_driving_params = IDMParams(desired_velocity=AUXA_IDM_PARAMS['desired_velocity'] , time_headway= AUXA_IDM_PARAMS['time_headway'], minimum_spacing=AUXA_IDM_PARAMS['minimum_spacing'])
        self.path_following_params = PathFollowingParams()
        self.aux_path_following_params = PathFollowingParams(lookahead_factor=1.5, max_heading_change=0.15)
        self.merge_params = MergeParams()
        self.dt = 0.1  # 时间步长 (s)
        self.prediction_horizon = 30  # 预测时域（步数）
        self.num_samples = 5  # 采样数量
    
