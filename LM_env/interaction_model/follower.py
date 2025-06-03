import numpy as np

class IDMFollowingModel:
    def __init__(self, 
                 v0=2.0,  # 期望速度 (m/s)
                 a_max=2.0,   # 最大加速度 (m/s²)
                 b=3.0,  # 舒适减速度 (m/s²)
                 s0=1,    # 最小安全间距 (m)
                 T=1):      # 时间间隔 (s)
        """
        初始化IDM跟车模型参数
        
        参数:
        - desired_velocity: 驾驶员期望的目标速度
        - max_acceleration: 最大加速度
        - comfortable_deceleration: 舒适减速度
        - minimum_spacing: 最小安全间距
        - time_headway: 期望的时间间隔
        """
        self.v0 = v0
        self.a_max = a_max
        self.b = b
        self.s0 = s0
        self.T = T

    def calculate_acceleration(self, ego_vehicle, leader_vehicle):
        """
        根据IDM模型计算加速度
        
        参数:
        - ego_vehicle: 主车状态字典
        - leader_vehicle: 前车状态字典
        
        返回:
        - 计算得到的加速度 (m/s²)
        """
        # 提取主车和前车的速度和位置
        v_ego = np.linalg.norm(ego_vehicle['velocity'])
        v_leader = np.linalg.norm(leader_vehicle['velocity'])
        
        # 计算当前间距
        s = np.linalg.norm(np.array(leader_vehicle['position']) - np.array(ego_vehicle['position'])) \
            - leader_vehicle['length']

        # 理想间距
        s_star = self.s0 + max(0, v_ego * self.T + 
                                (v_ego * (v_ego - v_leader)) / 
                                (2 * np.sqrt(self.a_max * self.b)))

        # IDM加速度计算公式
        acceleration = self.a_max * (1 - (v_ego / self.v0)**4 - 
                                     (s_star / s)**2)

        # 限制加速度范围
        return np.clip(acceleration, -self.b, self.a_max)
    
def idm_follow_leader(vehicle_state, 
                  leader=None,
                  idm_params=None):
    """
    跟车控制主函数
    
    参数:
    - vehicle_state: 主车状态
    - neighbors_vehicle_categories: 邻近车辆分类（可选）
    - idm_params: IDM模型参数（可选）
    
    返回:
    - 计算得到的加速度
    """
    if idm_params is None:
        idm_params = {}
        
     # TODO:确定不同的IDM参数策略   
    idm_model = IDMFollowingModel(**idm_params)

    if leader is None:
        return 0.0

    # 计算并返回加速度
    return idm_model.calculate_acceleration(vehicle_state, leader)

