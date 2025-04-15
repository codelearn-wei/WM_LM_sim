import numpy as np

class IDMFollowingModel:
    def __init__(self, 
                 desired_velocity=5.0,  # 期望速度 (m/s)
                 max_acceleration=3.0,   # 最大加速度 (m/s²)
                 comfortable_deceleration=4.0,  # 舒适减速度 (m/s²)
                 minimum_spacing=0.5,    # 最小安全间距 (m)
                 time_headway=1):      # 时间间隔 (s)
        """
        初始化IDM跟车模型参数
        
        参数:
        - desired_velocity: 驾驶员期望的目标速度
        - max_acceleration: 最大加速度
        - comfortable_deceleration: 舒适减速度
        - minimum_spacing: 最小安全间距
        - time_headway: 期望的时间间隔
        """
        self.v0 = desired_velocity
        self.a_max = max_acceleration
        self.b = comfortable_deceleration
        self.s0 = minimum_spacing
        self.T = time_headway

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

    def get_leader(self, vehicle_state, main_road_forward_vehicles):
        """
        从主道前车中找到最近的前车
        
        参数:
        - vehicle_state: 主车状态
        - main_road_forward_vehicles: 主道前车列表
        
        返回:
        - 最近的前车，如果没有前车则返回None
        """
        if not main_road_forward_vehicles:
            return None

        # 计算每辆前车与主车的距离
        distances = []
        for leader in main_road_forward_vehicles:
            distance = np.linalg.norm(
                np.array(leader['position']) - np.array(vehicle_state['position'])
            ) - leader['length']
            distances.append(distance)

        # 找到最近的前车
        if distances:
            closest_leader_index = np.argmin(distances)
            return main_road_forward_vehicles[closest_leader_index]
        
        return None

def idm_follow_leader(vehicle_state, 
                  neighbors_vehicle_categories=None,
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

    # 获取主道靠前车辆
    main_road_forward_vehicles = neighbors_vehicle_categories.get('主道靠前', [])

    # 创建IDM模型实例
    if idm_params is None:
        idm_params = {}
        
     # TODO:确定不同的IDM参数策略   
    idm_model = IDMFollowingModel(**idm_params)

    # 找到最近的前车
    leader = idm_model.get_leader(vehicle_state, main_road_forward_vehicles)

    # 如果没有前车，返回0加速度（保持当前速度）
    if leader is None:
        return 0.0

    # 计算并返回加速度
    return idm_model.calculate_acceleration(vehicle_state, leader)


# 示例使用
def example_usage():
    # 创建带自定义参数的IDM模型
    custom_idm = IDMFollowingModel(
        desired_velocity=22.0,    # 更高的期望速度
        max_acceleration=2.5,     # 更激进的加速
        comfortable_deceleration=2.8,  # 略低的减速度
        minimum_spacing=1.5,      # 更小的安全间距
        time_headway=1.2          # 更短的时间间隔
    )

    # 模拟车辆状态（示例）
    ego_vehicle = {
        'position': [100, 0],
        'velocity': [15, 0],
        'length': 4.5
    }

    leader_vehicle = {
        'position': [110, 0],
        'velocity': [14, 0],
        'length': 4.5
    }

    # 计算加速度
    acceleration = custom_idm.calculate_acceleration(ego_vehicle, leader_vehicle)
    print(f"计算得到的加速度: {acceleration} m/s²")

# 如果直接运行此脚本，执行示例
if __name__ == "__main__":
    example_usage()