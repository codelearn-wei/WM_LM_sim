import numpy as np

class VehicleKinematicsModel:
    """车辆运动学自行车模型类"""
    
    def __init__(self, wheelbase_ratio=0.6):
        """
        初始化车辆运动学模型
        
        参数:
        wheelbase_ratio: 轴距与车长的比例，默认为0.6
        """
        self.wheelbase_ratio = wheelbase_ratio
    
    def update(self, vehicle, acceleration, steering_angle, dt):
        """
        使用运动学自行车模型更新车辆状态
        
        参数:
        vehicle: Vehicle对象，包含车辆当前状态
        acceleration: 纵向加速度
        steering_angle: 前轮转角（弧度）
        dt: 时间步长
        """
        # 获取当前状态
        v = np.linalg.norm(vehicle.velocity)  # 速度大小
        heading = vehicle.heading  # 航向角
        
        # 更新速度大小（考虑加速度）
        v_new = v + acceleration * dt
        v_new = max(0, v_new)  # 确保速度非负
        
        # 计算轴距
        wheelbase = vehicle.length * self.wheelbase_ratio
        
        # 更新航向角（考虑前轮转角）
        if abs(v) > 1e-3:  # 只有当速度不接近零时才更新航向角
            heading_new = heading + (v / wheelbase) * np.tan(steering_angle) * dt
        else:
            heading_new = heading
        
        # 更新位置
        dx = v_new * np.cos(heading_new) * dt
        dy = v_new * np.sin(heading_new) * dt
        
        # 更新车辆状态
        vehicle.position[0] += dx
        vehicle.position[1] += dy
        vehicle.heading = heading_new
        vehicle.velocity[0] = v_new * np.cos(heading_new)
        vehicle.velocity[1] = v_new * np.sin(heading_new)
        vehicle.acceleration[0] = acceleration * np.cos(heading_new)
        vehicle.acceleration[1] = acceleration * np.sin(heading_new)