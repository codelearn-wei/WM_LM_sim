import numpy as np

        
# 新 Vehicle 类定义
class Vehicle:
    def __init__(self, x, y, v, a, heading, yaw_rate, length, width, **kwargs):
        self.x = x
        self.y = y
        self.v = v
        self.a = a
        self.length = length
        self.width = width
        self.yaw = heading
        self.heading = heading
        self.yaw_rate = yaw_rate
        self.attributes = kwargs  # 存储自定义属性
        self.strategy_params = {}  # 存储策略参数

    def set_strategy_params(self, **params):
        """设置或更新策略参数"""
        self.strategy_params.update(params)

    def get_state(self):
        """获取车辆当前状态"""
        state = {
            'x': self.x,
            'y': self.y,
            'v': self.v,
            'a': self.a,
            'heading': self.heading,
            'yaw_rate': self.yaw_rate,
            'length': self.length,
            'width': self.width
        }
        state.update(self.attributes)
        return state

# 运动学模型类
class VehicleKinematicsModel:
    def __init__(self, wheelbase_ratio=0.6):
        """初始化车辆运动学模型"""
        self.wheelbase_ratio = wheelbase_ratio
    
    def update(self, vehicle, acceleration, steering_angle, dt):
        """使用运动学自行车模型更新车辆状态"""
        v = vehicle.v  # 当前速度
        heading = vehicle.heading  # 当前航向角
        
        # 更新速度
        v_new = v + acceleration * dt
        v_new = max(0, v_new)  # 确保速度非负
        
        # 计算轴距
        wheelbase = vehicle.length * self.wheelbase_ratio
        
        # 更新航向角
        if abs(v) > 1e-3:  # 速度不接近零时更新航向角
            heading_new = heading + (v / wheelbase) * np.tan(steering_angle) * dt
        else:
            heading_new = heading
        
        # 更新位置
        dx = v_new * np.cos(heading_new) * dt
        dy = v_new * np.sin(heading_new) * dt
        
        # 更新车辆状态
        vehicle.x += dx
        vehicle.y += dy
        vehicle.heading = heading_new
        vehicle.v = v_new
        vehicle.a = acceleration
        vehicle.yaw = heading_new
        vehicle.yaw_rate = (heading_new - heading) / dt if dt > 0 else 0        
        
# class Vehicle:
#     def __init__(self, position, velocity, heading, length, width, **kwargs):
#         """
#         初始化车辆对象
        
#         Args:
#             position: 位置 [x, y]
#             velocity: 速度 [vx, vy]
#             heading: 航向角 (弧度)
#             length: 车辆长度
#             width: 车辆宽度
#             **kwargs: 其他自定义属性
#         """
#         self.position = np.array(position, dtype=float)  # 车辆中心位置 [x, y]
#         self.velocity = np.array(velocity, dtype=float)  # 速度 [vx, vy]
#         self.acceleration = np.array([0.0, 0.0], dtype=float)  # 加速度 [ax, ay]
#         self.heading = heading  # 航向角（弧度）
#         self.length = length  # 车辆长度
#         self.width = width  # 车辆宽度
#         self.attributes = kwargs  # 存储自定义属性
#         self.strategy_params = {}  # 存储策略参数
    
#     def set_strategy_params(self, **params):
#         """
#         设置或更新策略参数
        
#         Args:
#             **params: 策略参数键值对
#         """
#         self.strategy_params.update(params)
    
#     def get_state(self):
#         """
#         获取车辆当前状态
        
#         Returns:
#             dict: 包含所有基本属性和自定义属性的字典
#         """
#         state = {
#             'position': self.position.tolist(),
#             'velocity': self.velocity.tolist(),
#             'heading': self.heading,
#             'length': self.length,
#             'width': self.width
#         }
#         state.update(self.attributes)
#         return state
# class VehicleKinematicsModel:
#     """车辆运动学自行车模型类"""
    
#     def __init__(self, wheelbase_ratio=0.6):
#         """
#         初始化车辆运动学模型
        
#         参数:
#         wheelbase_ratio: 轴距与车长的比例，默认为0.6
#         """
#         self.wheelbase_ratio = wheelbase_ratio
    
#     def update(self, vehicle, acceleration, steering_angle, dt):
#         """
#         使用运动学自行车模型更新车辆状态
        
#         参数:
#         vehicle: Vehicle对象，包含车辆当前状态
#         acceleration: 纵向加速度
#         steering_angle: 前轮转角（弧度）
#         dt: 时间步长
#         """
#         # 获取当前状态
#         v = np.linalg.norm(vehicle.velocity)  # 速度大小
#         heading = vehicle.heading  # 航向角
        
#         # 更新速度大小（考虑加速度）
#         v_new = v + acceleration * dt
#         v_new = max(0, v_new)  # 确保速度非负
        
#         # 计算轴距
#         wheelbase = vehicle.length * self.wheelbase_ratio
        
#         # 更新航向角（考虑前轮转角）
#         if abs(v) > 1e-3:  # 只有当速度不接近零时才更新航向角
#             heading_new = heading + (v / wheelbase) * np.tan(steering_angle) * dt
#         else:
#             heading_new = heading
        
#         # 更新位置
#         dx = v_new * np.cos(heading_new) * dt
#         dy = v_new * np.sin(heading_new) * dt
        
#         # 更新车辆状态
#         vehicle.position[0] += dx
#         vehicle.position[1] += dy
#         vehicle.heading = heading_new
#         vehicle.velocity[0] = v_new * np.cos(heading_new)
#         vehicle.velocity[1] = v_new * np.sin(heading_new)
#         vehicle.acceleration[0] = acceleration * np.cos(heading_new)
#         vehicle.acceleration[1] = acceleration * np.sin(heading_new)