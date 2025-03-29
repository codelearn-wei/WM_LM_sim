class Vehicle:
    def __init__(self, position, velocity, heading, length, width, **kwargs):
        """
        初始化车辆对象
        
        Args:
            position: 位置 [x, y]
            velocity: 速度 [vx, vy]
            heading: 航向角 (弧度)
            length: 车辆长度
            width: 车辆宽度
            **kwargs: 其他自定义属性
        """
        self.position = np.array(position, dtype=float)  # 车辆中心位置 [x, y]
        self.velocity = np.array(velocity, dtype=float)  # 速度 [vx, vy]
        self.acceleration = np.array([0.0, 0.0], dtype=float)  # 加速度 [ax, ay]
        self.heading = heading  # 航向角（弧度）
        self.length = length  # 车辆长度
        self.width = width  # 车辆宽度
        self.attributes = kwargs  # 存储自定义属性
        self.strategy_params = {}  # 存储策略参数
    
    def set_strategy_params(self, **params):
        """
        设置或更新策略参数
        
        Args:
            **params: 策略参数键值对
        """
        self.strategy_params.update(params)
    
    def get_state(self):
        """
        获取车辆当前状态
        
        Returns:
            dict: 包含所有基本属性和自定义属性的字典
        """
        state = {
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'heading': self.heading,
            'length': self.length,
            'width': self.width
        }
        state.update(self.attributes)
        return state