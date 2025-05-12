import os
from dataclasses import dataclass
import torch

@dataclass
class PolicyConfig:
    # 基础设置
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # 数据设置
    data_dir: str = "LM_wm/training_data"
    history_steps: int = 5
    action_dim: int = 30
    num_workers = 8
    val_batch_size = 16
    
    # 模型设置
    hidden_dim: int = 512
    dino_dim: int = 768  # DINOv2输出维度
    
    # 训练设置
    val_interval: int = 1  # 每隔几个epoch验证一次
    save_interval: int = 2  # 每隔几个epoch保存一次检查点
    early_stopping_patience: int = 4
    log_interval: int = 10  # 每隔几个batch打印一次日志
    
    # 路径设置
    base_log_dir: str = "LM_wm/logs"
    log_dir: str = None
    save_dir: str = None
    
    def __post_init__(self):
        """初始化后处理，设置路径"""
        # 设置日志和保存路径
        self.log_dir = os.path.join(self.base_log_dir, "policy")
        self.save_dir = os.path.join(self.log_dir, "checkpoints")
        
        # 创建必要的目录
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True) 