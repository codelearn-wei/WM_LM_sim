from pathlib import Path
import torch
import torchvision

# Data paths
MAP_PATH = "LM_data/map/DR_CHN_Merging_ZS.json"
RAW_DATA_PATH = "LM_data/data/DR_CHN_Merging_ZS"
TRAINING_DATA_DIR = "LM_wm/training_data"
VALIDATION_DATA_DIR = "LM_wm/validation_data"
MODEL_DIR = "LM_wm/models/checkpoints"

# Training parameters
# BATCH_SIZE = 16
# NUM_EPOCHS = 50
# LEARNING_RATE = 1e-4
# HISTORY_STEPS = 20
# HIDDEN_DIM = 256
# VALIDATION_FREQ = 5
# EARLY_STOPPING_PATIENCE = 5

# Model parameters
# ACTION_DIM = 30  # 10辆车 × 3个动作值

# Data generation parameters
# NUM_WORKERS = 2
# PREFETCH_FACTOR = 2

# GPU settings
USE_GPU = True
GPU_ID = 0
CUDA_DEVICE = f"cuda:{GPU_ID}" if USE_GPU and torch.cuda.is_available() else "cpu"

# 性能优化设置
PIN_MEMORY = True
USE_AMP = True
USE_CUDNN_BENCHMARK = True
GRADIENT_ACCUMULATION_STEPS = 4
# 图像处理配置
IMAGE_SIZE = (224, 224)  # DINOv2的输入尺寸

# Create necessary directories
Path(TRAINING_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(VALIDATION_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True) 
# 在数据加载器中预处理图像
def preprocess_image(image):
    return torchvision.transforms.Resize(IMAGE_SIZE)(image)

class Config:
    def __init__(self):
        # 数据相关配置
        self.data_dir = "LM_wm/training_data"
        self.batch_size = 64
        self.num_workers = 8
        self.history_steps = 10
        self.max_vehicles = 10 ## 定义最大车辆数量
        self.action_num = 3 ## 定义动作的维度
        self.action_dim = self.max_vehicles * self.action_num ## 定义动作的维度
        
        # 模型相关配置
        self.hidden_dim = 256
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = IMAGE_SIZE
        
        # 训练相关配置
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.warmup_steps = 500
        self.gradient_clip = 1.0
        self.early_stopping_patience = 5
        
        # 添加掩码和区域关注配置
        self.focus_on_road = True  # 是否强制模型关注道路区域，忽略深灰色边界区域
        
        # 渐进式权重调整配置 - 简化版
        self.use_progressive_weights = True  # 是否使用渐进式权重调整
        
        # 1. 权重起始值和终止值
        self.weights = {
            # 起始值
            "initial": {
                "vehicle": 5.0,      # 车辆区域初始权重 
                "road": 3.0,         # 道路区域初始权重
                "boundary": 0.2,     # 边界区域初始权重
                "other_losses": 0.5  # 其他损失初始权重
            },
            # 终止值
            "final": {
                "vehicle": 40.0,     # 车辆区域最终权重
                "road": 10.0,        # 道路区域最终权重
                "boundary": 0.01,    # 边界区域最终权重
                "other_losses": 0.05 # 其他损失最终权重
            }
        }
        
        # 2. 线性变化的起止epoch
        self.weight_epochs = {
            "start": 0,  # 开始渐进式调整的epoch
            "end": 35    # 结束渐进式调整的epoch
        }
        
        # 为保持向后兼容性，设置对应的属性
        self.initial_vehicle_weight = self.weights["initial"]["vehicle"]
        self.initial_road_weight = self.weights["initial"]["road"]
        self.initial_boundary_weight = self.weights["initial"]["boundary"]
        self.initial_other_losses_weight = self.weights["initial"]["other_losses"]
        
        self.final_vehicle_weight = self.weights["final"]["vehicle"]
        self.final_road_weight = self.weights["final"]["road"]
        self.final_boundary_weight = self.weights["final"]["boundary"]
        self.final_other_losses_weight = self.weights["final"]["other_losses"]
        
        self.linear_start_epoch = self.weight_epochs["start"]
        self.linear_end_epoch = self.weight_epochs["end"]
        
        # 生成渐进式权重调度字典 (采用线性变化)
        self.weight_schedule = self._generate_linear_weight_schedule()
        
        # 训练模式配置
        self.train_mode = "feature"  # 可选: "feature" 或 "image"
        
        # 损失函数权重配置
        self.loss_weights = {
            "feature": {
                "feature_loss": 1.0
            },
            "image": {
                "reconstruction_loss": 1.0,
                "perceptual_loss": 0.5,
                "feature_loss": 0.3
            }
        }
        
        # 日志和保存配置
        self.log_dir = "LM_wm/logs"
        self.save_dir = "LM_wm/checkpoints"
        self.log_interval = 10
        self.save_interval = 5
        
        # 验证配置
        self.val_interval = 1
        self.val_batch_size = 16
        
        # 测试配置
        self.test_batch_size = 16
        self.test_interval = 5
        
        # 可视化配置
        self.vis_interval = 10
        self.num_vis_samples = 4
        
        # 数据生成配置
        self.num_trajectories = 1000
        self.trajectory_length = 50
        self.num_workers_gen = 4
        
        # 其他配置
        self.seed = 42
        self.debug = False
        
        # DINOv2模型配置
        self.dino_model_name = "facebook/dinov2-base"
        self.dino_output_dim = 768  # dinov2-base的输出维度
        
        # 图像解码器配置
        self.decoder_channels = [256, 128, 64, 32, 16]
        self.decoder_output_channels = 3  # RGB图像
        
        # 优化器配置
        self.optimizer = {
            "type": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "betas": (0.9, 0.999)
        }
        
        # 学习率调度器配置
        self.scheduler = {
            "type": "cosine",
            "T_max": 100,
            "eta_min": 1e-6
        }
        
        # 数据增强配置
        self.augmentation = {
            "enabled": True,
            "random_flip": True,
            "random_rotation": True,
            "color_jitter": True
        }
        
        # 多GPU训练配置
        self.multi_gpu = {
            "enabled": torch.cuda.device_count() > 1,
            "strategy": "ddp"  # 可选: "dp" 或 "ddp"
        }
        
        # 混合精度训练配置
        self.amp = {
            "enabled": True,
            "opt_level": "O1"  # 可选: "O0", "O1", "O2", "O3"
        }
        
        # 梯度累积配置
        self.gradient_accumulation = {
            "enabled": True,
            "steps": 4
        }
    
    def _generate_linear_weight_schedule(self):
        """生成线性变化的权重调度字典 - 简化版"""
        if not self.use_progressive_weights:
            return {}
            
        # 计算渐进式变化的epoch数
        epochs_count = self.weight_epochs["end"] - self.weight_epochs["start"]
        if epochs_count <= 0:
            # 如果起止范围无效，只使用最终权重
            return {0: self.weights["final"]}
            
        # 确保最终epoch不超过训练总轮数
        end_epoch = min(self.weight_epochs["end"], self.num_epochs - 1)
        
        # 创建权重调度字典
        schedule = {}
        
        # 添加起始点
        schedule[self.weight_epochs["start"]] = self.weights["initial"].copy()
        
        # 添加结束点
        schedule[end_epoch] = self.weights["final"].copy()
        
        # 添加中间检查点
        checkpoint_interval = 10
        if epochs_count > checkpoint_interval * 2:
            for epoch in range(self.weight_epochs["start"] + checkpoint_interval, 
                              end_epoch, checkpoint_interval):
                # 计算进度比例
                progress = (epoch - self.weight_epochs["start"]) / epochs_count
                
                # 为所有权重参数进行线性插值
                checkpoint_weights = {}
                for key in self.weights["initial"].keys():
                    initial_value = self.weights["initial"][key]
                    final_value = self.weights["final"][key]
                    checkpoint_weights[key] = initial_value + progress * (final_value - initial_value)
                
                schedule[epoch] = checkpoint_weights
        
        # 确保各个权重配置是独立的对象，避免引用问题
        for epoch in schedule.keys():
            if isinstance(schedule[epoch], dict):
                schedule[epoch] = schedule[epoch].copy()
                
        return schedule 