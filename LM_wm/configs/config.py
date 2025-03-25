from pathlib import Path
import torch
import torchvision
from transformers import AutoImageProcessor, AutoModel

# Data paths
MAP_PATH = "LM_data/map/DR_CHN_Merging_ZS.json"
RAW_DATA_PATH = "LM_data/data/DR_CHN_Merging_ZS"
TRAINING_DATA_DIR = "LM_wm/training_data"
VALIDATION_DATA_DIR = "LM_wm/validation_data"
MODEL_DIR = "LM_wm/models/checkpoints"

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
HISTORY_STEPS = 20
HIDDEN_DIM = 256
VALIDATION_FREQ = 5
EARLY_STOPPING_PATIENCE = 10

# Model parameters
ACTION_DIM = 30  # 10辆车 × 3个动作值

# Data generation parameters
NUM_WORKERS = 2
PREFETCH_FACTOR = 2

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
        self.batch_size = 32
        self.num_workers = 4
        self.history_steps = 5
        self.action_dim = 2
        
        # 模型相关配置
        self.hidden_dim = 256
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = IMAGE_SIZE
        
        # 训练相关配置
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.warmup_steps = 1000
        self.gradient_clip = 1.0
        self.early_stopping_patience = 10
        
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
        self.log_dir = "logs"
        self.save_dir = "checkpoints"
        self.log_interval = 10
        self.save_interval = 5
        
        # 验证配置
        self.val_interval = 1
        self.val_batch_size = 16
        
        # 测试配置
        self.test_batch_size = 16
        self.test_interval = 5
        
        # 可视化配置
        self.vis_interval = 50
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