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
IMAGE_SIZE = (224, 224)  # DINOv2 的标准输入尺寸

# Create necessary directories
Path(TRAINING_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(VALIDATION_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True) 
# 在数据加载器中预处理图像
def preprocess_image(image):
    return torchvision.transforms.Resize(IMAGE_SIZE)(image) 