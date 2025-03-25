from LM_wm.datasets.training_dataset import LMTrainingDataset
from LM_wm.configs.config import TRAINING_DATA_DIR, HISTORY_STEPS

def check_data_dimensions():
    # 加载数据集
    dataset = LMTrainingDataset(
        data_dir=TRAINING_DATA_DIR,
        history_steps=HISTORY_STEPS
    )
    
    # 获取第一个样本
    sample = dataset[0]
    
    # 打印原始图像维度
    bev_frames = sample['bev_frames']
    print(f"原始图像批次维度: {bev_frames.shape}")
    
    # 如果需要，还可以检查单个图像
    first_frame = bev_frames[0]
    print(f"单帧图像维度: {first_frame.shape}")
    
    # 打印目标图像维度
    next_frame = sample['next_frame']
    print(f"目标图像维度: {next_frame.shape}")

if __name__ == "__main__":
    check_data_dimensions()