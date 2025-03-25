import torch
from pathlib import Path
from LM_wm.datasets.training_dataset import LMTrainingDataset
from LM_wm.utils.image_utils import preprocess_image_for_model, visualize_processed_image
from LM_wm.configs.config import TRAINING_DATA_DIR, HISTORY_STEPS

def test_image_processing():
    """
    测试图像处理效果
    """
    # 创建输出目录
    output_dir = Path("LM_wm/test/test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 加载数据集
    dataset = LMTrainingDataset(
        data_dir=TRAINING_DATA_DIR,
        history_steps=HISTORY_STEPS
    )
    
    # 获取第一个样本
    sample = dataset[0]
    bev_frames = sample['bev_frames']  # (history_steps, C, H, W)
    
    # 处理第一帧图像
    original_frame = bev_frames[0]  # (C, H, W)
    processed_frame = preprocess_image_for_model(original_frame)
    
    # 可视化并保存结果
    visualize_processed_image(
        original_frame,
        processed_frame,
        save_path=output_dir / "processed_image_comparison.png"
    )
    
    # 打印图像信息
    # print(f"原始图像大小: {original_frame.shape}")
    # print(f"处理后图像大小: {processed_frame.shape}")
    # print(f"图像已保存到: {output_dir / 'processed_image_comparison.png'}")

if __name__ == "__main__":
    test_image_processing() 