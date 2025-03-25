import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

def maintain_aspect_ratio_resize(image, target_size=(224, 224)):
    """
    调整图像大小，保持宽高比，并填充到目标大小
    """
    if isinstance(image, torch.Tensor):
        # 如果输入是 PyTorch tensor，转换为 numpy array
        if image.dim() == 4:  # (B, C, H, W)
            image = image.squeeze(0)
        # 确保值在0-255范围内
        image = image.permute(1, 2, 0).cpu().numpy() * 255
        image = image.astype(np.uint8)
    
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # 计算目标高度，保持原始宽高比
    new_w = target_w
    new_h = int(h * (target_w / w))
    
    # 打印调试信息
    # print(f"原始图像大小: {h}x{w}")
    # print(f"调整后大小: {new_h}x{new_w}")
    # print(f"图像值范围: {image.min()}-{image.max()}")
    
    # 调整图像大小
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 创建目标大小的画布（灰色背景）
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 128
    
    # 计算垂直偏移量，使图像居中
    y_offset = (target_h - new_h) // 2
    
    # 将调整后的图像放在画布中心
    canvas[y_offset:y_offset+new_h, :] = resized
    
    return canvas

def preprocess_image_for_model(image, target_size=(224, 224)):
    """
    预处理图像用于模型输入
    """
    # 保持宽高比调整大小
    resized = maintain_aspect_ratio_resize(image, target_size)
    
    # 转换为 PIL Image
    if isinstance(resized, np.ndarray):
        resized = Image.fromarray(resized)
    
    # 只转换为张量，不进行归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 应用转换
    tensor = transform(resized)
    
    return tensor

def visualize_processed_image(original_image, processed_image, save_path=None):
    """
    可视化原始图像和处理后的图像
    """
    plt.figure(figsize=(12, 6))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    if isinstance(original_image, torch.Tensor):
        if original_image.dim() == 4:
            original_image = original_image.squeeze(0)
        original_image = original_image.permute(1, 2, 0).cpu().numpy()
        # 确保值在0-1范围内
        original_image = np.clip(original_image, 0, 1)
    plt.imshow(original_image)
    plt.title(f'Original Image {original_image.shape}')
    
    # 显示处理后的图像
    plt.subplot(1, 2, 2)
    if isinstance(processed_image, torch.Tensor):
        if processed_image.dim() == 4:
            processed_image = processed_image.squeeze(0)
        processed_image = processed_image.permute(1, 2, 0).cpu().numpy()
        # 确保值在0-1范围内
        processed_image = np.clip(processed_image, 0, 1)
    plt.imshow(processed_image)
    plt.title(f'Processed Image {processed_image.shape}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()