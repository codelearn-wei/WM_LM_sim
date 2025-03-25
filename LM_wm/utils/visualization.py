import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_predictions(pred_images, target_images=None, num_samples=4):
    """
    可视化预测结果
    
    Args:
        pred_images (torch.Tensor): 预测的图像 [B, C, H, W]
        target_images (torch.Tensor, optional): 目标图像 [B, C, H, W]
        num_samples (int): 要显示的样本数量
    """
    # 确保输入是4D张量
    if len(pred_images.shape) == 3:
        pred_images = pred_images.unsqueeze(0)
    
    # 限制显示的样本数量
    pred_images = pred_images[:num_samples]
    if target_images is not None:
        target_images = target_images[:num_samples]
    
    # 转换为numpy数组，注意添加detach()
    pred_images = pred_images.detach().cpu().numpy()
    if target_images is not None:
        target_images = target_images.detach().cpu().numpy()
    
    # 创建子图
    n_samples = len(pred_images)
    n_cols = 2 if target_images is not None else 1
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4*n_cols, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # 显示预测图像
        pred_img = pred_images[i].transpose(1, 2, 0)
        axes[i, 0].imshow(pred_img)
        axes[i, 0].set_title('Predicted')
        axes[i, 0].axis('off')
        
        # 如果有目标图像，显示目标图像
        if target_images is not None:
            target_img = target_images[i].transpose(1, 2, 0)
            axes[i, 1].imshow(target_img)
            axes[i, 1].set_title('Target')
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    return fig 