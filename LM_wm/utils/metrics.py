import torch
import numpy as np
from configs.config import MODEL_CONFIG

def compute_psnr(pred_image, target_image):
    """
    计算峰值信噪比 (PSNR)
    
    Args:
        pred_image (torch.Tensor): 预测图像 [C, H, W]
        target_image (torch.Tensor): 目标图像 [C, H, W]
    
    Returns:
        float: PSNR值
    """
    # 确保输入是张量
    if not isinstance(pred_image, torch.Tensor):
        pred_image = torch.tensor(pred_image)
    if not isinstance(target_image, torch.Tensor):
        target_image = torch.tensor(target_image)
    
    # 计算MSE
    mse = torch.mean((pred_image - target_image) ** 2)
    
    # 计算PSNR
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    
    return psnr.item()

def compute_ssim(pred_image, target_image, window_size=11, size_average=True):
    """
    计算结构相似性指数 (SSIM)
    
    Args:
        pred_image (torch.Tensor): 预测图像 [C, H, W]
        target_image (torch.Tensor): 目标图像 [C, H, W]
        window_size (int): 窗口大小
        size_average (bool): 是否返回平均值
    
    Returns:
        float: SSIM值
    """
    # 确保输入是张量
    if not isinstance(pred_image, torch.Tensor):
        pred_image = torch.tensor(pred_image)
    if not isinstance(target_image, torch.Tensor):
        target_image = torch.tensor(target_image)
    
    # 创建高斯窗口
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    # 计算SSIM
    channel = pred_image.size(0)
    window = create_window(window_size, channel)
    
    if pred_image.is_cuda:
        window = window.cuda(pred_image.get_device())
    window = window.type_as(pred_image)
    
    mu1 = torch.nn.functional.conv2d(pred_image.unsqueeze(0), window, padding=window_size//2, groups=channel)
    mu2 = torch.nn.functional.conv2d(target_image.unsqueeze(0), window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.conv2d(pred_image.unsqueeze(0) * pred_image.unsqueeze(0), window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(target_image.unsqueeze(0) * target_image.unsqueeze(0), window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(pred_image.unsqueeze(0) * target_image.unsqueeze(0), window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()

def compute_metrics(pred_image, target_image):
    """
    计算所有评估指标
    
    Args:
        pred_image (torch.Tensor): 预测图像 [C, H, W]
        target_image (torch.Tensor): 目标图像 [C, H, W]
    
    Returns:
        dict: 包含所有指标的字典
    """
    metrics = {
        'psnr': compute_psnr(pred_image, target_image),
        'ssim': compute_ssim(pred_image, target_image)
    }
    
    return metrics

def compute_batch_metrics(pred_images, target_images):
    """
    计算批次中所有图像的评估指标
    
    Args:
        pred_images (torch.Tensor): 预测图像 [B, C, H, W]
        target_images (torch.Tensor): 目标图像 [B, C, H, W]
    
    Returns:
        dict: 包含所有指标的平均值和标准差的字典
    """
    batch_size = pred_images.size(0)
    metrics_list = []
    
    for i in range(batch_size):
        metrics = compute_metrics(pred_images[i], target_images[i])
        metrics_list.append(metrics)
    
    # 计算平均值和标准差
    avg_metrics = {}
    std_metrics = {}
    
    for metric in metrics_list[0].keys():
        values = [m[metric] for m in metrics_list]
        avg_metrics[metric] = np.mean(values)
        std_metrics[metric] = np.std(values)
    
    return {
        'average': avg_metrics,
        'std': std_metrics
    } 