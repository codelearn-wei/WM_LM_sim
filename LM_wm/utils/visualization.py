import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm
from pylab import mpl
import datetime
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
def visualize_predictions(pred_images, target_images, save_path=None):
    """
    可视化预测结果和目标图像的对比
    
    Args:
        pred_images (torch.Tensor): 预测图像 [B, C, H, W]
        target_images (torch.Tensor): 目标图像 [B, C, H, W]
        save_path (str, optional): 保存路径
    
    Returns:
        matplotlib.figure.Figure: 可视化图像
    """
    # 确保输入是4D张量
    if len(pred_images.shape) == 3:
        pred_images = pred_images.unsqueeze(0)
    if len(target_images.shape) == 3:
        target_images = target_images.unsqueeze(0)
    
    # 计算要显示的样本数量
    num_samples = min(4, pred_images.size(0))
    
    # 创建图像网格
    fig = plt.figure(figsize=(15, 5))
    
    for i in range(num_samples):
        # 预测图像
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(pred_images[i].cpu().permute(1, 2, 0))
        plt.title(f'Prediction {i+1}')
        plt.axis('off')
    
    # 添加时间戳和批次信息
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.suptitle(f'Validation Results - {timestamp}', fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 如果提供了保存路径，保存图像
    if save_path:
        plt.savefig(save_path)
    
    return fig

def visualize_attention_maps(image, attention_map, save_path=None):
    """
    可视化注意力图

    Args:
        image (torch.Tensor): 输入图像 [B, C, H, W] 或 [C, H, W]
        attention_map (torch.Tensor): 注意力图 [B, H, W] 或 [H, W]
        save_path (str): 保存路径，如果为None则显示
    """
    # 确保输入是合适的维度
    if image.dim() == 4:
        image = image[0]  # 取第一个样本
    if attention_map.dim() == 3:
        attention_map = attention_map[0]  # 取第一个样本
        
    # 转换到numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().detach().numpy()
        image = np.clip(image, 0, 1)
    
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().detach().numpy()
    
    # 将注意力图调整到与图像相同大小
    H, W = image.shape[:2]
    attention_map = cv2.resize(attention_map, (W, H))
    
    # 创建热图
    attention_map_color = cm.jet(attention_map)[:, :, :3]  # 去掉alpha通道
    
    # 创建叠加图像
    alpha = 0.6
    overlay = alpha * attention_map_color + (1-alpha) * image
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(attention_map, cmap='jet')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return fig

def visualize_weighted_regions(image, save_path=None):
    """
    可视化不同区域的权重分布
    
    Args:
        image (torch.Tensor): 输入图像 [B, C, H, W]
        save_path (str, optional): 保存路径
    
    Returns:
        matplotlib.figure.Figure: 可视化图像
    """
    # 确保输入是4D张量
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    # 创建图像网格
    fig = plt.figure(figsize=(15, 5))
    
    # 显示原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image[0].cpu().permute(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    
    # 计算像素均值和标准差
    pixel_mean = torch.mean(image[0], dim=0)
    pixel_std = torch.std(image[0], dim=0)
    
    # 创建道路掩码
    road_mask = torch.logical_and(
        pixel_mean > 0.5,  # 较亮
        pixel_std > 0.03  # 颜色不均匀
    )
    
    # 创建车辆掩码（红色和蓝色区域）
    is_red = torch.logical_and(
        image[0, 0] > 0.6,  # 红色通道高
        torch.max(image[0, 1:], dim=0)[0] < 0.4  # 绿色和蓝色通道低
    )
    
    is_blue = torch.logical_and(
        image[0, 2] > 0.6,  # 蓝色通道高
        torch.max(image[0, :2], dim=0)[0] < 0.4  # 红色和绿色通道低
    )
    
    vehicle_mask = torch.logical_or(is_red, is_blue)
    
    # 显示道路区域
    plt.subplot(1, 3, 2)
    road_vis = image[0].cpu().permute(1, 2, 0).numpy()
    road_vis[~road_mask.cpu()] = 0
    plt.imshow(road_vis)
    plt.title('Road Regions')
    plt.axis('off')
    
    # 显示车辆区域
    plt.subplot(1, 3, 3)
    vehicle_vis = image[0].cpu().permute(1, 2, 0).numpy()
    vehicle_vis[~vehicle_mask.cpu()] = 0
    plt.imshow(vehicle_vis)
    plt.title('Vehicle Regions')
    plt.axis('off')
    
    # 添加时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.suptitle(f'Weighted Regions Visualization - {timestamp}', fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 如果提供了保存路径，保存图像
    if save_path:
        plt.savefig(save_path)
    
    return fig

def visualize_weight_schedule(config, save_path=None):
    """
    可视化权重调度曲线，展示随着训练进行权重如何变化。
    适配简化后的权重配置结构。

    Args:
        config: 配置对象，包含weight_schedule
        save_path: 保存路径，如果为None则显示
    """
    if not config.use_progressive_weights or not config.weight_schedule:
        print("没有配置渐进式权重调整")
        return
    
    # 提取权重数据和关键epoch点
    epochs = sorted(config.weight_schedule.keys())
    max_epoch = max(epochs)
    
    # 确保包含最终epoch
    if max_epoch < config.num_epochs - 1:
        epochs.append(config.num_epochs - 1)
    
    # 创建完整的epoch列表
    all_epochs = list(range(config.num_epochs))
    
    # 获取所有需要绘制的权重类型
    weight_types = list(config.weights["initial"].keys())
    
    # 初始化权重数据收集字典
    weight_values = {key: [] for key in weight_types}
    
    # 线性插值函数
    def interpolate_weight(start_epoch, end_epoch, start_weight, end_weight, current_epoch):
        if current_epoch <= start_epoch:
            return start_weight
        if current_epoch >= end_epoch:
            return end_weight
        progress = (current_epoch - start_epoch) / (end_epoch - start_epoch)
        return start_weight + progress * (end_weight - start_weight)
    
    # 对每个epoch计算所有权重值
    for epoch in all_epochs:
        # 初始化为初始和最终权重
        prev_epoch = config.weight_epochs["start"]
        next_epoch = config.weight_epochs["end"]
        prev_weights = config.weight_schedule.get(prev_epoch, config.weights["initial"])
        next_weights = config.weight_schedule.get(next_epoch, config.weights["final"])
        
        # 找到当前epoch所在的区间
        for i in range(len(epochs) - 1):
            if epochs[i] <= epoch < epochs[i + 1]:
                prev_epoch = epochs[i]
                # 确保next_epoch存在于weight_schedule中
                if epochs[i + 1] in config.weight_schedule:
                    next_epoch = epochs[i + 1]
                    next_weights = config.weight_schedule[next_epoch]
                else:
                    # 使用最近的一个已定义的权重点
                    next_epoch = epochs[i + 1]  # 保留epoch值用于插值
                    # 从最后一个定义的权重获取值
                    last_defined_epoch = max([e for e in epochs if e in config.weight_schedule and e <= next_epoch], default=prev_epoch)
                    next_weights = config.weight_schedule.get(last_defined_epoch, config.weights["final"])
                
                prev_weights = config.weight_schedule[prev_epoch]
                break
        
        # 如果epoch超出定义范围，使用最终权重
        if epoch > config.weight_epochs["end"]:
            prev_epoch = config.weight_epochs["end"]
            next_epoch = config.num_epochs - 1
            prev_weights = config.weight_schedule.get(prev_epoch, config.weights["final"])
            next_weights = config.weights["final"]
        
        # 计算并保存每种权重的值
        for weight_type in weight_types:
            start_value = prev_weights.get(weight_type, config.weights["initial"].get(weight_type, 1.0))
            end_value = next_weights.get(weight_type, config.weights["final"].get(weight_type, 1.0))
            
            weight_value = interpolate_weight(
                prev_epoch, next_epoch, 
                start_value, end_value,
                epoch
            )
            weight_values[weight_type].append(weight_value)
    
    # 创建图表
    plt.figure(figsize=(12, 7))
    
    # 绘制权重变化曲线
    colors = {
        "vehicle": "blue",
        "road": "green", 
        "boundary": "red",
        "other_losses": "magenta"
    }
    
    # 中文标签映射
    labels = {
        "vehicle": "车辆区域权重",
        "road": "道路区域权重", 
        "boundary": "边界区域权重",
        "other_losses": "其他损失权重"
    }
    
    # 绘制所有权重曲线
    for weight_type in weight_types:
        plt.plot(
            all_epochs, weight_values[weight_type], 
            color=colors.get(weight_type, "black"), 
            linewidth=2, 
            label=labels.get(weight_type, weight_type)
        )
    
    # 标记关键epoch点
    for epoch in epochs:
        if epoch in config.weight_schedule:
            plt.axvline(x=epoch, color='gray', linestyle='--', alpha=0.5)
            
            # 添加标记文本
            max_weight = max([max(values) for values in weight_values.values()])
            plt.text(
                epoch, max_weight * 1.05, 
                f"Epoch {epoch}", 
                horizontalalignment='center',
                verticalalignment='bottom',
                rotation=90
            )
    
    # 设置图表属性
    plt.title('渐进式权重调整曲线')
    plt.xlabel('Epoch')
    plt.ylabel('权重值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 添加配置摘要信息
    schedule_text = "权重调度关键点:\n"
    for epoch, weights in config.weight_schedule.items():
        # 生成每个关键点的权重信息
        weight_info = []
        for weight_type in weight_types:
            if weight_type in weights:
                weight_info.append(f"{labels.get(weight_type, weight_type)}={weights[weight_type]:.2f}")
        
        schedule_text += f"Epoch {epoch}: " + ", ".join(weight_info) + "\n"
    
    plt.figtext(0.5, 0.01, schedule_text, ha='center', fontsize=9,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.98])
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"权重调度可视化已保存到 {save_path}")
    else:
        plt.show()
    
    return plt.gcf() 