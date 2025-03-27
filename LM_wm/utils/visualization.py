import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
def visualize_predictions(pred_images, target_images, max_samples=4):
    """
    可视化预测结果和目标图像

    Args:
        pred_images (torch.Tensor): 预测图像 [B, C, H, W]
        target_images (torch.Tensor): 目标图像 [B, C, H, W]
        max_samples (int): 最大可视化样本数

    Returns:
        plt.Figure: Matplotlib figure
    """
    batch_size = min(pred_images.shape[0], max_samples)
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 3 * batch_size))
    
    # 单样本情况
    if batch_size == 1:
        axes = axes.reshape(1, 2)
    
    for i in range(batch_size):
        # 预测图像
        pred = pred_images[i].permute(1, 2, 0).cpu().detach().numpy()
        pred = np.clip(pred, 0, 1)
        axes[i, 0].imshow(pred)
        axes[i, 0].set_title('Predicted')
        axes[i, 0].axis('off')
        
        # 目标图像
        target = target_images[i].permute(1, 2, 0).cpu().detach().numpy()
        target = np.clip(target, 0, 1)
        axes[i, 1].imshow(target)
        axes[i, 1].set_title('Target')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
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
    可视化图像中的加权区域，突出显示道路和车辆标记

    Args:
        image (torch.Tensor): 输入图像 [B, C, H, W] 或 [C, H, W]
        save_path (str): 保存路径，如果为None则显示
    """
    # 确保输入是合适的维度
    if image.dim() == 4:
        image = image[0]  # 取第一个样本
        
    # 转换到numpy
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().detach().numpy()
        image_np = np.clip(image_np, 0, 1)
    else:
        image_np = image.copy()
    
    # 创建掩码
    # 检测深灰色边界区域
    pixel_mean = np.mean(image_np, axis=2)  # 计算每个像素三个通道的均值
    pixel_std = np.std(image_np, axis=2)    # 计算每个像素三个通道的标准差
    
    # 深灰色边界区域特征: 亮度较低且颜色均匀
    is_dark_boundary = np.logical_and(
        pixel_mean < 0.5,  # 较暗
        pixel_std < 0.03  # 颜色均匀
    )
    
    # 检测红色和蓝色区域(车辆标记)
    # 红色: R通道高，G和B通道低
    is_red = np.logical_and(
        image_np[:, :, 0] > 0.6,
        np.max(image_np[:, :, 1:], axis=2) < 0.4
    )
    
    # 蓝色: B通道高，R和G通道低
    is_blue = np.logical_and(
        image_np[:, :, 2] > 0.6,
        np.max(image_np[:, :, :2], axis=2) < 0.4
    )
    
    # 合并掩码
    vehicle_mask = np.logical_or(is_red, is_blue)
    road_mask = ~is_dark_boundary
    
    # 创建权重热图
    weight_map = np.zeros_like(is_dark_boundary, dtype=float)
    weight_map[is_dark_boundary] = 0.01  # 深灰色边界区域
    weight_map[np.logical_and(road_mask, ~vehicle_mask)] = 3.0  # 道路区域（不包括车辆）
    weight_map[vehicle_mask] = 15.0  # 车辆标记区域
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(is_dark_boundary, cmap='gray')
    axes[0, 1].set_title('Dark Boundary Mask')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(vehicle_mask, cmap='gray')
    axes[1, 0].set_title('Vehicle Markings Mask')
    axes[1, 0].axis('off')
    
    # 使用更明显的颜色图显示权重
    axes[1, 1].imshow(weight_map, cmap='viridis')
    axes[1, 1].set_title('Weight Map (Yellow=Higher)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
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