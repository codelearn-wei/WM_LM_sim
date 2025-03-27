import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from LM_wm.models.bev_encoder import WeightedMSELoss
from LM_wm.configs.config import Config
from LM_wm.test.test_region_detection import create_test_image

def test_weighted_loss():
    """测试加权损失函数是否正确应用权重于不同区域"""
    print("====== 测试加权损失函数 ======")
    
    # 创建测试图像和掩码
    target_image, road_mask, red_mask, blue_mask = create_test_image()
    vehicle_mask = np.logical_or(red_mask, blue_mask).astype(np.float32)
    
    # 创建预测图像 (与目标略有差异)
    with torch.no_grad():
        # 对每个区域添加不同程度的噪声，以便后续分析损失
        noise_level = 0.15
        pred_image = target_image.clone()
        
        # 对边界区域添加较小噪声
        boundary_indices = torch.from_numpy(np.logical_not(road_mask))
        pred_image[:, boundary_indices] += torch.randn_like(pred_image[:, boundary_indices]) * noise_level * 0.5
        
        # 对道路区域添加中等噪声
        road_only = np.logical_and(road_mask, np.logical_not(np.logical_or(red_mask, blue_mask)))
        road_indices = torch.from_numpy(road_only)
        pred_image[:, road_indices] += torch.randn_like(pred_image[:, road_indices]) * noise_level
        
        # 对车辆区域添加较大噪声
        vehicle_indices = torch.from_numpy(np.logical_or(red_mask, blue_mask))
        pred_image[:, vehicle_indices] += torch.randn_like(pred_image[:, vehicle_indices]) * noise_level * 2
        
        # 裁剪确保值在[0,1]范围
        pred_image = torch.clamp(pred_image, 0.0, 1.0)
        
        # 转换为批量形式
        target_batch = target_image.unsqueeze(0)
        pred_batch = pred_image.unsqueeze(0)
    
    # 创建损失函数实例
    config = Config()
    weighted_loss_fn = WeightedMSELoss(
        boundary_weight=config.boundary_weight,
        road_weight=config.road_weight,
        vehicle_weight=config.vehicle_weight
    )
    standard_loss_fn = torch.nn.MSELoss()
    
    # 计算两种损失
    weighted_loss = weighted_loss_fn(pred_batch, target_batch)
    standard_loss = standard_loss_fn(pred_batch, target_batch)
    
    print(f"加权损失值: {weighted_loss.item():.6f}")
    print(f"标准MSE损失值: {standard_loss.item():.6f}")
    print(f"加权/标准损失比例: {weighted_loss.item() / standard_loss.item():.2f}")
    
    # 计算权重地图用于可视化
    with torch.no_grad():
        # 检测边界区域
        pixel_mean = torch.mean(target_batch, dim=1, keepdim=True)
        pixel_std = torch.std(target_batch, dim=1, keepdim=True)
        boundary_mask = torch.logical_and(
            pixel_mean < 0.5,
            pixel_std < 0.03
        ).float().squeeze()
        
        # 检测车辆区域
        is_red = torch.logical_and(
            target_batch[:, 0] > 0.6,
            torch.max(target_batch[:, 1:], dim=1)[0] < 0.4
        ).float().squeeze()
        
        is_blue = torch.logical_and(
            target_batch[:, 2] > 0.6,
            torch.max(target_batch[:, :2], dim=1)[0] < 0.4
        ).float().squeeze()
        
        vehicle_mask_tensor = torch.logical_or(is_red, is_blue).float()
        road_mask_tensor = 1 - boundary_mask
        
        # 构建权重地图
        weight_map = torch.ones_like(boundary_mask) * config.road_weight
        weight_map[boundary_mask > 0.5] = config.boundary_weight
        weight_map[vehicle_mask_tensor > 0.5] = config.vehicle_weight
    
    # 分析各区域损失贡献
    analyze_regional_loss_contribution(
        pred_batch, target_batch, 
        boundary_mask, road_mask_tensor, vehicle_mask_tensor,
        config
    )
    
    # 可视化结果
    visualize_loss_results(
        target_image.permute(1, 2, 0).numpy(),
        pred_image.permute(1, 2, 0).numpy(),
        weight_map.numpy(),
        boundary_mask.numpy(),
        road_mask_tensor.numpy(),
        vehicle_mask_tensor.numpy(),
        config
    )

def analyze_regional_loss_contribution(pred, target, boundary_mask, road_mask, vehicle_mask, config):
    """分析各区域在损失中的贡献"""
    print("\n===== 区域损失贡献分析 =====")
    
    # 计算每个像素的MSE
    pixel_mse = (pred - target).pow(2).mean(dim=1).squeeze()
    
    # 边界区域损失
    boundary_indices = boundary_mask > 0.5
    boundary_loss = pixel_mse[boundary_indices].mean().item() if boundary_indices.any() else 0
    weighted_boundary_loss = boundary_loss * config.boundary_weight
    
    # 道路区域损失 (不包括车辆)
    road_only_indices = torch.logical_and(road_mask > 0.5, vehicle_mask < 0.5)
    road_loss = pixel_mse[road_only_indices].mean().item() if road_only_indices.any() else 0
    weighted_road_loss = road_loss * config.road_weight
    
    # 车辆区域损失
    vehicle_indices = vehicle_mask > 0.5
    vehicle_loss = pixel_mse[vehicle_indices].mean().item() if vehicle_indices.any() else 0
    weighted_vehicle_loss = vehicle_loss * config.vehicle_weight
    
    # 各区域像素数量
    total_pixels = pixel_mse.numel()
    boundary_pixels = boundary_indices.sum().item()
    road_only_pixels = road_only_indices.sum().item()
    vehicle_pixels = vehicle_indices.sum().item()
    
    print(f"边界区域: {boundary_pixels} 像素 ({boundary_pixels/total_pixels*100:.1f}%)")
    print(f"道路区域: {road_only_pixels} 像素 ({road_only_pixels/total_pixels*100:.1f}%)")
    print(f"车辆区域: {vehicle_pixels} 像素 ({vehicle_pixels/total_pixels*100:.1f}%)")
    
    print("\n各区域原始损失:")
    print(f"边界区域损失: {boundary_loss:.6f}")
    print(f"道路区域损失: {road_loss:.6f}")
    print(f"车辆区域损失: {vehicle_loss:.6f}")
    
    print("\n各区域加权后损失:")
    print(f"边界区域加权损失: {weighted_boundary_loss:.6f} (权重={config.boundary_weight})")
    print(f"道路区域加权损失: {weighted_road_loss:.6f} (权重={config.road_weight})")
    print(f"车辆区域加权损失: {weighted_vehicle_loss:.6f} (权重={config.vehicle_weight})")
    
    # 计算加权总损失和各区域百分比
    total_weighted_loss = weighted_boundary_loss + weighted_road_loss + weighted_vehicle_loss
    print("\n各区域在加权总损失中的贡献比例:")
    print(f"边界区域: {weighted_boundary_loss/total_weighted_loss*100:.2f}%")
    print(f"道路区域: {weighted_road_loss/total_weighted_loss*100:.2f}%")
    print(f"车辆区域: {weighted_vehicle_loss/total_weighted_loss*100:.2f}%")

def visualize_loss_results(target_img, pred_img, weight_map, boundary_mask, road_mask, vehicle_mask, config):
    """可视化损失相关结果"""
    # 计算误差图
    error_map = np.mean((pred_img - target_img) ** 2, axis=2)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行
    axes[0, 0].imshow(target_img)
    axes[0, 0].set_title('目标图像')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_img)
    axes[0, 1].set_title('预测图像')
    axes[0, 1].axis('off')
    
    error_img = axes[0, 2].imshow(error_map, cmap='hot')
    axes[0, 2].set_title('均方误差')
    axes[0, 2].axis('off')
    fig.colorbar(error_img, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 第二行
    # 创建区域掩码可视化
    region_viz = np.zeros(target_img.shape, dtype=np.float32)
    region_viz[boundary_mask > 0.5, :] = [1.0, 0.0, 0.0]  # 边界区域 - 红色
    road_only_mask = np.logical_and(road_mask > 0.5, np.logical_not(vehicle_mask > 0.5))
    region_viz[road_only_mask, :] = [0.0, 1.0, 0.0]  # 道路区域 - 绿色
    region_viz[vehicle_mask > 0.5, :] = [0.0, 0.0, 1.0]  # 车辆区域 - 蓝色
    
    axes[1, 0].imshow(region_viz)
    axes[1, 0].set_title('检测区域')
    axes[1, 0].axis('off')
    
    # 创建权重地图的热图
    weight_img = axes[1, 1].imshow(weight_map, cmap='viridis')
    axes[1, 1].set_title('权重地图')
    axes[1, 1].axis('off')
    cbar = fig.colorbar(weight_img, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.set_ticks([config.boundary_weight, config.road_weight, config.vehicle_weight])
    cbar.set_ticklabels([f'边界: {config.boundary_weight}', 
                        f'道路: {config.road_weight}', 
                        f'车辆: {config.vehicle_weight}'])
    
    # 创建加权误差图
    weighted_error = error_map * weight_map
    weighted_error_img = axes[1, 2].imshow(weighted_error, cmap='hot')
    axes[1, 2].set_title('加权误差')
    axes[1, 2].axis('off')
    fig.colorbar(weighted_error_img, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # 保存结果
    os.makedirs('LM_wm/test/results', exist_ok=True)
    plt.savefig('LM_wm/test/results/weighted_loss_test.png')
    plt.close()
    
    print("加权损失测试可视化已保存到 LM_wm/test/results/weighted_loss_test.png")
    
    # 额外创建饼图显示损失贡献比例
    create_loss_contribution_chart(target_img, pred_img, boundary_mask, road_mask, vehicle_mask, config)

def create_loss_contribution_chart(target_img, pred_img, boundary_mask, road_mask, vehicle_mask, config):
    """创建损失贡献比例饼图"""
    # 计算每个像素的MSE
    mse = np.mean((pred_img - target_img) ** 2, axis=2)
    
    # 提取各区域
    boundary_indices = boundary_mask > 0.5
    road_only_indices = np.logical_and(road_mask > 0.5, np.logical_not(vehicle_mask > 0.5))
    vehicle_indices = vehicle_mask > 0.5
    
    # 计算各区域损失
    boundary_loss = np.mean(mse[boundary_indices]) * config.boundary_weight if boundary_indices.any() else 0
    road_loss = np.mean(mse[road_only_indices]) * config.road_weight if road_only_indices.any() else 0
    vehicle_loss = np.mean(mse[vehicle_indices]) * config.vehicle_weight if vehicle_indices.any() else 0
    
    # 创建饼图
    plt.figure(figsize=(10, 8))
    
    # 损失贡献比例
    loss_values = [boundary_loss, road_loss, vehicle_loss]
    labels = [f'边界区域: {boundary_loss:.6f}\n({boundary_loss/sum(loss_values)*100:.1f}%)', 
              f'道路区域: {road_loss:.6f}\n({road_loss/sum(loss_values)*100:.1f}%)', 
              f'车辆区域: {vehicle_loss:.6f}\n({vehicle_loss/sum(loss_values)*100:.1f}%)']
    
    # 颜色
    colors = ['#FF6666', '#66FF66', '#6666FF']
    
    # 突出车辆区域
    explode = (0, 0, 0.1)
    
    plt.pie(loss_values, explode=explode, labels=labels, colors=colors, 
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # 保持圆形
    plt.title('各区域在加权损失中的贡献比例')
    
    # 显示权重
    plt.figtext(0.5, 0.01, 
                f'配置权重: 边界={config.boundary_weight}, 道路={config.road_weight}, 车辆={config.vehicle_weight}', 
                ha='center', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('LM_wm/test/results/loss_contribution_chart.png')
    plt.close()
    
    print("损失贡献比例图已保存到 LM_wm/test/results/loss_contribution_chart.png")

if __name__ == "__main__":
    test_weighted_loss() 