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

def create_test_image():
    """创建一个测试图像，包含深灰色边界、浅灰色道路和红蓝色车辆标记"""
    # 创建深灰色背景
    test_image = np.ones((224, 224, 3), dtype=np.float32) * 0.47
    
    # 添加浅灰色道路区域
    road_mask = np.zeros((224, 224), dtype=bool)
    road_mask[70:154, :] = True
    test_image[road_mask] = 0.65
    
    # 添加红色车辆标记
    red_mask = np.zeros((224, 224), dtype=bool)
    for x in range(50, 200, 30):
        red_mask[100:110, x:x+10] = True
    test_image[red_mask, 0] = 0.85  # 红色通道
    test_image[red_mask, 1:] = 0.2   # 绿色和蓝色通道
    
    # 添加蓝色车辆标记
    blue_mask = np.zeros((224, 224), dtype=bool)
    for x in range(65, 185, 30):
        blue_mask[115:125, x:x+10] = True
    test_image[blue_mask, 2] = 0.85  # 蓝色通道
    test_image[blue_mask, :2] = 0.2  # 红色和绿色通道
    
    # 转换为PyTorch张量
    tensor_image = torch.from_numpy(test_image).permute(2, 0, 1)
    
    # 返回图像及各区域掩码
    return tensor_image, road_mask, red_mask, blue_mask

def test_region_detection():
    """测试区域检测是否准确"""
    print("====== 测试区域检测准确性 ======")
    
    # 创建测试图像和已知掩码
    test_image, road_mask, red_mask, blue_mask = create_test_image()
    vehicle_mask = np.logical_or(red_mask, blue_mask)
    boundary_mask = ~road_mask
    
    # 准备为WeightedMSELoss使用
    target_batch = test_image.unsqueeze(0)
    
    # 使用损失函数的区域检测逻辑
    with torch.no_grad():
        # 检测深灰色边界区域
        pixel_mean = torch.mean(target_batch, dim=1, keepdim=True)
        pixel_std = torch.std(target_batch, dim=1, keepdim=True)
        
        detected_boundary = torch.logical_and(
            pixel_mean < 0.5,  # 较暗
            pixel_std < 0.03  # 颜色均匀
        ).float().squeeze()
        
        # 检测红色区域
        is_red = torch.logical_and(
            target_batch[:, 0] > 0.6,
            torch.max(target_batch[:, 1:], dim=1)[0] < 0.4
        ).float().squeeze()
        
        # 检测蓝色区域
        is_blue = torch.logical_and(
            target_batch[:, 2] > 0.6,
            torch.max(target_batch[:, :2], dim=1)[0] < 0.4
        ).float().squeeze()
        
        # 合并车辆区域
        detected_vehicle = torch.logical_or(is_red, is_blue).float()
        
        # 道路区域（非边界区域）
        detected_road = 1 - detected_boundary
    
    # 计算准确率和重叠度
    boundary_accuracy = calculate_iou(detected_boundary.numpy() > 0.5, boundary_mask)
    vehicle_accuracy = calculate_iou(detected_vehicle.numpy() > 0.5, vehicle_mask)
    road_accuracy = calculate_iou(detected_road.numpy() > 0.5, road_mask)
    
    print(f"边界区域检测 IoU: {boundary_accuracy:.4f}")
    print(f"车辆标记区域检测 IoU: {vehicle_accuracy:.4f}")
    print(f"道路区域检测 IoU: {road_accuracy:.4f}")
    
    # 可视化结果
    visualize_detection_results(
        test_image.permute(1, 2, 0).numpy(),
        detected_boundary.numpy(),
        detected_vehicle.numpy(),
        detected_road.numpy(),
        boundary_mask,
        vehicle_mask,
        road_mask
    )
    
    return boundary_accuracy, vehicle_accuracy, road_accuracy

def calculate_iou(detected_mask, true_mask):
    """计算交并比 (IoU)"""
    intersection = np.logical_and(detected_mask, true_mask).sum()
    union = np.logical_or(detected_mask, true_mask).sum()
    iou = intersection / union if union > 0 else 0
    return iou

def visualize_detection_results(original_img, detected_boundary, detected_vehicle, 
                               detected_road, true_boundary, true_vehicle, true_road):
    """可视化检测结果和真实掩码的比较"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # 第一行: 原始图像和生成的掩码
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('原始测试图像')
    axes[0, 0].axis('off')
    
    # 创建掩码可视化图像
    mask_viz = np.zeros(original_img.shape, dtype=np.float32)
    mask_viz[true_boundary, :] = [1.0, 0.0, 0.0]  # 边界区域 - 红色
    road_without_vehicle = np.logical_and(true_road, np.logical_not(true_vehicle))
    mask_viz[road_without_vehicle, :] = [0.0, 1.0, 0.0]  # 道路区域 - 绿色
    mask_viz[true_vehicle, :] = [0.0, 0.0, 1.0]  # 车辆区域 - 蓝色
    
    axes[0, 1].imshow(mask_viz)
    axes[0, 1].set_title('真实区域掩码')
    axes[0, 1].axis('off')
    
    # 创建检测掩码可视化
    detected_viz = np.zeros(original_img.shape, dtype=np.float32)
    detected_viz[detected_boundary > 0.5, :] = [1.0, 0.0, 0.0]  # 边界区域 - 红色
    detected_road_only = np.logical_and(detected_road > 0.5, np.logical_not(detected_vehicle > 0.5))
    detected_viz[detected_road_only, :] = [0.0, 1.0, 0.0]  # 道路区域 - 绿色
    detected_viz[detected_vehicle > 0.5, :] = [0.0, 0.0, 1.0]  # 车辆区域 - 蓝色
    
    axes[0, 2].imshow(detected_viz)
    axes[0, 2].set_title('检测区域掩码')
    axes[0, 2].axis('off')
    
    # 第二行：各个区域的真实掩码
    axes[1, 0].imshow(true_boundary, cmap='gray')
    axes[1, 0].set_title('真实边界区域')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.logical_and(true_road, np.logical_not(true_vehicle)), cmap='gray')
    axes[1, 1].set_title('真实道路区域(不含车辆)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(true_vehicle, cmap='gray')
    axes[1, 2].set_title('真实车辆区域')
    axes[1, 2].axis('off')
    
    # 第三行：各个区域的检测掩码
    axes[2, 0].imshow(detected_boundary, cmap='hot')
    axes[2, 0].set_title('检测到的边界区域')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(detected_road, cmap='hot')
    axes[2, 1].set_title('检测到的道路区域')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(detected_vehicle, cmap='hot')
    axes[2, 2].set_title('检测到的车辆区域')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    os.makedirs('LM_wm/test/results', exist_ok=True)
    plt.savefig('LM_wm/test/results/region_detection_test.png')
    plt.close()
    
    print("区域检测测试可视化已保存到 LM_wm/test/results/region_detection_test.png")

if __name__ == "__main__":
    test_region_detection() 