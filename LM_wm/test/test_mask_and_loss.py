import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from LM_wm.datasets.training_dataset import LMTrainingDataset
from LM_wm.models.bev_encoder import WeightedMSELoss, BEVPredictionModel
from LM_wm.utils.visualization import visualize_weighted_regions
from LM_wm.configs.config import Config

def test_road_mask_generation():
    """测试道路掩码生成是否正确区分深灰色边界和浅灰色道路"""
    print("====== 测试道路掩码生成 ======")
    
    # 从训练数据中加载一个示例图像
    config = Config()
    dataset = LMTrainingDataset(config.data_dir, config.history_steps, focus_on_road=True)
    
    # 如果数据集为空，创建一个测试图像
    if len(dataset) == 0:
        print("数据集为空，创建测试图像...")
        # 创建一个测试图像: 深灰色边界上下，中间是浅灰色道路，添加几个红蓝色块
        test_image = np.ones((224, 224, 3), dtype=np.float32) * 0.47  # 深灰色边界
        
        # 添加浅灰色道路区域
        road_mask = np.zeros((224, 224), dtype=bool)
        road_mask[70:154, :] = True  # 中间道路区域
        test_image[road_mask] = 0.65  # 浅灰色
        
        # 添加红色车辆标记
        for x in range(50, 200, 30):
            test_image[100:110, x:x+10, 0] = 0.8  # 设置红色通道
            test_image[100:110, x:x+10, 1:] = 0.2  # 设置绿色和蓝色通道
        
        # 添加蓝色车辆标记
        for x in range(65, 185, 30):
            test_image[115:125, x:x+10, 2] = 0.8  # 设置蓝色通道
            test_image[115:125, x:x+10, :2] = 0.2  # 设置红色和绿色通道
            
        # 转换为Tensor
        test_tensor = torch.from_numpy(test_image).permute(2, 0, 1)
    else:
        print(f"从数据集加载示例图像 (共 {len(dataset)} 个样本)")
        sample = dataset[0]
        test_tensor = sample['next_frame']
    
    # 生成道路掩码
    road_mask = dataset._create_road_mask(test_tensor)
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 显示原始图像
    plt.subplot(1, 3, 1)
    img_np = test_tensor.permute(1, 2, 0).numpy()
    plt.imshow(img_np)
    plt.title('原始图像')
    plt.axis('off')
    
    # 显示道路掩码
    plt.subplot(1, 3, 2)
    plt.imshow(road_mask.numpy(), cmap='gray')
    plt.title('道路掩码 (白色=道路区域)')
    plt.axis('off')
    
    # 显示应用掩码后的图像
    plt.subplot(1, 3, 3)
    masked_tensor = dataset._apply_focus_mask(test_tensor)
    masked_np = masked_tensor.permute(1, 2, 0).numpy()
    plt.imshow(masked_np)
    plt.title('应用掩码后的图像')
    plt.axis('off')
    
    # 保存结果
    os.makedirs('LM_wm/test/results', exist_ok=True)
    plt.savefig('LM_wm/test/results/road_mask_test.png')
    plt.close()
    
    print("道路掩码测试完成. 结果已保存到 LM_wm/test/results/road_mask_test.png")
    return test_tensor, road_mask

def test_weighted_loss_function(test_image, road_mask):
    """测试权重损失函数是否正确分配权重"""
    print("\n====== 测试权重损失函数 ======")
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建损失函数实例
    loss_fn = WeightedMSELoss(
        road_weight=config.road_weight,
        vehicle_weight=config.vehicle_weight,
        boundary_weight=config.boundary_weight
    )
    
    # 创建一个与测试图像相同的预测图像，但添加一些噪声
    pred_image = test_image.clone() + torch.randn_like(test_image) * 0.1
    
    # 扩展维度以匹配损失函数的输入要求 [B, C, H, W]
    target_batch = test_image.unsqueeze(0)
    pred_batch = pred_image.unsqueeze(0)
    road_mask_batch = road_mask.unsqueeze(0)
    
    # 计算带权重的损失
    weighted_loss = loss_fn(pred_batch, target_batch, road_mask_batch)
    
    # 计算普通MSE损失作为对比
    mse_loss = torch.nn.MSELoss()(pred_batch, target_batch)
    
    print(f"加权损失值: {weighted_loss.item():.6f}")
    print(f"普通MSE损失值: {mse_loss.item():.6f}")
    
    # 可视化权重分布
    visualize_result = visualize_weighted_regions(target_batch, save_path='LM_wm/test/results/weight_distribution.png')
    
    print("权重损失函数测试完成. 结果已保存到 LM_wm/test/results/weight_distribution.png")
    
    # 计算不同区域的损失贡献，以验证权重是否生效
    with torch.no_grad():
        mse = (pred_batch - target_batch) ** 2
        
        # 获取车辆区域掩码
        is_red = torch.logical_and(
            target_batch[:, 0] > 0.6,
            torch.max(target_batch[:, 1:], dim=1)[0] < 0.4
        ).float().unsqueeze(1)
        
        is_blue = torch.logical_and(
            target_batch[:, 2] > 0.6,
            torch.max(target_batch[:, :2], dim=1)[0] < 0.4
        ).float().unsqueeze(1)
        
        is_vehicle = torch.logical_or(is_red, is_blue).float()
        is_road = road_mask_batch.clone()
        is_boundary = 1 - is_road
        
        # 计算每个区域的平均MSE
        vehicle_mse = (mse * is_vehicle).sum() / (is_vehicle.sum() + 1e-8)
        road_mse = (mse * (is_road * (1 - is_vehicle))).sum() / ((is_road * (1 - is_vehicle)).sum() + 1e-8)
        boundary_mse = (mse * is_boundary).sum() / (is_boundary.sum() + 1e-8)
        
        print("\n各区域MSE贡献分析:")
        print(f"车辆标记区域 MSE: {vehicle_mse.item():.6f}")
        print(f"道路区域 MSE: {road_mse.item():.6f}")
        print(f"边界区域 MSE: {boundary_mse.item():.6f}")
        
        # 应用权重后的MSE
        weighted_vehicle_mse = vehicle_mse * config.vehicle_weight
        weighted_road_mse = road_mse * config.road_weight
        weighted_boundary_mse = boundary_mse * config.boundary_weight
        
        print("\n应用权重后各区域MSE贡献:")
        print(f"车辆标记区域 加权MSE: {weighted_vehicle_mse.item():.6f} (权重={config.vehicle_weight})")
        print(f"道路区域 加权MSE: {weighted_road_mse.item():.6f} (权重={config.road_weight})")
        print(f"边界区域 加权MSE: {weighted_boundary_mse.item():.6f} (权重={config.boundary_weight})")
        
        # 计算各区域在总损失中的比例
        total_weighted = weighted_vehicle_mse + weighted_road_mse + weighted_boundary_mse
        print("\n各区域在总损失中的占比:")
        print(f"车辆标记区域: {(weighted_vehicle_mse / total_weighted * 100).item():.2f}%")
        print(f"道路区域: {(weighted_road_mse / total_weighted * 100).item():.2f}%")
        print(f"边界区域: {(weighted_boundary_mse / total_weighted * 100).item():.2f}%")
    
    return weighted_loss, mse_loss



def main():
    """运行所有测试函数"""
    os.makedirs('LM_wm/test/results', exist_ok=True)
    print("开始测试掩码生成和损失函数...\n")
    
    # 测试道路掩码生成
    test_image, road_mask = test_road_mask_generation()
    
    # 测试权重损失函数
    test_weighted_loss_function(test_image, road_mask)
    
    
    print("\n所有测试完成！请查看 LM_wm/test/results 目录下的结果图像。")

if __name__ == "__main__":
    main() 