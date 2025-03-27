# BEV轨迹预测模型 - 改进版

本项目包含一个鸟瞰视角(BEV)预测模型，已经进行了增强以集中关注道路区域和车辆标记，同时忽略上下边界的深灰色区域。

## 主要改进

### 1. 区域专注策略

我们实现了多种技术，使模型忽略上下边界的深灰色区域，而关注道路区域（包括道路上的浅灰色部分）：

- **自定义加权损失函数**: 添加了`WeightedMSELoss`，为不同区域分配不同权重：
  - 车辆标记（红/蓝色块）获得最高权重 (15.0)
  - 道路区域获得中等权重 (3.0)
  - 深灰色边界区域获得接近零的权重 (0.01)

- **掩码训练**: 添加了道路掩码生成，可以准确识别：
  - 上下深灰色边界区域（需要忽略）
  - 道路区域（需要关注，包括道路的浅灰色部分）
  - 车辆标记区域（最重要的部分）

- **随机噪声掩码**: 在训练过程中对边界区域应用随机灰色噪声，帮助模型学习这些区域是无关紧要的。

### 2. 车辆标记增强

- **颜色识别**: 实现了检测图像中红色和蓝色车辆标记的算法
- **加权损失**: 为包含车辆标记的像素分配最高权重 (通过`vehicle_weight`参数配置，默认值15.0)
- **可视化调试**: 添加了可视化工具，确认车辆标记的正确检测

### 3. 注意力机制可视化

- 由于模型使用DINOv2作为骨干网络，其中包含自注意力机制：
  - 添加了训练期间的注意力图可视化
  - 添加了`get_attention_maps()`方法来检索注意力权重
  - 创建了将注意力热图覆盖在输入图像上的可视化

- 这使你可以验证模型的注意力是否正确地集中在道路区域，特别是车辆标记上。

## 使用方法

### 使用集成主程序 (推荐)

本项目提供了一个集成主程序，可以通过一个命令执行数据生成、模型训练和预测等功能：

```bash
# 生成训练数据
python -m LM_wm.main --generate_data

# 训练模型
python -m LM_wm.main --train --mode image

# 使用训练好的模型进行预测
python -m LM_wm.main --predict --sample_idx 5

# 执行完整流程（生成数据并训练）
python -m LM_wm.main --all --mode image
```

预测选项:
- `--checkpoint`: 模型检查点路径，默认使用最佳模型
- `--data_dir`: 测试数据目录，默认使用配置中的数据目录
- `--output_dir`: 预测结果输出目录，默认为 `LM_wm/predictions`
- `--batch_size`: 预测批次大小，默认为 4
- `--sample_idx`: 指定预测单个样本的索引，如果不指定则进行批量预测

示例:
```bash
# 使用特定检查点对特定样本进行预测
python -m LM_wm.main --predict --checkpoint LM_wm/checkpoints/best_model.pth --sample_idx 5

# 指定测试数据目录和输出目录进行批量预测
python -m LM_wm.main --predict --data_dir LM_wm/validation_data --output_dir results/predictions
```

### 各功能的单独使用

您也可以分别使用训练脚本和预测脚本进行操作:

#### 训练模型

```bash
python -m LM_wm.scripts.train --mode train --train_mode image
```

参数说明：
- `--mode`: 指定运行模式，可选 `train` 或 `predict`
- `--train_mode`: 指定训练模式，可选 `feature` 或 `image`

#### 使用模型进行预测

有多种方式使用训练好的模型进行预测：

##### 方式1：使用训练脚本的预测模式

```bash
python -m LM_wm.scripts.train --mode predict [--选项]
```

示例：
```bash
# 使用默认设置预测所有样本
python -m LM_wm.scripts.train --mode predict

# 预测特定样本
python -m LM_wm.scripts.train --mode predict --sample_idx 5
```

##### 方式2：使用专用的预测脚本

```bash
python -m LM_wm.scripts.predict [--选项]
```

示例：
```bash
# 使用默认设置预测所有样本
python -m LM_wm.scripts.predict

# 预测特定样本
python -m LM_wm.scripts.predict --sample_idx 5
```

##### 方式3：在Python脚本中调用预测函数

您还可以在其他Python脚本中导入并调用预测函数：

```python
from LM_wm.scripts.predict import predict_main_func

# 调用预测函数
predict_main_func(
    checkpoint_path="LM_wm/checkpoints/best_model.pth",
    data_dir="LM_wm/validation_data",
    output_dir="results/predictions",
    batch_size=4,
    sample_idx=None  # 设置为None进行批量预测，或指定样本索引进行单样本预测
)
```

### 预测选项

无论使用哪种方式进行预测，都支持以下选项：

- `--checkpoint`: 模型检查点路径，默认使用最佳模型 (`LM_wm/checkpoints/best_model.pth`)
- `--data_dir`: 测试数据目录，默认使用配置中的数据目录
- `--output_dir`: 预测结果输出目录，默认为 `LM_wm/predictions`
- `--batch_size`: 预测批次大小，默认为 4
- `--sample_idx`: 指定预测单个样本的索引，如果不指定则进行批量预测

### 预测输出

预测脚本会生成以下输出：
1. **预测与目标对比图**: 可视化预测结果与实际目标的对比
2. **权重区域图**: 显示不同图像区域的权重分配
3. **保存预测图像**: 将预测图像保存为PNG文件
4. **保存目标图像**: 将目标图像保存为PNG文件

所有输出都会保存在指定的输出目录中，每个样本会有自己的子目录。

## 配置

你可以通过`Config`类中的以下参数控制行为：

```python
# 区域关注配置
self.focus_on_road = True     # 强制模型关注道路区域
self.road_weight = 3.0        # 道路区域权重
self.vehicle_weight = 15.0    # 车辆标记区域权重
self.boundary_weight = 0.01   # 深灰色边界区域权重
```

## 可视化

改进后的模型在训练过程中创建多种可视化：

1. **权重区域图**: 显示不同图像区域的权重分配
2. **道路掩码**: 可视化检测到的道路与深灰色边界区域
3. **注意力图**: 显示模型关注的位置
4. **预测结果**: 预测图像与目标图像的对比

这些可视化保存在`logs/visualizations`目录中。

## 测试框架

项目包含一个全面的测试框架，用于验证区域检测和加权损失功能是否按预期工作。测试脚本位于`LM_wm/test/`目录下。

### 测试组件

1. **区域检测测试 (`test_region_detection.py`)**
   - 测试边界区域、道路区域和车辆标记的检测准确性
   - 生成综合可视化结果，显示各区域掩码与检测到的掩码的比较
   - 计算每种区域类型的IoU(交并比)得分

2. **加权损失函数测试 (`test_weighted_loss.py`)**
   - 验证损失函数是否正确应用配置的权重
   - 分析各区域(边界、道路、车辆)对总损失的贡献
   - 生成权重地图、误差图和加权误差图的可视化
   - 创建饼图展示各区域在加权损失中的贡献比例

3. **掩码与损失集成测试 (`test_mask_and_loss.py`)**
   - 测试掩码生成和损失函数的集成
   - 验证模型注意力机制是否与区域重要性一致

### 运行测试

使用命令行脚本`LM_wm/test/run_tests.py`可以轻松运行测试：

```bash
# 运行所有测试并生成综合报告
python LM_wm/test/run_tests.py --all

# 仅运行区域检测测试
python LM_wm/test/run_tests.py --region

# 仅运行加权损失函数测试
python LM_wm/test/run_tests.py --loss

# 仅运行掩码和损失集成测试
python LM_wm/test/run_tests.py --integration
```

### 测试报告

完整的测试会生成HTML格式的综合报告，包含：
- 测试摘要（通过/失败项目数、总耗时）
- 各测试项目的详细结果
- 所有测试可视化的图片库

报告保存在`LM_wm/test/results/test_report.html`。各个测试的图形输出保存在`LM_wm/test/results/`目录下。

## 注意事项

- 保持了224x224的图像输入大小
- 模型继续使用带有内置注意力机制的DINOv2骨干网络
- 训练过程现在包括基于区域重要性的梯度缩放
- 模型经过优化，能够准确区分上下深灰色边界区域和道路的浅灰色部分