import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model, AutoModel
import torchvision
from torch.optim import AdamW

class BEVEncoder(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化DINOv2模型和处理器
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", do_rescale=False,use_fast=True)
        self.encoder = Dinov2Model.from_pretrained("facebook/dinov2-base").to(self.device)
        
        # 冻结预训练模型参数
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # 设置输出维度
        self.output_dim = self.encoder.config.hidden_size
            
    def forward(self, images):
        """
        编码BEV图像

        Args:
            images (torch.Tensor): 形状为 (batch_size, seq_len, channels, height, width) 的图像序列
                                或 (batch_size, channels, height, width) 的单个图像

        Returns:
            torch.Tensor: 编码后的特征
        """
        # 检查输入维度
        if len(images.shape) == 5:  # 序列数据
            batch_size, seq_len, channels, height, width = images.shape
            # 重塑为 (batch_size * seq_len, channels, height, width)
            images = images.reshape(-1, channels, height, width)
        elif len(images.shape) == 4:  # 单个图像
            batch_size, channels, height, width = images.shape
            seq_len = 1
        else:
            raise ValueError(f"Invalid image shape. Expected 4 or 5 dimensions, but got {len(images.shape)}")
        
        # 准备输入
        inputs = self.processor(images, return_tensors="pt", do_rescale=False)
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # 获取特征
        outputs = self.encoder(pixel_values)
        
        # 使用CLS token作为图像特征
        features = outputs.last_hidden_state[:, 0]  # (batch_size * seq_len, hidden_size)
        
        # 如果是序列数据，重塑回 (batch_size, seq_len, hidden_size)
        if seq_len > 1:
            features = features.reshape(batch_size, seq_len, -1)
            
        return features

class ActionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(self.device)
        
    def forward(self, actions):
        """
        编码动作序列

        Args:
            actions (torch.Tensor): 形状为 (batch_size, seq_len, action_dim) 的动作张量

        Returns:
            torch.Tensor: 编码后的特征
        """
        batch_size, seq_len, _ = actions.shape
        actions_flat = actions.reshape(batch_size * seq_len, -1)
        features = self.encoder(actions_flat)
        return features.reshape(batch_size, seq_len, self.hidden_dim)

class BEVPredictor(nn.Module):
    """BEV预测器"""
    def __init__(self, bev_dim: int, action_dim: int, hidden_dim: int, device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        # 特征压缩器
        self.feature_compressor = nn.Sequential(
            nn.Linear(bev_dim + action_dim, hidden_dim),
            nn.ReLU()
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, bev_dim),
            nn.LayerNorm(bev_dim)
        )
        
    def forward(self, bev_features: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        """
        预测下一帧的BEV特征

        Args:
            bev_features (torch.Tensor): BEV特征 [batch_size, history_steps, bev_dim]
            action_features (torch.Tensor): 动作特征 [batch_size, history_steps, action_dim]

        Returns:
            torch.Tensor: 预测的下一帧BEV特征 [batch_size, bev_dim]
        """
        # 确保输入在正确的设备上
        bev_features = bev_features.to(self.device)
        action_features = action_features.to(self.device)
        
        # 组合特征
        combined_features = torch.cat([bev_features, action_features], dim=-1)
        
        # 压缩特征维度
        compressed_features = self.feature_compressor(combined_features)
        
        # LSTM处理
        lstm_out, _ = self.lstm(compressed_features)
        
        # 只使用最后一个时间步的输出
        last_features = lstm_out[:, -1, :]
        
        # 生成预测
        prediction = self.output_layer(last_features)
        
        return prediction

class UNetDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32, 16], output_channels=3):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.hidden_dim = hidden_dims[0]  # 使用第一个维度作为主要隐藏维度
        
        # 初始特征转换
        self.initial_conv = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (self.hidden_dim, 7, 7))
        )
        
        # 上采样层
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dims[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[1]),
            nn.ReLU()
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[1], self.hidden_dims[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[2]),
            nn.ReLU()
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[2], self.hidden_dims[3], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[3]),
            nn.ReLU()
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[3], self.hidden_dims[4], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[4]),
            nn.ReLU()
        )
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[4], output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        self.skip1 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[1], self.hidden_dims[2], 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[2], self.hidden_dims[3], 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[3], self.hidden_dims[4], 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[4], self.hidden_dims[4], 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
    def forward(self, x):
        # 初始特征转换
        x = self.initial_conv(x)  # (B, 256, 7, 7)
        
        # 上采样路径
        x1 = self.up1(x)          # (B, 128, 14, 14)
        x2 = self.up2(x1)         # (B, 64, 28, 28)
        x3 = self.up3(x2)         # (B, 32, 56, 56)
        x4 = self.up4(x3)         # (B, 16, 112, 112)
        
        # 跳跃连接 - 修改为正确的连接方式
        x2 = x2 + self.skip1(x1)  # 将128通道转换为64通道
        x3 = x3 + self.skip2(x2)  # 将64通道转换为32通道
        x4 = x4 + self.skip3(x3)  # 将32通道转换为16通道
        
        # 最终输出
        out = self.final_conv(x4)  # (B, 3, 224, 224)
        return out

class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        # 加载预训练的VGG模型
        vgg = torchvision.models.vgg16(pretrained=True).features[:16].to(device)
        self.vgg = nn.Sequential(*list(vgg))
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.eval()
        
    def forward(self, x, y):
        # 提取特征
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        return nn.MSELoss()(x_features, y_features)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        
    def forward(self, pred_image, target_image, pred_features, target_features):
        # 图像重建损失
        reconstruction_loss = self.mse_loss(pred_image, target_image)
        
        # 感知损失
        perceptual_loss = self.perceptual_loss(pred_image, target_image)
        
        # 特征空间损失
        feature_loss = self.mse_loss(pred_features, target_features)
        
        # 组合损失
        total_loss = reconstruction_loss + self.alpha * perceptual_loss + self.beta * feature_loss
        return total_loss

class WeightedMSELoss(nn.Module):
    """自定义的带权重的MSE损失函数，对车辆标记区域给予更高权重，忽略深灰色边界区域
    支持渐进式权重调整，随着训练进行逐渐增加车辆区域权重
    """
    def __init__(self, road_weight=3.0, vehicle_weight=15.0, boundary_weight=0.01, 
                 weight_schedule=None, current_epoch=0):
        super().__init__()
        self.road_weight = road_weight      # 道路区域权重
        self.vehicle_weight = vehicle_weight  # 车辆标记区域权重
        self.boundary_weight = boundary_weight  # 深灰色边界区域权重
        
        # 渐进式权重调整
        self.weight_schedule = weight_schedule  # 权重调度字典，格式: {epoch: {"vehicle": weight, "road": weight}}
        self.current_epoch = current_epoch
        
        # 更新当前使用的权重
        self._update_weights()
        
    def _update_weights(self):
        """根据当前epoch更新权重"""
        if self.weight_schedule is None:
            return
            
        # 获取初始和最终权重
        initial_weights = self.weight_schedule.get(0, {
            "vehicle": self.vehicle_weight,
            "road": self.road_weight,
            "boundary": self.boundary_weight
        })
        
        final_weights = self.weight_schedule.get(max(self.weight_schedule.keys()), {
            "vehicle": self.vehicle_weight,
            "road": self.road_weight,
            "boundary": self.boundary_weight
        })
        
        # 计算当前epoch的进度
        max_epoch = max(self.weight_schedule.keys())
        progress = min(1.0, self.current_epoch / max_epoch)
        
        # 线性插值计算当前权重
        self.vehicle_weight = initial_weights["vehicle"] + (final_weights["vehicle"] - initial_weights["vehicle"]) * progress
        self.road_weight = initial_weights["road"] + (final_weights["road"] - initial_weights["road"]) * progress
        self.boundary_weight = initial_weights["boundary"] + (final_weights["boundary"] - initial_weights["boundary"]) * progress
    
    def update_epoch(self, epoch):
        """更新当前epoch并相应地调整权重"""
        self.current_epoch = epoch
        self._update_weights()
        
        # 打印当前权重值，用于调试
        print(f"Epoch {epoch} - Current weights: vehicle={self.vehicle_weight:.2f}, "
              f"road={self.road_weight:.2f}, boundary={self.boundary_weight:.2f}")
    
    def forward(self, pred, target, road_mask=None):
        """计算加权MSE损失
        
        Args:
            pred (torch.Tensor): 预测图像 [B, C, H, W]
            target (torch.Tensor): 目标图像 [B, C, H, W]
            road_mask (torch.Tensor, optional): 道路掩码 [B, H, W]
            
        Returns:
            torch.Tensor: 加权MSE损失
        """
        # 计算像素级MSE
        mse = (pred - target) ** 2
        
        # 如果提供了道路掩码，直接使用
        if road_mask is not None:
            # 确保掩码维度正确 [B, H, W] -> [B, 1, H, W]
            if road_mask.dim() == 3:
                road_mask = road_mask.unsqueeze(1)
                
            # 使用提供的road_mask (1=道路区域，0=深灰色边界区域)
            # 赋予道路区域权重
            mask = torch.ones_like(road_mask) * self.boundary_weight  # 初始化为边界权重
            mask = torch.where(road_mask > 0.5, torch.tensor(self.road_weight, device=mse.device), mask)  # 道路区域赋予较高权重
            
            # 检测红蓝标记区域(车辆)
            # 红色: R通道高，G和B通道低
            is_red = torch.logical_and(
                target[:, 0] > 0.6,  # 红色通道高
                torch.max(target[:, 1:], dim=1)[0] < 0.4  # 绿色和蓝色通道低
            ).float().unsqueeze(1)
            
            # 蓝色: B通道高，R和G通道低
            is_blue = torch.logical_and(
                target[:, 2] > 0.6,  # 蓝色通道高
                torch.max(target[:, :2], dim=1)[0] < 0.4  # 红色和绿色通道低
            ).float().unsqueeze(1)
            
            # 车辆标记区域给予最高权重
            is_vehicle = torch.logical_or(is_red, is_blue).float()
            vehicle_mask = is_vehicle * self.vehicle_weight
            mask = torch.where(vehicle_mask > 0, vehicle_mask, mask)
        else:
            # 如果没有提供掩码，通过图像颜色创建掩码
            # 识别深灰色边界区域 - 较暗的灰色
            pixel_mean = torch.mean(target, dim=1, keepdim=True)
            pixel_std = torch.std(target, dim=1, keepdim=True)
            
            is_dark_boundary = torch.logical_and(
                pixel_mean < 0.5,  # 较暗
                pixel_std < 0.03  # 颜色均匀
            ).float()
            
            # 创建基础权重掩码: 道路区域高权重，边界区域低权重
            mask = torch.ones_like(is_dark_boundary) * self.road_weight  # 默认为道路权重
            mask = torch.where(is_dark_boundary > 0.5, torch.tensor(self.boundary_weight, device=mse.device), mask)
            
            # 检测红蓝标记区域(车辆)
            # 红色: R通道高，G和B通道低
            is_red = torch.logical_and(
                target[:, 0] > 0.6,  # 红色通道高
                torch.max(target[:, 1:], dim=1)[0] < 0.4  # 绿色和蓝色通道低
            ).float().unsqueeze(1)
            
            # 蓝色: B通道高，R和G通道低
            is_blue = torch.logical_and(
                target[:, 2] > 0.6,  # 蓝色通道高
                torch.max(target[:, :2], dim=1)[0] < 0.4  # 红色和绿色通道低
            ).float().unsqueeze(1)
            
            # 车辆标记区域给予最高权重
            is_vehicle = torch.logical_or(is_red, is_blue).float()
            vehicle_mask = is_vehicle * self.vehicle_weight
            mask = torch.where(vehicle_mask > 0, vehicle_mask, mask)
        
        # 应用掩码
        weighted_mse = mse * mask
        
        return weighted_mse.mean()

class DINOEncoder(nn.Module):
    """DINOv2特征编码器"""
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        
        # 冻结DINOv2参数
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, image):
        """编码图像"""
        # 确保图像在正确的设备上
        image = image.to(self.device)
        
        if len(image.shape) == 4:  # (B, C, H, W)
            B, C, H, W = image.shape
            encodings = []
            for i in range(B):
                single_image = image[i]
                single_image = single_image.permute(1, 2, 0)
                single_image = (single_image * 255).byte()
                single_image = single_image.cpu().numpy()
                
                inputs = self.processor(images=single_image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(pixel_values)
                
                encodings.append(outputs.last_hidden_state[:, 0, :])
            
            return torch.cat(encodings, dim=0)
        else:  # (C, H, W)
            image = image.permute(1, 2, 0)
            image = (image * 255).byte()
            image = image.cpu().numpy()
            
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(pixel_values)
            
            return outputs.last_hidden_state[:, 0, :]

class FeatureEncoder(nn.Module):
    """特征编码器"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

class ActionEncoder(nn.Module):
    """动作编码器"""
    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

class FeaturePredictor(nn.Module):
    """特征预测器"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.predictor(x)

class BEVPredictionModel(nn.Module):
    def __init__(self, action_dim, history_steps, hidden_dim, device, mode='feature',
                road_weight=3.0, vehicle_weight=15.0, boundary_weight=0.01,
                other_losses_weight=0.05, weight_schedule=None, current_epoch=0):
        super().__init__()
        self.device = device
        self.mode = mode
        self.history_steps = history_steps
        
        # 储存权重配置
        self.road_weight = road_weight
        self.vehicle_weight = vehicle_weight
        self.boundary_weight = boundary_weight
        self.other_losses_weight = other_losses_weight  # 添加对other_losses的权重控制
        self.weight_schedule = weight_schedule
        self.current_epoch = current_epoch
        
        # 初始化各个模块
        self.dino_encoder = DINOEncoder(device)
        self.feature_encoder = FeatureEncoder(768, hidden_dim)## MLP
        self.action_encoder = ActionEncoder(30, hidden_dim)## MLP
        self.feature_predictor = FeaturePredictor(768 + hidden_dim, hidden_dim, 768)## MLP可以考虑改成上面的LSTM
        
        # 特征损失函数
        self.feature_loss_fn = nn.MSELoss()
        
        # 如果是图像生成模式，初始化图像解码器和相关损失函数
        if mode == 'image':
            self.image_decoder = UNetDecoder(
                input_dim=768,
                hidden_dims=[256, 128, 64, 32, 16],
                output_channels=3
            )
            # 使用自定义的带权重的损失函数，支持渐进式权重调整
            self.image_loss_fn = WeightedMSELoss(
                road_weight=self.road_weight,
                vehicle_weight=self.vehicle_weight,
                boundary_weight=self.boundary_weight,
                weight_schedule=self.weight_schedule,
                current_epoch=self.current_epoch
            )
            self.combined_loss_fn = CombinedLoss(alpha=0.5, beta=0.3)
        
        # 初始化优化器
        self.optimizer = AdamW(self.parameters(), lr=1e-4)
        
        # 保存注意力图以便可视化
        self.attention_maps = None
    
    def update_epoch(self, epoch):
        """更新当前epoch，用于渐进式权重调整"""
        self.current_epoch = epoch
        if self.mode == 'image':
            self.image_loss_fn.update_epoch(epoch)
            
            # 更新other_losses的权重
            if self.weight_schedule and epoch in self.weight_schedule:
                self.other_losses_weight = self.weight_schedule[epoch].get("other_losses", self.other_losses_weight)
            
            # 记录当前使用的权重值，便于调试
            current_weights = {
                'vehicle': self.image_loss_fn.vehicle_weight,
                'road': self.image_loss_fn.road_weight,
                'boundary': self.image_loss_fn.boundary_weight,
                'other_losses': self.other_losses_weight
            }
            print(f"Epoch {epoch}: Using weights {current_weights}")
    
    def forward(self, bev_frames, actions, next_frame):
        """前向传播"""
        # 确保输入在正确的设备上
        bev_frames = bev_frames.to(self.device)
        actions = actions.to(self.device)
        next_frame = next_frame.to(self.device)
        
        # 编码历史帧
        history_encodings = []
        for t in range(self.history_steps):
            frame = bev_frames[:, t]
            encoding = self.dino_encoder(frame)
            history_encodings.append(encoding)
        
        # 堆叠历史编码
        history_encodings = torch.stack(history_encodings, dim=1)
        
        # 编码动作
        batch_size, seq_len, _ = actions.shape
        actions_flat = actions.reshape(-1, 30)
        action_encodings = self.action_encoder(actions_flat)
        action_encodings = action_encodings.reshape(batch_size, seq_len, -1)
        
        # 连接特征————没有使用LSTM，feature_predictor可以考虑改成LSTM
        combined_features = torch.cat([history_encodings[:, -1], action_encodings[:, -1]], dim=-1)
        
        # 预测下一帧特征
        pred_features = self.feature_predictor(combined_features)
        
        # 获取目标特征
        target_features = self.dino_encoder(next_frame)
        
        if self.mode == 'feature':
            return pred_features, target_features
        else:
            # 图像生成模式
            pred_image = self.image_decoder(pred_features)
            return pred_features, target_features, pred_image, next_frame
    
    def get_attention_maps(self):
        """获取模型的注意力图"""
        # 确保DINOv2模型返回注意力权重
        if hasattr(self.dino_encoder.model, 'get_last_selfattention'):
            return self.dino_encoder.model.get_last_selfattention()
        return None
    
    def compute_loss(self, pred_features, target_features, pred_image=None, target_image=None, road_mask=None):
        """计算损失函数"""
        # 特征预测损失
        feature_loss = self.feature_loss_fn(pred_features, target_features)
        
        if self.mode == 'feature':
            return feature_loss
        else:
            # 图像生成损失 - 使用自定义的带权重的损失函数
            weighted_image_loss = self.image_loss_fn(pred_image, target_image, road_mask)
            
            # 感知损失和其他损失
            other_losses = self.combined_loss_fn(pred_image, target_image, pred_features, target_features)
            
            # 组合损失 - 使用渐进式other_losses权重
            total_loss = weighted_image_loss + other_losses * self.other_losses_weight
            return total_loss

    def create_road_mask(self, image):
        """创建道路掩码，只过滤上下边界的深灰色区域，保留道路部分（包括道路的浅灰色部分）"""
        # 检测深灰色边界区域 - 较暗且通道间颜色均匀
        pixel_mean = torch.mean(image, dim=1)  # 计算每个像素三个通道的均值
        pixel_std = torch.std(image, dim=1)    # 计算每个像素三个通道的标准差
        
        # 深灰色边界区域特征: 亮度较低且颜色均匀
        is_dark_boundary = torch.logical_and(
            pixel_mean < 0.5,  # 较暗
            pixel_std < 0.03  # 颜色均匀
        )
        
        # 创建掩码: 非深灰色边界(道路+车辆)为1，深灰色边界为0
        road_mask = ~is_dark_boundary
        
        return road_mask.float()