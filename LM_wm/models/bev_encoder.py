import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model, AutoModel
from LM_wm.configs.config import CUDA_DEVICE
from utils.image_utils import preprocess_image_for_model

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

class BEVPredictionModel(nn.Module):
    """BEV预测模型"""
    def __init__(self, action_dim, history_steps, hidden_dim, device):
        super().__init__()
        self.device = device
        self.history_steps = history_steps
        
        # 初始化DINOv2模型和处理器
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base",
            use_fast=False
        )
        self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base")
        
        # 将DINOv2模型移到指定设备
        self.dino_model = self.dino_model.to(device)
        self.dino_model.eval()  # 设置为评估模式
        
        # 冻结DINOv2参数
        for param in self.dino_model.parameters():
            param.requires_grad = False
            
        # 获取DINOv2的输出维度
        dino_output_dim = self.dino_model.config.hidden_size
        
        # 其他层保持不变
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)
        
        self.lstm = nn.LSTM(
            input_size=dino_output_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        ).to(device)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dino_output_dim)
        ).to(device)
        
    def encode_image(self, image):
        """
        使用DINOv2编码图像
        """
        # 确保图像在正确的设备上
        image = image.to(self.device)
        
        # 预处理图像，保持宽高比
        processed_image = preprocess_image_for_model(image)
        processed_image = processed_image.to(self.device)
        
        # 使用处理器处理图像
        inputs = self.processor(images=processed_image, return_tensors="pt")
        
        # 将处理后的输入移到正确的设备上
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 使用DINOv2进行推理
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            
        return outputs.last_hidden_state[:, 0, :]
    
    def forward(self, bev_frames, actions, next_frame):
        batch_size = bev_frames.size(0)
        
        # 确保所有输入都在正确的设备上
        bev_frames = bev_frames.to(self.device)
        actions = actions.to(self.device)
        next_frame = next_frame.to(self.device)
        
        # 编码历史帧
        history_encodings = []
        for t in range(self.history_steps):
            frame = bev_frames[:, t]
            encoding = self.encode_image(frame)
            history_encodings.append(encoding)
        
        # 堆叠历史编码
        history_encodings = torch.stack(history_encodings, dim=1)
        
        # 编码动作
        action_encodings = self.action_encoder(actions)
        
        # 连接历史编码和动作编码
        lstm_input = torch.cat([history_encodings, action_encodings], dim=-1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(lstm_input)
        
        # 解码预测
        pred_encoding = self.decoder(lstm_out[:, -1])
        
        # 编码目标帧
        target_encoding = self.encode_image(next_frame)
        
        return pred_encoding, target_encoding
    
    def compute_loss(self, pred_encoding, target_encoding):
        """
        计算预测编码和目标编码之间的损失
        """
        return nn.MSELoss()(pred_encoding, target_encoding)