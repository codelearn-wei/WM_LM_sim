import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
from LM_wm.configs.config import IMAGE_SIZE
from LM_wm.utils.image_utils import maintain_aspect_ratio_resize

class LMTrainingDataset(Dataset):
    def __init__(self, data_dir: str, history_steps: int = 20, focus_on_road: bool = True):
        """
        初始化训练数据集

        Args:
            data_dir (str): 数据目录路径
            history_steps (int, optional): 历史步长. 默认为 20.
            focus_on_road (bool, optional): 是否强制模型关注道路区域. 默认为 True.
        """
        self.data_dir = Path(data_dir)
        self.history_steps = history_steps
        self.focus_on_road = focus_on_road
        self.sequences = self._build_sequences()
        
    def _build_sequences(self):
        """构建训练序列"""
        sequences = []
        
        # 遍历所有场景目录
        for scene_dir in self.data_dir.iterdir():
            if not scene_dir.is_dir():
                continue
                
            # 加载动作数据
            action_file = scene_dir / 'actions.npz'
            if not action_file.exists():
                continue
                
            action_data = np.load(action_file, allow_pickle=True)
            frame_actions = action_data['frame_actions'].item()
            
            # 获取所有帧
            frames = sorted(list(scene_dir.glob("frame_*.png")))
            frame_numbers = [int(f.stem.split('_')[1]) for f in frames]
            
            # 构建序列
            for i in range(len(frames) - self.history_steps - 1):
                start_frame = frame_numbers[i]
                end_frame = frame_numbers[i + self.history_steps]
                
                # 确保序列连续
                if end_frame - start_frame != self.history_steps:
                    continue
                
                seq = {
                    'scene_dir': scene_dir,
                    'start_frame': start_frame,
                    'frames': frames[i:i + self.history_steps + 1],
                    'actions': [frame_actions[j] for j in range(start_frame, end_frame + 1)]
                }
                sequences.append(seq)
        
        return sequences
    
    def _process_image(self, image_path):
        """处理图像"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 使用maintain_aspect_ratio_resize调整图像尺寸，保持宽高比
        img = maintain_aspect_ratio_resize(img, target_size=IMAGE_SIZE)
        # 确保图像是3通道的
        if len(img.shape) != 3:
            raise ValueError(f"图像维度不正确: {img.shape}")
        # 直接转换为tensor并归一化，确保维度为(C, H, W)
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return img
    
    def _create_road_mask(self, img):
        """
        创建道路掩码，仅识别上下边界的深灰色区域(背景)，保留道路区域(包括道路的浅灰色部分)
        
        Args:
            img (torch.Tensor): 图像张量，形状为 [C, H, W]
            
        Returns:
            torch.Tensor: 掩码张量，形状为 [H, W]，道路区域为1，深灰色边界背景为0
        """
        # 将图像转为 [H, W, C] 用于处理
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        # 根据颜色识别区域
        # 道路区域: 较亮的灰色，一般亮度值在0.55-0.75之间（归一化后）
        # 深灰色边界: 较暗的灰色，一般亮度值在0.4-0.5之间（归一化后）
        # 车辆标记: 鲜明的红色和蓝色
        
        # 计算像素亮度
        pixel_mean = np.mean(img_np, axis=2)  # 计算每个像素三个通道的均值
        
        # 识别深灰色边界区域 - 亮度较低且颜色均匀
        is_dark_gray = np.logical_and(
            pixel_mean < 0.5,  # 深灰色区域较暗
            np.std(img_np, axis=2) < 0.03  # 颜色非常均匀
        )
        
        # 创建掩码: 非深灰色边界(道路+车辆)为1，深灰色边界为0
        road_mask = ~is_dark_gray
        
        return torch.from_numpy(road_mask).float()
    
    def _apply_focus_mask(self, img):
        """
        应用专注掩码，强制模型只忽略上下深灰色边界区域
        
        Args:
            img (torch.Tensor): 图像张量，形状为 [C, H, W]
            
        Returns:
            torch.Tensor: 处理后的图像张量，形状为 [C, H, W]
        """
        if not self.focus_on_road:
            return img
            
        # 创建道路掩码 - 只识别深灰色边界区域
        road_mask = self._create_road_mask(img)
        
        # 创建随机噪声，用于替换深灰色边界区域
        # 使用较小的噪声范围，使模型能够轻松识别这是不重要的区域
        noise = torch.rand_like(img) * 0.05 + 0.48  # 0.48-0.53的随机噪声
        
        # 应用掩码: 保留道路区域，深灰色边界区域替换为随机噪声
        masked_img = img.clone()
        for c in range(3):  # 对每个通道应用掩码
            masked_img[c] = img[c] * road_mask + noise[c] * (1 - road_mask)
        
        return masked_img
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """获取训练样本"""
        seq = self.sequences[idx]
        
        # 读取并处理BEV图像序列
        bev_frames = []
        for frame_path in seq['frames'][:-1]:  # 不包括最后一帧
            img = self._process_image(frame_path)
            # 对历史帧应用专注掩码
            img = self._apply_focus_mask(img)
            bev_frames.append(img)
        bev_frames = torch.stack(bev_frames)
        
        # 读取下一帧 - 不应用掩码，因为这是目标
        next_frame = self._process_image(seq['frames'][-1])
        
        # 处理动作序列
        actions = []
        for frame_action in seq['actions'][:-1]:  # 不包括最后一帧的动作
            action_tensor = torch.from_numpy(frame_action).float()
            actions.append(action_tensor)
        actions = torch.stack(actions)
        
        # 添加道路掩码用于训练
        road_mask = self._create_road_mask(next_frame)
        
        return {
            'bev_frames': bev_frames,
            'actions': actions,
            'next_frame': next_frame,
            'road_mask': road_mask
        }
