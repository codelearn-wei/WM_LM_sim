import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

class DINOFeatureExtractor:
    """DINOv2特征提取器，仅用于推理"""
    def __init__(self, device):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        
        # 冻结DINOv2参数
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def extract_features(self, image):
        """提取图像特征"""
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