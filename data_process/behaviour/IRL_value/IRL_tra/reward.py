import torch
import torch.nn as nn
import torch.optim as optim

class RewardFunction:
    """奖励函数的接口"""
    def __init__(self, device='cpu'):
        self.device = device

    def compute(self, features):
        raise NotImplementedError

class LinearRewardFunction(RewardFunction):
    """线性奖励函数的实现，使用Tensor运算"""

    def __init__(self, theta_initial, device='cpu'):
        super().__init__(device=device)
        self.theta = torch.tensor(theta_initial, dtype=torch.float32, device=self.device)  # 确保theta是一个Tensor并且在正确的设备上

    def compute(self, feature):
        return torch.sum(feature * self.theta)  # 计算奖励值

class NeuralNetworkRewardFunction(RewardFunction, nn.Module):
    """基于神经网络的奖励函数"""

    def __init__(self, input_dim, device='cpu'):
        RewardFunction.__init__(self, device=device)
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)  # 将网络层移到指定的设备
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)  # 定义优化器

    def compute(self, feature):
        feature = feature.to(self.device)  # 确保输入特征也在正确的设备上
        output = self.layers(feature)
        return output.squeeze()  # 移除所有单一维度

    def train_custom_reward_function(self, human_features, feature_expectations):
        human_features = human_features.to(self.device)  # 确保数据在正确的设备上
        feature_expectations = feature_expectations.to(self.device)
        loss = torch.norm(human_features - feature_expectations)  # 自定义损失函数
        self.optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数
        return loss.item()  # 返回损失值