# IRL可以捕捉价值取向和社会偏好
# TODO:能否开发成实时在线辨识的算法？
# 正向训练一个动作选择器
import torch
from reward import NeuralNetworkRewardFunction

class InverseReinforcementLearning:
    def __init__(self, features, human_features, reward_function, lr=0.05, lam=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
                 n_iters=2000, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 将features和human_features转为Tensor，并移动到指定的device
        self.features = [torch.tensor(feature[0], dtype=torch.float32, device=self.device) for feature in features]
        self.human_features = torch.tensor(human_features, dtype=torch.float32, device=self.device)
        self.feature_expectations = 0
        self.reward_function = reward_function
        self.reward_function.theta = reward_function.theta.clone().detach().to(self.device)
        self.reward_function.theta.requires_grad_(True)  # 如果需要梯度
        self.lr = lr
        self.lam = lam
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.n_iters = n_iters
        self.pm = None
        self.pv = None
        self.iteration = 0

    def compute_rewards_(self):
        """使用传入的奖励函数计算奖励，确保输入是Tensor"""
        return torch.stack([self.reward_function.compute(feature) for feature in self.features])

    def compute_probabilities(self, rewards):
        """使用softmax计算选择每条轨迹的概率"""
        exp_rewards = torch.exp(rewards - torch.max(rewards))  # 防止数值溢出
        return exp_rewards / torch.sum(exp_rewards)

    def calculate_gradients(self, probs):
        """使用Tensor操作计算theta的梯度"""
        print("概率",probs)
        traj_features = torch.stack([feature for feature in self.features])
        self.feature_expectations = torch.matmul(probs, traj_features)
        gradient = self.human_features - self.feature_expectations - 2 * self.lam * self.reward_function.theta
        return gradient

    def update_theta(self, gradient):
        """使用Adam-like优化器更新theta"""
        if self.pm is None or self.pv is None:
            self.pm = torch.zeros_like(gradient, device=self.device)
            self.pv = torch.zeros_like(gradient, device=self.device)
        self.pm = self.beta1 * self.pm + (1 - self.beta1) * gradient
        self.pv = self.beta2 * self.pv + (1 - self.beta2) * (gradient ** 2)
        mhat = self.pm / (1 - self.beta1 ** (self.iteration + 1))
        vhat = self.pv / (1 - self.beta2 ** (self.iteration + 1))
        update_vec = mhat / (torch.sqrt(vhat) + self.eps)
        self.reward_function.theta = self.reward_function.theta + self.lr * update_vec

    def run(self):
        """执行IRL过程指定的迭代次数"""
        for self.iteration in range(self.n_iters):
            rewards = self.compute_rewards_()
            probs = self.compute_probabilities(rewards)
            if isinstance(self.reward_function, NeuralNetworkRewardFunction):
                traj_features = torch.stack([feature for feature in self.features])
                self.feature_expectations = torch.matmul(probs, traj_features)
                loss = self.reward_function.train_custom_reward_function(self.human_features, self.feature_expectations)
                print(f"Iteration {self.iteration + 1}: Loss = {loss}")
            else:
                gradient = self.calculate_gradients(probs)
                self.update_theta(gradient)