import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Optional, Dict, Any, Union, List, Tuple
from torch.nn import functional as F
from gymnasium import spaces
from functools import partial

import torch as th
from torch import nn
from typing import Optional, Union, Tuple
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
class LagrangianActorCriticPolicy(ActorCriticPolicy):
    """
    支持拉格朗日约束的 Actor-Critic 策略网络，用于 PPO。
    在原有基础上增加 cost value network，用于估计 cost 的值函数。

    :param observation_space: 观测空间
    :param action_space: 动作空间
    :param lr_schedule: 学习率调度器
    :param net_arch: 网络架构
    :param activation_fn: 激活函数
    :param ortho_init: 是否使用正交初始化
    :param use_sde: 是否使用状态依赖噪声探索
    :param log_std_init: 对数标准差初始值
    :param full_std: 是否使用完整的标准差参数
    :param use_expln: 是否使用 expln 函数替代 exp
    :param squash_output: 是否使用 tanh 压缩输出
    :param features_extractor_class: 特征提取器类
    :param features_extractor_kwargs: 特征提取器参数
    :param share_features_extractor: 是否共享特征提取器
    :param normalize_images: 是否标准化图像
    :param optimizer_class: 优化器类
    :param optimizer_kwargs: 优化器参数
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        # 添加 cost value network，与 value_net 结构相同但独立
        self.cost_value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # 如果使用正交初始化，则对 cost_value_net 也应用
        if self.ortho_init:
            self.cost_value_net.apply(partial(self.init_weights, gain=1))

    def _build(self, lr_schedule: Schedule) -> None:
        """
        构建网络和优化器，添加 cost_value_net 的初始化。
        
        :param lr_schedule: 学习率调度器
        """
        super()._build(lr_schedule)
        # cost_value_net 已在前面的 __init__ 中创建并初始化，此处无需重复

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        前向传播，同时返回动作、reward value、cost value 和动作的对数概率。

        :param obs: 观测
        :param deterministic: 是否使用确定性动作
        :return: (actions, values, cost_values, log_prob)
        """
        # 提取特征
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # 计算 reward value 和 cost value
        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_vf)

        # 获取动作分布并采样动作
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        return actions, values, cost_values, log_prob

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor], th.Tensor]:
        """
        评估给定观测和动作的 value、log_prob、entropy 和 cost value。

        :param obs: 观测
        :param actions: 动作
        :return: (values, log_prob, entropy, cost_values)
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_vf)
        entropy = distribution.entropy()

        return values, log_prob, entropy, cost_values

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        预测给定观测的 reward value 和 cost value。

        :param obs: 观测
        :return: (values, cost_values)
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
  
        return values

    def predict_cost_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        单独预测 cost value（可选方法）。

        :param obs: 观测
        :return: cost_values
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.cost_value_net(latent_vf)