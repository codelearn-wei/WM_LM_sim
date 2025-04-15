import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from typing import Generator, Optional, Union
from dataclasses import dataclass
from stable_baselines3.common.vec_env import VecNormalize

# 定义新的样本数据类
@dataclass
class ConstrainedRolloutBufferSamples:
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    costs: th.Tensor
    cost_returns: th.Tensor
    cost_advantages: th.Tensor

class ConstrainedRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer for constrained reinforcement learning, extending the base RolloutBuffer
    to include cost-related data for Lagrangian methods.

    :param buffer_size: Max number of elements in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param gamma: Discount factor for rewards
    :param cost_gamma: Discount factor for costs (can be different from gamma)
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        cost_gamma: float = 0.99,  # 折扣因子用于 cost
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        self.cost_gamma = cost_gamma
        # 添加 cost 相关的数组
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def reset(self) -> None:
        """重置 buffer，包括 cost 相关的数据"""
        super().reset()
        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        cost: np.ndarray,  # 新增 cost 参数
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        添加 transition 数据，包括 cost。

        :param obs: Observation
        :param action: Action
        :param reward: Reward
        :param cost: Cost associated with the transition
        :param episode_start: Start of episode signal
        :param value: Estimated value of the current state
        :param log_prob: Log probability of the action
        """
        super().add(obs, action, reward, episode_start, value, log_prob)
        # 由于 super().add 已将 pos 自增，这里存储前一个位置的 cost
        self.costs[self.pos - 1] = np.array(cost).copy()

    def compute_returns_and_advantage(
        self,
        last_values: th.Tensor,
        last_cost_values: th.Tensor,  # 新增 last_cost_values 参数
        dones: np.ndarray,
    ) -> None:
        """
        计算 rewards 和 costs 的 returns 和 advantages。

        :param last_values: State value estimation for the last step (for rewards)
        :param last_cost_values: State cost value estimation for the last step
        :param dones: If the last step was a terminal step
        """
        # 计算 rewards 的 returns 和 advantages
        super().compute_returns_and_advantage(last_values, dones)

        # 将 last_cost_values 转换为 numpy
        last_cost_values = last_cost_values.clone().cpu().numpy().flatten()

        # 计算 cost 的 advantages 和 returns
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_cost_values = last_cost_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                # 假设后续会有 cost_values，这里先用 0 占位
                next_cost_values = 0  # 可根据算法调整
            delta = self.costs[step] + self.cost_gamma * next_cost_values * next_non_terminal
            last_gae_lam = delta + self.cost_gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.cost_advantages[step] = last_gae_lam
        self.cost_returns = self.cost_advantages  # 如果有 cost_values，可加上

    def get(self, batch_size: Optional[int] = None) -> Generator[ConstrainedRolloutBufferSamples, None, None]:
        """生成包含 cost 数据的样本"""
        assert self.full, "Buffer must be full before generating samples"
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "costs",
                "cost_returns",
                "cost_advantages",
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> ConstrainedRolloutBufferSamples:
        """返回包含 cost 数据的样本"""
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.costs[batch_inds].flatten(),
            self.cost_returns[batch_inds].flatten(),
            self.cost_advantages[batch_inds].flatten(),
        )
        return ConstrainedRolloutBufferSamples(*tuple(map(self.to_torch, data)))
      
