import gymnasium as gym
from LM_env.envs.merge_env import MergeEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
# 定义make_env创建强化学习环境


def make_env(map_path="LM_env/LM_map/LM_static_map.pkl", render_mode=None, max_episode_steps=500):
    """Factory function to create the environment with specified parameters"""
    env = MergeEnv(
        map_path=map_path,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps
    )
    return env



render_mode = 'human' 
episodes = 100
max_steps = 200
env = make_env(render_mode=render_mode)
# check_env(env)
# 初始化PPO模型参数
model = SAC("MlpPolicy", env, verbose=1)
# 训练模型
model.learn(total_timesteps=10000)




