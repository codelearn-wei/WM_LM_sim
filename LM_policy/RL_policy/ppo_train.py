import gymnasium as gym
from LM_env.envs.merge_env import MergeEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np
from stable_baselines3 import PPO
from config.ppo_config import ppo_config
# 定义make_env创建强化学习环境

def make_env(map_path="LM_env/LM_map/LM_static_map.pkl", render_mode=None, max_episode_steps=500):
    """Factory function to create the environment with specified parameters"""
    env = MergeEnv(
        map_path=map_path,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps
    )
    return env

def initialize_ppo(config=None):
    """
    根据配置文件初始化PPO算法
    
    Args:
        env_name (str): 环境名称
        config (dict): PPO配置参数，如果为None则使用默认配置
    
    Returns:
        PPO: 初始化好的PPO模型
    """
    # 如果没有提供配置，使用默认配置
    if config is None:
        config = ppo_config.copy()
    
 
    # 初始化PPO模型
    model = PPO(
        policy=config["policy"],
        env=env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        clip_range_vf=config["clip_range_vf"],
        normalize_advantage=config["normalize_advantage"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        use_sde=config["use_sde"],
        sde_sample_freq=config["sde_sample_freq"],
        target_kl=config["target_kl"],
        stats_window_size=config["stats_window_size"],
        tensorboard_log=config["tensorboard_log"],
        verbose=config["verbose"],
        seed=config["seed"],
        device=config["device"],
        policy_kwargs=config["policy_kwargs"],
        _init_setup_model=True
    )
    
    return model


render_mode = 'human' 
episodes = 100
max_steps = 200
env = make_env(render_mode=render_mode)

# 初始化PPO模型参数
model = initialize_ppo()
# 训练模型
model.learn(total_timesteps=10000)


# 设定PPO算法参数（基于配置文件设计）



# # 检测环境是否符合gymnasium的标准
# # check_env(env)
