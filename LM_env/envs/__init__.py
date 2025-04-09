 
 #! 需要使用pip安装并注册（还没有完成注册，等调试好了再继续注册）
 #TODO：对最终的环境完成注册
 
from LM_env.envs.merge_env import MergeEnv
from gymnasium.envs.registration import register

register(
    id='MergeEnv-v0',  # 环境的唯一标识符
    entry_point='merge_env.envs:MergeEnv',  # 如果在同一文件运行，指向当前模块的 MergeEnv 类
    max_episode_steps=500,  # 与您的 max_episode_steps 参数一致
    kwargs={
        'map_path': "LM_env/LM_map/LM_static_map.pkl",
        'render_mode': None,
        'dt': 0.1,
        'max_episode_steps': 500,
        'other_vehicle_strategy': 'interactive'
    }
)