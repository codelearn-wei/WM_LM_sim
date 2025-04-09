# ppo_config.py
ppo_config = {
    "policy": "MlpPolicy",
    "learning_rate": 0.0003,
    "n_steps": 300,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": None,
    "stats_window_size": 100,
    "tensorboard_log": None,
    "verbose": 1,
    "seed": None,
    "device": "auto",
    "policy_kwargs": {
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])]
    }
}