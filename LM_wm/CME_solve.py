from LM_wm.models.feature_extractor import DINOFeatureExtractor
from LM_wm.models.wm_policy import WM_Policy
from LM_wm.configs.policy_config import PolicyConfig
import numpy as np
import torch
import cv2
from pathlib import Path
from LM_wm.utils.image_utils import maintain_aspect_ratio_resize
from LM_wm.configs.config import IMAGE_SIZE
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class Env:
    def __init__(self, data_dir: str, scene_name: str, history_steps: int = 20, target_frame: int = 40, image_size: tuple = IMAGE_SIZE):
        """
        初始化环境类，专注于单一场景的提取和规划
        
        Args:
            data_dir (str): 数据根目录路径
            scene_name (str): 要模拟的场景名称
            history_steps (int): 历史帧数量
            target_frame (int): 目标帧索引（从0开始计数）
            image_size (tuple): 目标图像尺寸
        """
        self.data_dir = Path(data_dir) / scene_name
        self.history_steps = history_steps
        self.target_frame = target_frame
        self.image_size = image_size

        # 加载所有图像帧路径
        self.frame_paths = sorted(self.data_dir.glob("frame_*.png"))
        self.total_frames = len(self.frame_paths)
        
        if self.total_frames <= self.target_frame:
            raise ValueError(f"场景 {scene_name} 的帧数不足: {self.total_frames}, 需要至少 {self.target_frame+1} 帧")
            
        if self.target_frame <= self.history_steps:
            raise ValueError(f"目标帧 {target_frame} 必须大于历史帧数 {history_steps}")
            
        self.current_index = history_steps  # 从历史帧数后开始

        # 预加载所有需要用到的帧以提高性能
        self.frames = {}
        self._preload_frames()
        
        print(f"环境初始化完成，总帧数: {self.total_frames}, 目标帧: {self.target_frame}")

    def _preload_frames(self):
        """预加载所有需要用到的帧以提高性能"""
        print("预加载帧...")
        for i in range(self.target_frame + 1):
            self.frames[i] = self._load_frame(self.frame_paths[i])
        print("预加载完成")

    def reset(self):
        """重置环境到初始状态，返回初始观测值"""
        self.current_index = self.history_steps
        return self.get_obs()

    def get_obs(self):
        """获取当前历史帧序列作为观测值"""
        if self.current_index < self.history_steps:
            raise ValueError("当前索引不足以形成历史帧序列")
        
        # 获取最近的 history_steps 个帧
        start_idx = self.current_index - self.history_steps
        end_idx = self.current_index
        frames = [self.frames[i] for i in range(start_idx, end_idx)]
        return torch.stack(frames, dim=0)  # 形状: [T, C, H, W]

    def get_goal_obs(self):
        """获取目标观测值（第target_frame帧）"""
        return self.frames[self.target_frame]  # 形状: [C, H, W]

    def step(self, action=None):
        """
        执行一步动作，更新环境状态
        """
        self.current_index += 1
        done = self.current_index >= self.target_frame
        obs = self.get_obs() if not done else None
        reward = 0
        info = {}
        return obs, reward, done, info

    def _load_frame(self, frame_path):
        """加载并处理单帧图像"""
        img = cv2.imread(str(frame_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = maintain_aspect_ratio_resize(img, target_size=self.image_size)
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # 形状: [C, H, W]
        return img

    def get_actual_trajectory(self):
        """获取从起始帧到目标帧的实际轨迹"""
        return [i for i in range(self.history_steps, self.target_frame + 1)]


# 使用编码器 - 添加缓存以避免重复计算
class CachedEncoder:
    def __init__(self, encoder, device):
        self.encoder = encoder
        self.device = device
        self.cache = {}  # 缓存编码结果
        
    def encode_observation(self, obs):
        """编码观测值，支持历史帧序列或单帧，使用缓存提高性能"""
        if obs.dim() == 4:  # [T, C, H, W]
            features = []
            for t in range(obs.size(0)):
                frame = obs[t].permute(1, 2, 0).numpy()  # [H, W, C]
                # 使用更可靠的缓存键 - 使用帧的哈希值
                frame_id = hash(frame.tobytes())
                
                if frame_id not in self.cache:
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                    with torch.no_grad():
                        z = self.encoder.extract_features(frame_tensor)  # [1, dino_dim]
                    self.cache[frame_id] = z.squeeze(0).cpu().numpy()
                
                features.append(self.cache[frame_id])
            
            return np.stack(features, axis=0)  # [T, dino_dim]
        
        elif obs.dim() == 3:  # [C, H, W]
            frame = obs.permute(1, 2, 0).numpy()  # [H, W, C]
            frame_id = hash(frame.tobytes())
            
            if frame_id not in self.cache:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    z = self.encoder.extract_features(frame_tensor)  # [1, dino_dim]
                self.cache[frame_id] = z.squeeze(0).cpu().numpy()
            
            return self.cache[frame_id]  # [dino_dim]
        else:
            raise ValueError("Unexpected observation shape")

# 简化版本的预测轨迹函数 - 移除批处理
def predict_trajectory(history_z, action_sequence, policy, device):
    """预测潜在状态轨迹，基于历史特征"""
    trajectory = [history_z[-1]]  # 起始状态
    current_history = history_z.copy()  # [T, dino_dim]
    
    for t in range(len(action_sequence)):
        # 获取当前动作
        a = action_sequence[t]
        
        # 转换为张量
        history_tensor = torch.from_numpy(current_history).float().unsqueeze(0).to(device)  # [1, T, dino_dim]
        a_tensor = torch.from_numpy(a).float().unsqueeze(0).to(device)  # [1, 1, action_dim]
        
        # 预测下一个状态
        with torch.no_grad():
            pred_z = policy.predict_next_state(history_tensor, a_tensor)  # [1, dino_dim]
        
        # 转换为numpy
        next_z = pred_z.squeeze(0).cpu().numpy()  # [dino_dim]
        
        # 添加到轨迹
        trajectory.append(next_z)
        
        # 更新历史序列，移除最早的状态，添加新预测状态
        current_history = np.vstack([current_history[1:], next_z])
    
    return trajectory

# 简化版本的CEM规划 - 移除批处理
def cem_plan(history_z, zg, policy, T, N, K, num_iters, action_dim, action_range=(-1, 1), device='cuda'):
    """CEM 规划函数 - 序列化评估"""
    # 初始化动作序列的均值和标准差
    mean = np.zeros((T, action_dim))
    std = np.ones((T, action_dim))
    
    # 进度显示
    pbar = tqdm(range(num_iters), desc="CEM规划")
    
    for _ in pbar:
        # 采样 N 个动作序列
        noise = np.random.randn(N, T, action_dim) * std[np.newaxis, :, :]
        action_sequences = mean[np.newaxis, :, :] + noise
        action_sequences = np.clip(action_sequences, action_range[0], action_range[1])
        
        # 评估每个序列
        costs = np.zeros(N)
        for i in range(N):
            # 预测每个动作序列的轨迹
            traj = predict_trajectory(history_z, action_sequences[i], policy, device)
            zT = traj[-1]  # 最终状态
            
            # 计算与目标状态的MSE
            costs[i] = np.mean((zT - zg) ** 2)
        
        # 显示当前最佳成本
        pbar.set_postfix({"min_cost": costs.min()})
        
        # 选择前 K 个精英序列
        elite_indices = np.argsort(costs)[:K]
        elite_sequences = action_sequences[elite_indices]
        
        # 更新均值和标准差
        mean = np.mean(elite_sequences, axis=0)  # [T, action_dim]
        std = np.std(elite_sequences, axis=0) + 1e-6  # 防止标准差为0

    return mean  # 返回优化后的动作序列


def visualize_results(actual_z_trajectory, planned_z_trajectory, goal_z, action_sequence, title="轨迹对比"):
    """可视化实际轨迹、规划轨迹和动作序列"""
    # 创建一个包含两个子图的画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # 绘制轨迹在潜在空间中的前两个维度 (第一个子图)
    dim1, dim2 = 0, 1  # 可以修改为其他维度
    
    # 实际轨迹
    ax1.plot([z[dim1] for z in actual_z_trajectory], [z[dim2] for z in actual_z_trajectory], 
             'b-', linewidth=2, label="实际轨迹")
    
    # 起始点
    ax1.scatter([actual_z_trajectory[0][dim1]], [actual_z_trajectory[0][dim2]], 
                color='green', s=100, label="起始点")
    
    # 规划轨迹
    ax1.plot([z[dim1] for z in planned_z_trajectory], [z[dim2] for z in planned_z_trajectory], 
             'r--', linewidth=2, label="规划轨迹")
    
    # 目标点
    ax1.scatter([goal_z[dim1]], [goal_z[dim2]], color='red', s=100, label="目标点")
    
    ax1.set_title(f"{title} - 潜在空间轨迹")
    ax1.set_xlabel(f"潜在维度 {dim1}")
    ax1.set_ylabel(f"潜在维度 {dim2}")
    ax1.legend()
    ax1.grid(True)
    
    # 绘制动作序列 (第二个子图)
    action_dim = action_sequence.shape[1]
    timesteps = np.arange(len(action_sequence))
    
    # 为每个动作维度绘制一条线
    for d in range(action_dim):
        ax2.plot(timesteps, action_sequence[:, d], label=f"动作维度 {d}")
    
    ax2.set_title("规划动作序列")
    ax2.set_xlabel("时间步")
    ax2.set_ylabel("动作值")
    ax2.legend()
    ax2.grid(True)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()
    
    # 计算误差
    final_planned_z = planned_z_trajectory[-1]
    mse = np.mean((final_planned_z - goal_z) ** 2)
    print(f"规划终点与目标点的MSE: {mse:.6f}")
    
    # 分析动作序列
    action_mean = np.mean(action_sequence, axis=0)
    action_std = np.std(action_sequence, axis=0)
    print(f"动作序列统计:\n平均值: {action_mean}\n标准差: {action_std}")


def load_model(checkpoint_path, config):
    """加载训练好的模型"""
    model = WM_Policy(
        action_dim=config.action_dim,
        history_steps=config.history_steps,
        hidden_dim=config.hidden_dim,
        device=config.device,
        mode='feature'
    ).to(config.device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    model.eval()
    
    print(f"成功加载模型检查点: {checkpoint_path}")
    print(f"模型训练到第 {checkpoint['epoch'] + 1} 个epoch，损失值: {checkpoint['loss']:.6f}")
    
    return model


def main():
    # 记录开始时间
    start_time = time.time()
    
    # 设置配置
    config = PolicyConfig()
    device = torch.device(config.device)
    
    # 初始化特征提取器和世界模型
    feature_extractor = DINOFeatureExtractor(device=device)
    cached_encoder = CachedEncoder(feature_extractor, device)
    
    # 加载预训练权重
    policy_weights_path = "LM_wm/logs/policy/checkpoints/best_model.pth"
    wm_policy = load_model(policy_weights_path, config)
    
    # 初始化环境 - 指定目标帧为40
    env = Env(data_dir="LM_wm/training_data", scene_name="vehicle_tracks_000", 
              history_steps=config.history_steps, target_frame=40)
    
    # 重置环境并获取初始观测和目标观测
    o0 = env.reset()
    og = env.get_goal_obs()
    
    print("编码初始观测和目标观测...")
    history_z = cached_encoder.encode_observation(o0)
    zg = cached_encoder.encode_observation(og)
    print(f"编码完成. 历史特征形状: {history_z.shape}, 目标特征形状: {zg.shape}")
    
    # 计算从起始状态到目标状态所需的步数
    planning_steps = env.target_frame - env.current_index
    print(f"规划步数: {planning_steps}")
    
    # 使用CEM规划最优动作序列
    print("开始CEM规划...")
    cem_start_time = time.time()
    action_sequence = cem_plan(
        history_z, zg, wm_policy, 
        T=planning_steps,          # 规划步数 
        N=100,                     # 采样数量
        K=20,                      # 精英数量
        num_iters=100,             # 迭代次数
        action_dim=config.action_dim,
        device=config.device
    )
    cem_time = time.time() - cem_start_time
    print(f"CEM规划完成，耗时: {cem_time:.2f}秒")
    
    # 使用规划的动作序列预测轨迹
    print("预测规划轨迹...")
    planned_trajectory_z = predict_trajectory(history_z, action_sequence, wm_policy, device)
    
    # 获取实际轨迹
    print("获取实际轨迹...")
    actual_trajectory_z = [history_z[-1]]  # 起始点
    current_obs = o0
    env.reset()  # 重置环境
    
    # 收集实际轨迹
    done = False
    while not done:
        obs, _, done, _ = env.step()
        if not done:
            current_obs = obs
            state_z = cached_encoder.encode_observation(current_obs)[-1]  # 取最新的状态
            actual_trajectory_z.append(state_z)
    
    # 添加最终目标状态
    actual_trajectory_z.append(zg)
    
    # 可视化结果（新增加了动作序列的可视化）
    visualize_results(actual_trajectory_z, planned_trajectory_z, zg, action_sequence, "单场景轨迹规划 (0-40帧)")
    
    # 打印总耗时
    total_time = time.time() - start_time
    print(f"任务完成! 总耗时: {total_time:.2f}秒")


# 添加多场景批量规划和可视化功能
def batch_planning(data_dir, scene_names, target_frames, config):
    """
    批量规划并可视化多个场景的轨迹
    
    Args:
        data_dir (str): 数据根目录
        scene_names (list): 场景名称列表
        target_frames (list): 每个场景对应的目标帧
        config (PolicyConfig): 配置对象
    """
    device = torch.device(config.device)
    
    # 初始化模型
    feature_extractor = DINOFeatureExtractor(device=device)
    cached_encoder = CachedEncoder(feature_extractor, device)
    
    # 加载预训练权重
    policy_weights_path = "LM_wm/logs/policy/checkpoints/best_model.pth"
    wm_policy = load_model(policy_weights_path, config)
    
    # 为每个场景执行规划
    for scene_name, target_frame in zip(scene_names, target_frames):
        print(f"\n处理场景: {scene_name}, 目标帧: {target_frame}")
        
        # 初始化环境
        env = Env(data_dir=data_dir, scene_name=scene_name, 
                  history_steps=config.history_steps, target_frame=target_frame)
        
        # 重置环境并获取初始观测和目标观测
        o0 = env.reset()
        og = env.get_goal_obs()
        
        # 编码
        history_z = cached_encoder.encode_observation(o0)
        zg = cached_encoder.encode_observation(og)
        
        # 规划步数
        planning_steps = env.target_frame - env.current_index
        
        # CEM规划
        action_sequence = cem_plan(
            history_z, zg, wm_policy, 
            T=planning_steps, 
            N=100, 
            K=20, 
            num_iters=50,  # 减少迭代次数以加快批处理
            action_dim=config.action_dim,
            device=config.device
        )
        
        # 预测轨迹
        planned_trajectory_z = predict_trajectory(history_z, action_sequence, wm_policy, device)
        
        # 获取实际轨迹
        actual_trajectory_z = [history_z[-1]]
        env.reset()
        
        done = False
        while not done:
            obs, _, done, _ = env.step()
            if not done:
                state_z = cached_encoder.encode_observation(obs)[-1]
                actual_trajectory_z.append(state_z)
        
        actual_trajectory_z.append(zg)
        
        # 可视化
        visualize_results(
            actual_trajectory_z, 
            planned_trajectory_z, 
            zg, 
            action_sequence,
            f"场景_{scene_name}_帧{target_frame}"
        )


if __name__ == "__main__":
    # 执行单场景规划
    main()
    
    # 如果需要批量规划多个场景，可以取消下面的注释
    # scene_names = ["vehicle_tracks_000", "vehicle_tracks_001", "vehicle_tracks_002"]
    # target_frames = [40, 35, 30]
    # config = PolicyConfig()
    # batch_planning("LM_wm/training_data", scene_names, target_frames, config)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     # 加载模型
# from LM_wm.models.feature_extractor import DINOFeatureExtractor  # 用于隐状态的推理
# from LM_wm.models.wm_policy import WM_Policy  # policy模型
# from LM_wm.configs.policy_config import PolicyConfig
# import numpy as np
# import torch
# import cv2
# from pathlib import Path
# from LM_wm.utils.image_utils import maintain_aspect_ratio_resize
# from LM_wm.configs.config import IMAGE_SIZE
# import matplotlib.pyplot as plt

# class Env:
#     def __init__(self, data_dir: str, scene_name: str, history_steps: int = 20, predict_steps: int = 1, image_size: tuple = IMAGE_SIZE):
#         """
#         初始化环境类

#         Args:
#             data_dir (str): 数据根目录路径
#             scene_name (str): 要模拟的场景名称
#             history_steps (int, optional): 历史帧数量. 默认为 20.
#             predict_steps (int, optional): 预测步数（目标帧偏移）. 默认为 1.
#             image_size (tuple, optional): 目标图像尺寸. 默认为 IMAGE_SIZE.
#         """
#         self.data_dir = Path(data_dir) / scene_name
#         self.history_steps = history_steps
#         self.predict_steps = predict_steps
#         self.image_size = image_size

#         # 加载所有图像帧路径
#         self.frame_paths = sorted(self.data_dir.glob("frame_*.png"))
#         self.total_frames = len(self.frame_paths)
#         self.current_index = 10

#         # 加载动作数据
#         action_file = self.data_dir / 'actions.npz'
#         if action_file.exists():
#             action_data = np.load(action_file, allow_pickle=True)
#             self.frame_actions = action_data['frame_actions'].item()
#         else:
#             self.frame_actions = None

#         if self.total_frames < self.history_steps + self.predict_steps:
#             raise ValueError(f"场景 {scene_name} 的帧数不足: {self.total_frames}")

#     def reset(self):
#         """重置环境到初始状态，返回初始观测值"""
#         self.current_index = self.history_steps
#         return self.get_obs()

#     def get_obs(self):
#         """获取当前历史帧序列作为观测值"""
#         if self.current_index < self.history_steps:
#             raise ValueError("当前索引不足以形成历史帧序列")
        
#         # 获取最近的 history_steps 个帧
#         start_idx = self.current_index - self.history_steps
#         end_idx = self.current_index
#         frames = [self._load_frame(self.frame_paths[i]) for i in range(start_idx, end_idx)]
#         return torch.stack(frames, dim=0)  # 形状: [T, C, H, W]

#     def get_goal_obs(self):
#         """获取目标观测值（当前帧后的 predict_steps 帧）"""
#         goal_index = self.current_index + self.predict_steps
#         if goal_index >= self.total_frames:
#             raise ValueError("目标帧索引超出范围")
#         return self._load_frame(self.frame_paths[goal_index])  # 形状: [C, H, W]

#     def step(self, action=None):
#         """
#         执行一步动作，更新环境状态

#         Args:
#             action (optional): 代理提供的动作（当前实现中未使用，可扩展）

#         Returns:
#             tuple: (obs, reward, done, info)
#                 - obs: 新观测值
#                 - reward: 奖励（当前为0，可自定义）
#                 - done: 是否结束
#                 - info: 附加信息
#         """
#         self.current_index += 1
#         done = self.current_index >= self.total_frames - self.predict_steps
#         obs = self.get_obs() if not done else None
#         reward = 0  # 可根据任务定义奖励
#         info = {}
#         return obs, reward, done, info

#     def _load_frame(self, frame_path):
#         """加载并处理单帧图像"""
#         img = cv2.imread(str(frame_path))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = maintain_aspect_ratio_resize(img, target_size=self.image_size)
#         img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # 形状: [C, H, W]
#         return img

#     def is_done(self):
#         """检查环境是否结束"""
#         return self.current_index >= self.total_frames - self.predict_steps

# # 使用编码器
# def encode_observation(obs, enc, device):
#     """编码观测值，支持历史帧序列或单帧"""
#     if obs.dim() == 4:  # [T, C, H, W]
#         features = []
#         for t in range(obs.size(0)):
#             frame = obs[t].permute(1, 2, 0).numpy()  # [H, W, C]
#             frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
#             with torch.no_grad():
#                 z = enc.extract_features(frame_tensor)  # [1, dino_dim]
#             features.append(z.squeeze(0).cpu().numpy())
#         return np.stack(features, axis=0)  # [T, dino_dim]
#     elif obs.dim() == 3:  # [C, H, W]
#         frame = obs.permute(1, 2, 0).numpy()  # [H, W, C]
#         frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
#         with torch.no_grad():
#             z = enc.extract_features(frame_tensor)  # [1, dino_dim]
#         return z.squeeze(0).cpu().numpy()  # [dino_dim]
#     else:
#         raise ValueError("Unexpected observation shape")

# # 预测轨迹
# def predict_trajectory(history_z, action_sequence, p, device):
#     """预测潜在状态轨迹，基于历史特征"""
#     trajectory = [history_z[-1]]  # 初始状态为历史中的最后一个
#     current_history = history_z.copy()  # [T, dino_dim]
#     for t in range(len(action_sequence)):
#         a = action_sequence[t]
#         # 准备输入
#         history_tensor = torch.from_numpy(current_history).float().unsqueeze(0).to(device)  # [1, T, dino_dim]
#         a_tensor = torch.from_numpy(a).float().unsqueeze(0).to(device)  # [1, 1, action_dim]
#         with torch.no_grad():
#             pred_z = p.predict_next_state(history_tensor, a_tensor)  # 假设 p 返回 [1, dino_dim]
#         next_z = pred_z.squeeze(0).cpu().numpy()  # [dino_dim]
#         trajectory.append(next_z)
#         # 更新历史
#         current_history = np.vstack([current_history[1:], next_z])
#     return trajectory



# # CEM 规划
# def cem_plan(history_z, zg, p, T, N, K, num_iters, action_dim, action_range=(-1, 1), device='cuda'):
#     """CEM 规划函数"""
#     # 初始化动作序列的均值和协方差
#     mean = np.zeros((T, action_dim))
#     cov = np.array([np.eye(action_dim) * 1.0 for _ in range(T)])  # [T, action_dim, action_dim]

#     for _ in range(num_iters):
#         # 采样 N 个动作序列
#         action_sequences = np.array([
#             np.random.multivariate_normal(mean[t], cov[t]) for t in range(T) for _ in range(N)
#         ]).reshape(N, T, action_dim)
#         action_sequences = np.clip(action_sequences, action_range[0], action_range[1])

#         # 计算每个序列的成本
#         costs = []
#         for i in range(N):
#             traj = predict_trajectory(history_z, action_sequences[i], p, device)
#             zT = traj[-1]  # 最终状态
#             cost = np.mean((zT - zg) ** 2)  # MSE
#             costs.append(cost)
#         costs = np.array(costs)

#         # 选择前 K 个精英序列
#         elite_indices = np.argsort(costs)[:K]
#         elite_sequences = action_sequences[elite_indices]

#         # 更新均值和协方差
#         mean = np.mean(elite_sequences, axis=0)  # [T, action_dim]
#         cov = np.array([np.cov(elite_sequences[:, t, :], rowvar=False) for t in range(T)])

#     return mean  # 返回优化后的动作序列

# # 主循环
# def main_loop(env, enc, p, config):
#     max_steps = 200
#     trajectory_log = []  # 记录实际轨迹
#     for step in range(max_steps):
#         o0 = env.get_obs()
#         og = env.get_goal_obs()
#         history_z = encode_observation(o0, enc, config.device)
#         zg = encode_observation(og, enc, config.device)
        
#         # 使用 env.predict_steps=40 作为规划步数，而不是 config.history_steps
#         action_sequence = cem_plan(history_z, zg, p, T=env.predict_steps, N=100, K=10, num_iters=5,
#                                    action_dim=config.action_dim, device=config.device)
        
#         # 预测轨迹
#         planned_traj = predict_trajectory(history_z, action_sequence, p, config.device)
#         trajectory_log.append(history_z[-1])  # 记录当前状态

#         # 执行动作
#         k = 10
#         for i in range(min(k, len(action_sequence))):
#             obs, _, done, _ = env.step(action_sequence[i])
#             if done:
#                 break

#         # 可视化（每 10 步显示一次）
#         if step % 10 == 0:
#             plt.plot([z[0] for z in trajectory_log], label="实际轨迹")
#             plt.plot([z[0] for z in planned_traj], label="规划轨迹")
#             plt.scatter([zg[0]], [zg[1]], color='red', label="目标")
#             plt.legend()
#             plt.title(f"Step {step}")
#             plt.show()
#     return trajectory_log
            
            
# def load_model(checkpoint_path, config):
#     """
#     加载训练好的模型
    
#     Args:
#         checkpoint_path: 模型检查点路径
#         config: 配置对象
        
#     Returns:
#         加载好的模型
#     """
    
#     model = WM_Policy(
#         action_dim=config.action_dim,
#         history_steps=config.history_steps,
#         hidden_dim=config.hidden_dim,
#         device=device,
#         mode='feature'
#     ).to(device)
    
#     # 加载检查点
#     checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
#     # 加载模型权重
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     # 设置为评估模式
#     model.eval()
    
#     print(f"成功加载模型检查点: {checkpoint_path}")
#     print(f"模型训练到第 {checkpoint['epoch'] + 1} 个epoch，损失值: {checkpoint['loss']:.6f}")
    
#     return model

# if __name__ == "__main__":
#     # 设置配置
#     config = PolicyConfig()
#     device = torch.device(config.device)

#     # 初始化特征提取器和世界模型
#     feature_extractor = DINOFeatureExtractor(device=device)


#     # 加载预训练权重（假设已保存）
#     policy_weights_path = "LM_wm/logs/policy/checkpoints/best_model.pth"
#     wm_policy  = load_model(policy_weights_path, config)

#     # 初始化环境
#     env = Env(data_dir="LM_wm/training_data", scene_name="vehicle_tracks_000", 
#               history_steps=config.history_steps, predict_steps=20)## 

#     # 运行主循环
#     main_loop(env, feature_extractor, wm_policy, config)