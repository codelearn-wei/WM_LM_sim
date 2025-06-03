import matplotlib.pyplot as plt
import numpy as np
import pickle
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
def visualize_trajectory_with_mask(trajectory_data, trajectory_idx=0, save_path=None):
    """
    可视化带掩码和不带掩码的轨迹数据
    
    Args:
        trajectory_data: 轨迹数据列表
        trajectory_idx: 要可视化的轨迹索引
        save_path: 保存图片的路径（可选）
    """
    if trajectory_idx >= len(trajectory_data):
        print(f"轨迹索引 {trajectory_idx} 超出范围，总共有 {len(trajectory_data)} 条轨迹")
        return
    
    traj = trajectory_data[trajectory_idx]
    
    # 提取数据
    ego_history = traj['ego_history']  # [history_frames, 7]
    agent_history = traj['agent_history']  # [history_frames, num_agents, 7]
    ego_future = traj['ego_future']  # [future_frames, 3]
    agent_future = traj['agent_future']  # [future_frames, num_agents, 3]
    
    agent_history_mask = traj['agent_history_mask']  # [history_frames, num_agents]
    agent_future_mask = traj['agent_future_mask']  # [future_frames, num_agents]
    
    agent_ids = traj['agent_ids']
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'轨迹 {trajectory_idx} 掩码验证可视化\nEgo ID: {traj["ego_id"]}, Agent IDs: {agent_ids}', 
                 fontsize=14, fontweight='bold')
    
    # 定义颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_ids)))
    ego_color = 'red'
    
    # 1. 历史轨迹 - 不带掩码
    ax1 = axes[0, 0]
    plot_trajectories(ax1, ego_history, agent_history, None, colors, ego_color, 
                     "历史轨迹 (不带掩码)", agent_ids, is_history=True)
    
    # 2. 历史轨迹 - 带掩码
    ax2 = axes[0, 1]
    plot_trajectories(ax2, ego_history, agent_history, agent_history_mask, colors, ego_color, 
                     "历史轨迹 (带掩码)", agent_ids, is_history=True)
    
    # 3. 未来轨迹 - 不带掩码
    ax3 = axes[1, 0]
    plot_trajectories(ax3, ego_future, agent_future, None, colors, ego_color, 
                     "未来轨迹 (不带掩码)", agent_ids, is_history=False)
    
    # 4. 未来轨迹 - 带掩码
    ax4 = axes[1, 1]
    plot_trajectories(ax4, ego_future, agent_future, agent_future_mask, colors, ego_color, 
                     "未来轨迹 (带掩码)", agent_ids, is_history=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    # plt.show()
    
    # 打印掩码统计信息
    print_mask_statistics(traj, trajectory_idx)

def plot_trajectories(ax, ego_traj, agent_traj, mask, colors, ego_color, title, agent_ids, is_history=True):
    """
    绘制轨迹
    """
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 绘制ego轨迹
    ego_x = ego_traj[:, 0]
    ego_y = ego_traj[:, 1]
    ax.plot(ego_x, ego_y, 'o-', color=ego_color, linewidth=3, markersize=8, 
            label='Ego Vehicle', alpha=0.8)
    
    # 标记起始点
    ax.scatter(ego_x[0], ego_y[0], color=ego_color, s=100, marker='s', 
               edgecolors='black', linewidth=2, label='Start', zorder=10)
    
    # 绘制agent轨迹
    num_agents = len(agent_ids)
    
    for i in range(num_agents):
        agent_id = agent_ids[i]
        if agent_id == -1:  # 跳过填充的无效agent
            continue
            
        color = colors[i]
        
        # 提取位置数据
        agent_x = agent_traj[:, i, 0]
        agent_y = agent_traj[:, i, 1]
        
        if mask is not None:
            # 应用掩码
            valid_mask = mask[:, i]
            
            # 绘制有效点
            valid_x = agent_x[valid_mask]
            valid_y = agent_y[valid_mask]
            
            if len(valid_x) > 0:
                ax.scatter(valid_x, valid_y, color=color, s=30, alpha=0.8, 
                          label=f'Agent {agent_id} (valid)')
                
                # 连接有效点
                if len(valid_x) > 1:
                    ax.plot(valid_x, valid_y, color=color, alpha=0.6, linewidth=2)
            
            # 绘制无效点（用x标记）
            invalid_mask = ~valid_mask
            invalid_x = agent_x[invalid_mask]
            invalid_y = agent_y[invalid_mask]
            
            if len(invalid_x) > 0:
                ax.scatter(invalid_x, invalid_y, color=color, marker='x', s=50, 
                          alpha=0.5, label=f'Agent {agent_id} (masked)')
        else:
            # 不应用掩码，绘制所有点
            ax.plot(agent_x, agent_y, 'o-', color=color, alpha=0.7, 
                   markersize=4, label=f'Agent {agent_id}')
    
    # 设置图例
    if len(ax.get_legend_handles_labels()[0]) > 0:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

def print_mask_statistics(traj, trajectory_idx):
    """
    打印掩码统计信息
    """
    print(f"\n=== 轨迹 {trajectory_idx} 掩码统计信息 ===")
    
    agent_history_mask = traj['agent_history_mask']
    agent_future_mask = traj['agent_future_mask']
    agent_ids = traj['agent_ids']
    
    print(f"历史帧数: {agent_history_mask.shape[0]}")
    print(f"未来帧数: {agent_future_mask.shape[0]}")
    print(f"Agent数量: {len(agent_ids)}")
    
    print("\n各Agent在历史序列中的有效率:")
    for i, agent_id in enumerate(agent_ids):
        if agent_id != -1:
            valid_ratio = agent_history_mask[:, i].sum() / agent_history_mask.shape[0]
            print(f"  Agent {agent_id}: {valid_ratio:.2%} ({agent_history_mask[:, i].sum()}/{agent_history_mask.shape[0]})")
    
    print("\n各Agent在未来序列中的有效率:")
    for i, agent_id in enumerate(agent_ids):
        if agent_id != -1:
            valid_ratio = agent_future_mask[:, i].sum() / agent_future_mask.shape[0]
            print(f"  Agent {agent_id}: {valid_ratio:.2%} ({agent_future_mask[:, i].sum()}/{agent_future_mask.shape[0]})")
    
    # 总体统计
    total_history_valid = agent_history_mask.sum()
    total_history_positions = agent_history_mask.size
    total_future_valid = agent_future_mask.sum()
    total_future_positions = agent_future_mask.size
    
    print(f"\n总体统计:")
    print(f"  历史序列有效率: {total_history_valid/total_history_positions:.2%}")
    print(f"  未来序列有效率: {total_future_valid/total_future_positions:.2%}")

def compare_multiple_trajectories(trajectory_data, num_samples=3, save_dir=None):
    """
    比较多个轨迹的掩码效果
    """
    if len(trajectory_data) < num_samples:
        num_samples = len(trajectory_data)
    
    # 随机选择几个轨迹进行比较
    indices = np.random.choice(len(trajectory_data), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        print(f"\n{'='*50}")
        print(f"可视化轨迹 {idx}")
        print(f"{'='*50}")
        
        save_path = None
        if save_dir:
            save_path = f"{save_dir}/trajectory_{idx}_mask_comparison.png"
        
        visualize_trajectory_with_mask(trajectory_data, idx, save_path)

def analyze_mask_effectiveness(trajectory_data):
    """
    分析整个数据集的掩码有效性
    """
    print("\n" + "="*60)
    print("数据集掩码有效性分析")
    print("="*60)
    
    total_trajectories = len(trajectory_data)
    history_valid_ratios = []
    future_valid_ratios = []
    
    for traj in trajectory_data:
        agent_history_mask = traj['agent_history_mask']
        agent_future_mask = traj['agent_future_mask']
        
        history_valid_ratio = agent_history_mask.sum() / agent_history_mask.size
        future_valid_ratio = agent_future_mask.sum() / agent_future_mask.size
        
        history_valid_ratios.append(history_valid_ratio)
        future_valid_ratios.append(future_valid_ratio)
    
    history_valid_ratios = np.array(history_valid_ratios)
    future_valid_ratios = np.array(future_valid_ratios)
    
    print(f"总轨迹数: {total_trajectories}")
    print(f"\n历史序列掩码统计:")
    print(f"  平均有效率: {history_valid_ratios.mean():.2%}")
    print(f"  最小有效率: {history_valid_ratios.min():.2%}")
    print(f"  最大有效率: {history_valid_ratios.max():.2%}")
    print(f"  标准差: {history_valid_ratios.std():.2%}")
    
    print(f"\n未来序列掩码统计:")
    print(f"  平均有效率: {future_valid_ratios.mean():.2%}")
    print(f"  最小有效率: {future_valid_ratios.min():.2%}")
    print(f"  最大有效率: {future_valid_ratios.max():.2%}")
    print(f"  标准差: {future_valid_ratios.std():.2%}")

# 使用示例
if __name__ == "__main__":
    # 加载数据
    data_path = r"src\datasets\data\train_trajectories_with_mask.pkl"
    
    try:
        with open(data_path, 'rb') as f:
            trajectory_data = pickle.load(f)
        
        print(f"成功加载 {len(trajectory_data)} 条轨迹数据")
        
        # 分析整个数据集的掩码有效性
        analyze_mask_effectiveness(trajectory_data)
        
        # 可视化单个轨迹
        print("\n开始可视化...")
        # visualize_trajectory_with_mask(trajectory_data, trajectory_idx=1000)
        
        # 比较多个轨迹
        compare_multiple_trajectories(trajectory_data, num_samples=100, save_dir=r"test\test_mask")
        
    except FileNotFoundError:
        print(f"文件未找到: {data_path}")
        print("请确认文件路径是否正确")
    except Exception as e:
        print(f"加载数据时出错: {e}")