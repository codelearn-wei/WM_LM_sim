import pickle
import os
def load_frame_data(file_path: str):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"Data successfully loaded from {file_path}")
    return data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory(history, future, show_heading=False):
    """
    绘制历史轨迹和未来轨迹。

    参数：
    - history: numpy 数组，形状为 (N, 7)，列 0/1/4 分别表示 x, y, heading（角度制或弧度制）
    - future: numpy 数组，形状为 (M, 3)，列 0/1/2 分别表示 x, y, heading
    - show_heading: bool，是否绘制航向箭头
    """
    plt.figure(figsize=(10, 6))

    # 轨迹点
    plt.plot(history[:, 0], history[:, 1], 'bo-', label='History trajectory')
    plt.plot(future[:, 0], future[:, 1], 'ro-', label='Future trajectory')

    if show_heading:
        # 绘制历史航向箭头
        for i in range(0, len(history), max(len(history)//10,1)):
            x, y, heading = history[i, 0], history[i, 1], history[i, 4]
            dx, dy = 0.2 * np.cos(heading), 0.2 * np.sin(heading)
            plt.arrow(x, y, dx, dy, color='blue', head_width=0.05)

        # 绘制未来航向箭头
        for i in range(0, len(future), max(len(future)//10,1)):
            x, y, heading = future[i, 0], future[i, 1], future[i, 2]
            dx, dy = 0.2 * np.cos(heading), 0.2 * np.sin(heading)
            plt.arrow(x, y, dx, dy, color='red', head_width=0.05)

    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Trajectory Visualization')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_multi_agent_trajectories(history, future, agent_ids=None, show_heading=False):
    """
    绘制多 agent 的历史与未来轨迹，智能体颜色固定。
    
    参数：
    - history: np.ndarray, shape (T, N, 7)，历史轨迹（x, y 在 [0,1]，heading 在 [4]）
    - future: np.ndarray, shape (T, N, 3)，未来轨迹（x, y, heading）
    - agent_ids: list[int]，要绘制的 agent 索引，默认为全部
    - show_heading: bool，是否绘制航向箭头
    """
    T, N, _ = history.shape
    if agent_ids is None:
        agent_ids = list(range(N))

    cmap = get_cmap('tab10')  # 最多10个颜色循环使用
    plt.figure(figsize=(12, 8))

    for idx, i in enumerate(agent_ids):
        color = cmap(idx % 10)  # 固定颜色分配
        hist_x = history[:, i, 0]
        hist_y = history[:, i, 1]
        hist_h = history[:, i, 4]

        fut_x = future[:, i, 0]
        fut_y = future[:, i, 1]
        fut_h = future[:, i, 2]

        # 绘制轨迹
        plt.plot(hist_x, hist_y, 'o-', label=f'Agent {i} - History', color=color)
        plt.plot(fut_x, fut_y, 'x--', label=f'Agent {i} - Future', color=color)

        if show_heading:
            for j in range(0, T, max(1, T // 10)):
                dx = 0.2 * np.cos(hist_h[j])
                dy = 0.2 * np.sin(hist_h[j])
                plt.arrow(hist_x[j], hist_y[j], dx, dy, head_width=0.05, color=color)

                dx_f = 0.2 * np.cos(fut_h[j])
                dy_f = 0.2 * np.sin(fut_h[j])
                plt.arrow(fut_x[j], fut_y[j], dx_f, dy_f, head_width=0.05, color=color, alpha=0.7)

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Multi-Agent Trajectories')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    
    output_dir = r"src\datasets\data"
    loaded_data = load_frame_data(os.path.join(output_dir, "train_trajectories_1.pkl"))
    # history_tra = loaded_data[101]['ego_history']
    # future_tra = loaded_data[101]['ego_future']
    # plot_trajectory(history_tra , future_tra)
    
    agent_history_tra = loaded_data[101]['agent_history']
    agent_future_tra = loaded_data[101]['agent_future']
    
    plot_multi_agent_trajectories(agent_history_tra , agent_future_tra)
    
    print(loaded_data[0]['ego_history'])
    print(loaded_data[0]['ego_future'])
    
