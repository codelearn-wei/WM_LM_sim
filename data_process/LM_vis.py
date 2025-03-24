
# !提供可视化场景类，回放测试时，参考LM_scene.py中的get_scene()方法。
# !具体而言，修改添加和删减get_scene()中的车辆，替换背景车为主车即可。
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, Slider
import cv2
import os
from LM_scene import LMScene

class SceneVisualizer:
    def __init__(self, scenes, map_data):
        self.scenes = scenes
        self.map_data = map_data
        self.current_scene_idx = 0
        self.current_frame_idx = 0  # 当前帧的索引
        self.current_frame = 0  # 当前帧的值
        self.is_playing = True  # 是否正在播放
        self.playback_speed = 100  # 默认播放速度（毫秒每帧）

         # 创建图形和坐标轴
        self.fig, self.ax = plt.subplots(figsize=(18, 10))  # 增大图形大小
        self.ax.set_xlabel("X Coordinate", fontsize=12)  # 增大字体
        self.ax.set_ylabel("Y Coordinate", fontsize=12)  # 增大字体

        # 绘制地图边界
        self._plot_map_boundaries()

        # 车辆形状和轨迹
        self.main_vehicle_shape = None
        self.bg_vehicle_shapes = []
        self.main_vehicle_trajectory, = self.ax.plot([], [], 'g--', linewidth=1)
        self.bg_vehicle_trajectories = []

        # 车辆ID标签
        self.main_vehicle_id_label = None
        self.bg_vehicle_id_labels = []

        # 添加按钮和进度条
        self._add_buttons()
        self._add_slider()

    def _plot_map_boundaries(self):
        self.ax.set_aspect('equal', adjustable='box')

        # 绘制地图边界
        self.ax.plot(self.map_data['upper_boundary'][:, 0], self.map_data['upper_boundary'][:, 1], 'k-', linewidth=2)
        self.ax.plot(self.map_data['main_lower_boundary'][:, 0], self.map_data['main_lower_boundary'][:, 1], 'k-', linewidth=2)
        self.ax.plot(self.map_data['auxiliary_dotted_line'][:, 0], self.map_data['auxiliary_dotted_line'][:, 1], 'r--', linewidth=2)

        # 设置坐标轴范围
        x_min = min(np.min(self.map_data['upper_boundary'][:, 0]), 
                    np.min(self.map_data['main_lower_boundary'][:, 0]), 
                    np.min(self.map_data['auxiliary_dotted_line'][:, 0]))
        x_max = max(np.max(self.map_data['upper_boundary'][:, 0]), 
                    np.max(self.map_data['main_lower_boundary'][:, 0]), 
                    np.max(self.map_data['auxiliary_dotted_line'][:, 0]))
        y_min = min(np.min(self.map_data['upper_boundary'][:, 1]), 
                    np.min(self.map_data['main_lower_boundary'][:, 1]), 
                    np.min(self.map_data['auxiliary_dotted_line'][:, 1]))
        y_max = max(np.max(self.map_data['upper_boundary'][:, 1]), 
                    np.max(self.map_data['main_lower_boundary'][:, 1]), 
                    np.max(self.map_data['auxiliary_dotted_line'][:, 1]))

        self.ax.set_xlim(x_min - 1, x_max + 1)
        self.ax.set_ylim(y_min - 1, y_max + 1)

    def _get_vehicle_shape(self, x, y, yaw, length, width):
        half_length = length / 2
        half_width = width / 2
        corners = np.array([[-half_length, -half_width],
                            [half_length, -half_width],
                            [half_length, half_width],
                            [-half_length, half_width]])
        
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        
        rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([x, y])
        return rotated_corners

    def _draw_vehicle(self, x, y, yaw, length, width, color='g'):
        corners = self._get_vehicle_shape(x, y, yaw, length, width)
        vehicle_shape = Polygon(corners, closed=True, fill=None, edgecolor=color, linewidth=2)
        self.ax.add_patch(vehicle_shape)
        return vehicle_shape

    def _get_future_trajectory(self, scene, current_frame, vehicle_type, vehicle_id=None):
        frames = sorted(scene['frames'].keys())
        current_frame_idx = frames.index(current_frame)
        future_frames = frames[current_frame_idx:current_frame_idx + 20] 

        x_trajectory = []
        y_trajectory = []

        for frame in future_frames:
            if vehicle_type == 'main_vehicle':
                vehicle_info = scene['frames'][frame]['main_vehicle']
            else:
                vehicle_info = next((v for v in scene['frames'][frame]['background_vehicles'] if v['id'] == vehicle_id), None)
                if not vehicle_info:
                    break
            x_trajectory.append(vehicle_info['x'])
            y_trajectory.append(vehicle_info['y'])

        return x_trajectory, y_trajectory

    def _update_plot(self, frame):
        scene = self.scenes[self.current_scene_idx]
        frames = sorted(scene['frames'].keys())
        self.current_frame = frames[frame]
        self.current_frame_idx = frame  # 更新当前帧索引

        # 主车信息
        main_vehicle_info = scene['frames'][self.current_frame]['main_vehicle']
        main_x = main_vehicle_info['x']
        main_y = main_vehicle_info['y']
        main_yaw = main_vehicle_info['psi_rad']
        vehicle_length = main_vehicle_info['length']
        vehicle_width = main_vehicle_info['width']

        # 绘制主车形状
        if self.main_vehicle_shape is not None:
            self.main_vehicle_shape.remove()
        self.main_vehicle_shape = self._draw_vehicle(main_x, main_y, main_yaw, vehicle_length, vehicle_width, color='g')

        # 更新主车轨迹
        main_trajectory_x, main_trajectory_y = self._get_future_trajectory(scene, self.current_frame, 'main_vehicle')
        self.main_vehicle_trajectory.set_data(main_trajectory_x, main_trajectory_y)

        # 更新主车ID标签
        if self.main_vehicle_id_label is not None:
            self.main_vehicle_id_label.remove()
        self.main_vehicle_id_label = self.ax.text(main_x, main_y, f"ID: {main_vehicle_info['id']}", color='b', fontsize=8, ha='center', va='center')

        # 背景车辆
        bg_vehicles_info = scene['frames'][self.current_frame]['background_vehicles']
        bg_x = [v['x'] for v in bg_vehicles_info]
        bg_y = [v['y'] for v in bg_vehicles_info]

        # 移除旧的背景车辆
        for shape in self.bg_vehicle_shapes:
            shape.remove()
        self.bg_vehicle_shapes = []
        for trajectory in self.bg_vehicle_trajectories:
            trajectory.remove()
        self.bg_vehicle_trajectories = []
        for label in self.bg_vehicle_id_labels:
            label.remove()
        self.bg_vehicle_id_labels = []

        # 绘制背景车辆
        for vehicle_info in bg_vehicles_info:
            bg_vehicle_x = vehicle_info['x']
            bg_vehicle_y = vehicle_info['y']
            bg_vehicle_yaw = vehicle_info['psi_rad']
            bg_vehicle_length = vehicle_info['length']
            bg_vehicle_width = vehicle_info['width']
            vehicle_id = vehicle_info['id']

            shape = self._draw_vehicle(bg_vehicle_x, bg_vehicle_y, bg_vehicle_yaw, bg_vehicle_length, bg_vehicle_width, color='b')
            self.bg_vehicle_shapes.append(shape)
            # 获得背景车辆轨迹
            bg_trajectory_x, bg_trajectory_y = self._get_future_trajectory(scene, self.current_frame, 'background_vehicles', vehicle_id)
            # 绘制背景车辆轨迹
            trajectory_line, = self.ax.plot(bg_trajectory_x, bg_trajectory_y, 'b--', linewidth=1)
            self.bg_vehicle_trajectories.append(trajectory_line)

            label = self.ax.text(bg_vehicle_x, bg_vehicle_y, f"ID: {vehicle_id}", color='g', fontsize=8, ha='center', va='center')
            self.bg_vehicle_id_labels.append(label)

        self.ax.set_title(f"Scene {scene['scene_id']} - Frame {self.current_frame} (Time: {self.current_frame * 0.1:.1f}s)")
        return [self.main_vehicle_shape, self.main_vehicle_trajectory] + self.bg_vehicle_shapes + self.bg_vehicle_trajectories + self.bg_vehicle_id_labels + [self.main_vehicle_id_label]

    def _add_buttons(self):
        # 添加播放/暂停按钮
        ax_play_pause = plt.axes([0.3, 0.2, 0.15, 0.06])  # 调整按钮大小和位置
        self.btn_play_pause = Button(
            ax_play_pause, 
            'Pause', 
            color='lightblue',  # 按钮颜色
            hovercolor='lightgreen'  # 鼠标悬停时的颜色
        )
        self.btn_play_pause.label.set_fontsize(12)  # 设置字体大小
        self.btn_play_pause.on_clicked(self._toggle_play)

        # 添加逐帧播放按钮
        ax_next_frame = plt.axes([0.5, 0.2, 0.15, 0.06])  # 调整按钮大小和位置
        self.btn_next_frame = Button(
            ax_next_frame, 
            'Next Frame', 
            color='lightblue',  # 按钮颜色
            hovercolor='lightgreen'  # 鼠标悬停时的颜色
        )
        self.btn_next_frame.label.set_fontsize(12)  # 设置字体大小
        self.btn_next_frame.on_clicked(self._next_frame)

    def _add_slider(self):
        # 添加进度条
        ax_slider = plt.axes([0.2, 0.3, 0.65, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label='Frame',
            valmin=0,
            valmax=len(self.scenes[self.current_scene_idx]['frames']) - 1,
            valinit=0,
            valstep=1,
            color='lightblue',  # 滑块颜色
            track_color='lightgray',  # 滑轨颜色
            handle_style={'facecolor': 'blue', 'edgecolor': 'white', 'size': 10}  # 滑块手柄样式
        )
        self.slider.label.set_fontsize(12)  # 设置字体大小
        self.slider.on_changed(self._update_frame_from_slider)

    def _update_frame_from_slider(self, val):
        """
        当进度条的值改变时，更新当前帧
        """
        if not self.is_playing:  # 只有在暂停状态下才能拖动进度条
            self.current_frame_idx = int(val)
            self._update_plot(self.current_frame_idx)
            self.fig.canvas.draw()

    def _toggle_play(self, event):
        """
        切换播放/暂停状态
        """
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play_pause.label.set_text("Pause")
            self.animation.event_source.start()  # 继续播放
        else:
            self.btn_play_pause.label.set_text("Play")
            self.animation.event_source.stop()  # 暂停播放

    def _next_frame(self, event):
        """
        逐帧播放，手动推进到下一帧
        """
        if not self.is_playing:  # 只有在暂停状态下才能逐帧播放
            # 获取当前场景的帧总数
            num_frames = len(self.scenes[self.current_scene_idx]['frames'])
            
            # 如果当前帧已经是最后一帧，则回到第一帧
            if self.current_frame_idx >= num_frames - 1:
                self.current_frame_idx = 0
            else:
                # 否则，推进到下一帧
                self.current_frame_idx += 1

            # 更新进度条的值
            self.slider.set_val(self.current_frame_idx)

            # 更新绘图
            self._update_plot(self.current_frame_idx)
            self.fig.canvas.draw()

    def show_scene(self, scene_idx):
        self.current_scene_idx = scene_idx
        scene = self.scenes[scene_idx]
        frames = sorted(scene['frames'].keys())
        num_frames = len(frames)

        # 更新进度条的最大值
        self.slider.valmax = num_frames - 1
        self.slider.ax.set_xlim(0, num_frames - 1)

        # 创建动画
        self.animation = animation.FuncAnimation(
            self.fig,
            self._update_plot,
            frames=num_frames,
            interval=self.playback_speed,  # 使用当前播放速度
            repeat=False
        )
        plt.show()
            
    def save_scene(self, output_dir="output_videos", fps=10):
        """
        保存所有场景的动画为视频文件
        :param output_dir: 输出目录
        :param fps: 视频的帧率
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for scene_idx in range(len(self.scenes)):
            self.current_scene_idx = scene_idx  # 更新当前场景索引
            scene = self.scenes[scene_idx]
            frames = sorted(scene['frames'].keys())
            num_frames = len(frames)

            # 打印调试信息
            print(f"Processing scene {scene['scene_id']} with {num_frames} frames")

            # 创建视频文件
            output_path = os.path.join(output_dir, f"scene_{scene['scene_id']}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择编码格式
            width, height = self.fig.canvas.get_width_height()
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 逐帧生成视频
            for frame_idx in range(num_frames):
                if frame_idx >= len(frames):  # 检查索引是否超出范围
                    print(f"Warning: Frame index {frame_idx} is out of range for scene {scene['scene_id']}")
                    break

                self.current_frame_idx = frame_idx  # 更新当前帧索引
                self._update_plot(frame_idx)  # 更新绘图
                self.fig.canvas.draw()  # 强制重绘图

                # 从 matplotlib figure 中获取当前帧
                buf = self.fig.canvas.buffer_rgba()  # 获取 RGBA 图像数据
                img = np.frombuffer(buf, dtype=np.uint8).reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # 转换为 BGR 格式

                # 将图像写入视频文件
                video_writer.write(img)

            # 释放资源
            video_writer.release()
            print(f"Saved scene {scene['scene_id']} to {output_path}")
            
# 示例用法
if __name__ == "__main__":
    LM_scene = LMScene("LM_data\map\DR_CHN_Merging_ZS.json", "LM_data/data/DR_CHN_Merging_ZS/vehicle_tracks_000.csv")
    map_data = LM_scene.map_dict
    scenes_list = LM_scene.get_scene()
    
    visualizer = SceneVisualizer(scenes_list, map_data)
    # 展示某一个场景
    visualizer.show_scene(2)
    # 保存所有场景的动画为视频文件
    # visualizer.save_scene(output_dir="output_videos", fps=10)
    
    print("Scene count:", len(scenes_list))
