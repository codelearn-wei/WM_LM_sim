import numpy as np
from pathlib import Path
import cv2
import sys
import os
from collections import defaultdict
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from data_process.LM_scene import LMScene
from data_process.train_raw_data import (
    organize_by_frame,
    classify_vehicles_by_frame_1,
    filter_vehicles_by_x,
    classify_vehicles,
    filter_all_boundaries
)
from data_process.train_bev_gen import BEVGenerator

class TrainingDataGenerator:
    def __init__(self, map_path: str, base_data_path: str):
        """
        初始化训练数据生成器

        Args:
            map_path (str): 地图文件路径
            base_data_path (str): 原始数据目录路径
        """
        self.map_path = map_path
        self.base_data_path = Path(base_data_path)
        self.vehicle_history = defaultdict(list)  # 用于存储每个车辆的历史数据
        
    def _update_vehicle_history(self, vehicles):
        """
        更新车辆历史数据

        Args:
            vehicles (list): 当前帧的车辆数据列表
        """
        for vehicle in vehicles:
            self.vehicle_history[vehicle['track_id']].append(vehicle)
            # 只保留最近的两个时间戳的数据
            if len(self.vehicle_history[vehicle['track_id']]) > 2:
                self.vehicle_history[vehicle['track_id']].pop(0)

    def _compute_vehicle_actions(self, track_id):
        """
        计算车辆动作（加速度和航向角变化）

        Args:
            track_id: 车辆ID

        Returns:
            dict: 包含加速度和航向角变化的字典，如果无法计算则返回None
        """
        vehicle_data = self.vehicle_history[track_id]
        if len(vehicle_data) < 2:
            return None

        prev = vehicle_data[-2]  # 前一帧
        curr = vehicle_data[-1]  # 当前帧
        dt = (curr['timestamp'] - prev['timestamp'])/1000

        if dt > 0:
            # 计算加速度
            ax = (curr['vx'] - prev['vx']) / dt
            ay = (curr['vy'] - prev['vy']) / dt

            # 计算航向角变化
            delta_psi = curr['psi_rad'] - prev['psi_rad']
            # 标准化到 [-pi, pi]
            delta_psi = (delta_psi + np.pi) % (2 * np.pi) - np.pi

            return {
                'ax': ax,
                'ay': ay,
                'delta_psi': delta_psi,
                'is_lane_changing': curr['lane_type'] == '变道车辆'
            }
        return None

    def _process_scene_data(self, scene: LMScene):
        """
        处理单个场景的数据

        Args:
            scene (LMScene): 场景对象

        Returns:
            tuple: (处理后的车辆数据, 边界数据)
        """
        # 获取并过滤车辆数据
        frame_data = organize_by_frame(scene.vehicles)
        filtered_data = filter_vehicles_by_x(frame_data, x_threshold=1055)
        
        # 获取道路边界
        upper_bd = scene.get_upper_boundary()
        auxiliary_bd = scene.get_auxiliary_dotted_line()
        lower_bd = scene.get_main_lower_boundary()
        
        # 分类车辆并过滤边界
        classified_data = classify_vehicles(filtered_data, upper_bd, auxiliary_bd)
        filtered_boundaries = filter_all_boundaries(
            upper_bd, auxiliary_bd, lower_bd, x_threshold=1055
        )
        
        return classified_data, filtered_boundaries

    def _process_frame_actions(self, vehicles, max_vehicles=10):
        """
        处理一帧中的车辆动作数据，只保留非主道车辆的动作

        Args:
            vehicles (list): 当前帧的车辆列表
            max_vehicles (int): 最大车辆数

        Returns:
            numpy.ndarray: 动作数组，形状为 (max_vehicles * 3,)，包含ax, ay, delta_psi
        """
        # 更新车辆历史数据
        self._update_vehicle_history(vehicles)
        
        # 初始化动作数组：[ax1, ay1, dpsi1, ax2, ay2, dpsi2, ...]
        actions = np.zeros(max_vehicles * 3)
        
        # 获取非主道车辆的动作
        lane_changing_vehicles = [v for v in vehicles if v['lane_type'] == '变道车辆']
        
        # 填充动作数据
        for i, vehicle in enumerate(lane_changing_vehicles[:max_vehicles]):
            action_data = self._compute_vehicle_actions(vehicle['track_id'])
            if action_data is not None:
                idx = i * 3
                actions[idx:idx+3] = [
                    action_data['ax'],
                    action_data['ay'],
                    action_data['delta_psi']
                ]
        
        return actions

    def generate_training_data(self, output_dir: str, image_size: tuple = (300, 800)):
        """
        生成训练数据，包括BEV图像和对应的动作数据

        Args:
            output_dir (str): 输出目录
            image_size (tuple, optional): BEV图像尺寸. 默认为 (300, 800).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 遍历所有场景数据
        for csv_file in self.base_data_path.glob("*.csv"):
            scene_id = csv_file.stem
            scene_output_dir = output_path / scene_id
            scene_output_dir.mkdir(exist_ok=True)
            
            # 重置车辆历史数据
            self.vehicle_history.clear()
            
            # 加载场景
            scene = LMScene(self.map_path, str(csv_file))
            classified_data, (filtered_upper_bd, filtered_auxiliary_bd, filtered_lower_bd) = self._process_scene_data(scene)
            
            # 初始化BEV生成器
            bev_gen = BEVGenerator(image_size=image_size, resolution=0.1, range_m=25)
            bev_gen.set_scene_center(filtered_upper_bd, filtered_lower_bd)
            
            # 生成每一帧的数据
            frame_actions = {}
            for frame_id, vehicles in classified_data.items():
                # 生成BEV图像
                bev_gen.generate_bev(
                    vehicles,
                    filtered_upper_bd,
                    filtered_auxiliary_bd,
                    filtered_lower_bd,
                    frame_id,
                    output_dir=str(scene_output_dir)
                )
                
                # 计算并保存动作数据
                frame_actions[frame_id] = self._process_frame_actions(vehicles)
            
            # 保存动作数据
            action_file = scene_output_dir / 'actions.npz'
            np.savez(action_file, frame_actions=frame_actions)
            
            print(f"已处理场景: {scene_id}") 
        