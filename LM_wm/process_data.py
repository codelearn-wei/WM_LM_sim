import os
from typing import List, Dict, Any
from pathlib import Path
import time
from tqdm import tqdm
import os
import torch
import torch.utils.data as torch_data
from data_process.LM_vis import SceneVisualizer
from data_process.LM_scene import LMScene
from data_process.train_bev_gen import BEVGenerator
from data_process.train_dataloader import TrafficDataset 
from data_process.relative_dataloder import RelativeTrafficDataset 
from data_process.x_y_dataloader import XYTrafficDataset 

from data_process.train_raw_data import (
    organize_by_frame, 
    filter_vehicles_by_x, 
    classify_vehicles, 
    filter_all_boundaries
)

class LaneMergingDataProcessor:
    def __init__(self, map_path: str, base_data_path: str):
        """
        初始化数据处理器

        Args:
            map_path (str): 地图文件路径
            base_data_path (str): 基础数据目录路径
        """
        self.map_path = map_path
        self.base_data_path = base_data_path

    def save_scenes_as_videos(self, output_base_dir: str, fps: int = 10):
        """
        将所有场景保存为视频文件

        Args:
            output_base_dir (str): 输出基础目录
            fps (int, optional): 视频帧率. 默认为10.
        """
        Path(output_base_dir).mkdir(parents=True, exist_ok=True)

        for csv_file in Path(self.base_data_path).glob("*.csv"):
            file_name = csv_file.stem
            output_dir = Path(output_base_dir) / file_name
            output_dir.mkdir(parents=True, exist_ok=True)

            scene = LMScene(self.map_path, str(csv_file))
            visualizer = SceneVisualizer(scene.get_scene(), scene.map_dict)
            visualizer.save_scene(output_dir=str(output_dir), fps=fps)

            print(f"处理 {csv_file.name}, 场景数量: {len(scene.get_scene())}")

    def generate_real_training_data(
        self, 
        output_base_dir: str, 
        history_steps: int = 20,  # 历史时间步长
        future_steps: int = 1,    # 预测未来时间步长 
        max_vehicles: int = 10,   # 最大车辆数
        max_aux_vehicles: int = 5,  # 最大辅助车辆数
        x_threshold: float = 1055.0,  # X轴过滤阈值
    ) -> List[torch_data.Dataset]:
        """
        生成真实训练数据集，带进度可视化

        Args:
            output_base_dir (str): 输出基础目录
            history_steps (int, optional): 历史时间步长. 默认为 20.
            future_steps (int, optional): 预测未来时间步长. 默认为 1.
            max_vehicles (int, optional): 最大主车辆数. 默认为 10.
            max_aux_vehicles (int, optional): 最大辅助车辆数. 默认为 5.
            x_threshold (float, optional): X轴过滤阈值. 默认为 1055.0.

        Returns:
            List[torch_data.Dataset]: 训练数据集列表
        """
        Path(output_base_dir).mkdir(parents=True, exist_ok=True)
        all_datasets = []

        # 获取所有CSV文件
        csv_files = list(Path(self.base_data_path).glob("*.csv"))
        print(f"找到 {len(csv_files)} 个CSV文件待处理")
        
        # 使用tqdm创建总进度条
        for file_idx, csv_file in enumerate(tqdm(csv_files, desc="处理CSV文件")):
            start_time = time.time()
            file_name = csv_file.stem
            output_dir = Path(output_base_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[{file_idx+1}/{len(csv_files)}] 处理文件: {file_name}")
            
            # 加载场景数据
            print("  ├─ 加载场景数据...")
            scene = LMScene(self.map_path, str(csv_file))
            
            # 组织帧数据
            print("  ├─ 组织帧数据...")
            frame_data = organize_by_frame(scene.vehicles)
            total_frames = len(frame_data)
            print(f"  │  └─ 总帧数: {total_frames}")
            
            # 过滤车辆
            print("  ├─ 按X轴过滤车辆...")
            filtered_data = filter_vehicles_by_x(frame_data, x_threshold=x_threshold)
            
            # 获取边界
            print("  ├─ 提取道路边界...")
            upper_bd = scene.get_upper_boundary()
            auxiliary_bd = scene.get_auxiliary_dotted_line()
            
            # 分类车辆
            print("  ├─ 分类车辆...")
            raw_data = classify_vehicles(
                filtered_data, 
                upper_bd, 
                auxiliary_bd, 
            )
            
            # 创建数据集
            print("  ├─ 创建数据集...")
            try:
                # 使用带有进度显示的包装器
                with tqdm(total=100, desc="  │  └─ 数据集构建", leave=False) as pbar:
                    # 设置监控点
                    def progress_callback(stage, progress):
                        pbar.set_description(f"  │  └─ {stage}")
                        pbar.update(int(progress * 100) - pbar.n)
                    
                    # 创建数据集实例
                    dataset = XYTrafficDataset(
                        raw_data,
                        history_steps=history_steps, 
                        future_steps=future_steps, 
                        max_nearby_vehicles=max_vehicles, 
                        max_aux_vehicles=max_aux_vehicles,
                        progress_callback=progress_callback  # 添加进度回调
                    )
                
                if len(dataset) > 0:
                    dataset_path = output_dir / f"{file_name}_dataset.pt"
                    
                    # 保存数据集
                    print(f"  ├─ 保存数据集 ({len(dataset)} 个样本)...")
                    torch.save(dataset, dataset_path)
                    
                    all_datasets.append(dataset)
                else:
                    print(f"  └─ 文件 {file_name} 未生成训练数据 (样本数为0)")
            
            except Exception as e:
                print(f"  └─ 处理文件 {file_name} 时出错: {str(e)}")
                
        # 打印整体统计
        total_samples = sum(len(dataset) for dataset in all_datasets)
        print(f"\n处理完成! 共生成 {len(all_datasets)} 个数据集，总计 {total_samples} 个样本")
        
        return all_datasets
    
    def generate_bev_images(self, output_base_dir: str, image_size: tuple = (300, 800), resolution: float = 0.1):
        """
        生成BEV（鸟瞰图）图像

        Args:
            output_base_dir (str): 输出基础目录
            image_size (tuple, optional): 图像尺寸. 默认为 (300, 800).
            resolution (float, optional): 分辨率. 默认为 0.1.
        """
        Path(output_base_dir).mkdir(parents=True, exist_ok=True)

        for csv_file in Path(self.base_data_path).glob("*.csv"):
            file_name = csv_file.stem
            output_dir = Path(output_base_dir) / file_name
            output_dir.mkdir(parents=True, exist_ok=True)

            scene = LMScene(self.map_path, str(csv_file))
            frame_data = organize_by_frame(scene.vehicles)
            filtered_data = filter_vehicles_by_x(frame_data, x_threshold=1055)

            upper_bd = scene.get_upper_boundary()
            auxiliary_bd = scene.get_auxiliary_dotted_line()
            lower_bd = scene.get_main_lower_boundary()
            
            classified_data = classify_vehicles(filtered_data, upper_bd, auxiliary_bd)
            
            bev_gen = BEVGenerator(image_size=image_size, resolution=resolution, range_m=25)
            
            filtered_upper_bd, filtered_auxiliary_bd, filtered_lower_bd = filter_all_boundaries(
                upper_bd, auxiliary_bd, lower_bd, x_threshold=1055
            )
            
            for frame_id, vehicles in classified_data.items():
                bev_gen.set_scene_center(filtered_upper_bd, filtered_lower_bd)
                
                bev_gen.generate_bev(
                    vehicles, 
                    filtered_upper_bd, 
                    filtered_auxiliary_bd, 
                    filtered_lower_bd, 
                    frame_id, 
                    output_dir=str(output_dir)
                )
            
            print(f"处理 {csv_file.name}, BEV图像保存到 {output_dir}")

    @staticmethod
    def load_datasets(output_base_dir: str) -> List[torch_data.Dataset]:
        """
        加载保存的数据集

        Args:
            output_base_dir (str): 数据集基础目录

        Returns:
            List[torch_data.Dataset]: 已加载的数据集列表
        """
        datasets = []
        for dataset_file in Path(output_base_dir).rglob("*_dataset.pt"):
            dataset = torch.load(dataset_file)
            datasets.append(dataset)
        return datasets


def main():
    # 配置参数
    map_path = "LM_data/map/DR_CHN_Merging_ZS.json"
    base_data_path = "LM_data/data/DR_CHN_Merging_ZS"
    
    # 初始化数据处理器
    processor = LaneMergingDataProcessor(map_path, base_data_path)

    # 可选的处理任务
    # 1. 生成场景视频
    # processor.save_scenes_as_videos("case_videos_all")

    # 2. 生成真实训练数据
    # 使用自定义参数生成训练数据
    
    processor.generate_real_training_data(
        "get_LM_scene/train_data_xy",
        history_steps=20,      # 增加历史时间步长
        future_steps=1,        # 增加预测未来时间步长
        max_vehicles=10,       # 增加最大主车辆数
        max_aux_vehicles=5,   # 增加最大辅助车辆数
    )

    # 3. 生成BEV图像
    # processor.generate_bev_images("get_LM_scene/train_bev_images")


if __name__ == "__main__":
    main()