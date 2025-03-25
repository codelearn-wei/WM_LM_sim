from typing import List
from pathlib import Path
import time
from tqdm import tqdm
import torch
import torch.utils.data as torch_data
from LM_vis import SceneVisualizer
from LM_scene import LMScene
from train_bev_gen import BEVGenerator
from train_raw_data import (
    organize_by_frame, 
    classify_vehicles_by_frame_1,
    filter_vehicles_by_x, 
    classify_vehicles, 
    filter_all_boundaries
)

#! 数据处理主类
#! 三种功能
#! 1. 生成回放场景视频
#! 2. 生成真实训练BEV图像

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

    # 2. 生成BEV图像
    processor.generate_bev_images("train_bev_images")


if __name__ == "__main__":
    main()