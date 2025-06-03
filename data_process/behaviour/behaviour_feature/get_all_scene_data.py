# 获得划分的每个场景的数据
import os
import pickle
from get_all_frame_data import frame_data, load_frame_data
from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_file(file_key_frames):
    """处理单个文件的数据"""
    file_key, frames = file_key_frames
    # 收集所有汇入车辆的ID
    merge_vehicle_ids = set()
    for frame_data in frames.values():
        for vehicle in frame_data:
            if vehicle['lane_type'] == '变道车辆':
                merge_vehicle_ids.add(vehicle['track_id'])
    
    # 为每个汇入车辆创建一个场景，并显示场景级进度
    scenes = {}
    for scene_index, merge_vehicle_id in enumerate(tqdm(merge_vehicle_ids, desc=f"处理 {file_key} 的场景")):
        scene_data = []
        for frame_idx, frame_data in frames.items():
            main_vehicle = None
            env_vehicles = []
            for vehicle in frame_data:
                if vehicle['track_id'] == merge_vehicle_id:
                    main_vehicle = deepcopy(vehicle)
                    main_vehicle['is_main_vehicle'] = True
                else:
                    env_vehicle = deepcopy(vehicle)
                    env_vehicle['is_main_vehicle'] = False
                    env_vehicles.append(env_vehicle)
            if main_vehicle:
                frame_vehicles = [main_vehicle] + env_vehicles
                scene_data.append(frame_vehicles)
        if scene_data:
            scenes[scene_index] = scene_data
    return file_key, scenes

def divide_into_scenes(data):
    """将车辆轨迹数据划分成场景，以每辆汇入车辆为主车"""
    result = {}
    # 准备多进程任务
    tasks = [(file_key, frames) for file_key, frames in data.items()]
    # 使用多进程处理每个文件并显示文件级进度
    with Pool(processes=cpu_count()) as pool:
        for file_key, scenes in tqdm(pool.imap_unordered(process_file, tasks), 
                                   total=len(tasks), 
                                   desc="处理文件进度"):
            result[file_key] = scenes
    return result

def save_scenes_data(scenes, output_path):
    """保存场景数据到pickle文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(scenes, f)
    print(f"场景数据已保存到: {output_path}")

def load_scenes_data(input_path):
    """从pickle文件中读取场景数据"""
    with open(input_path, 'rb') as f:
        scenes = pickle.load(f)
    print(f"已从 {input_path} 加载场景数据")
    # 计算总场景数
    total_scenes = sum(len(scenes[file_key]) for file_key in scenes)
    print(f"共加载了 {total_scenes} 个场景")
    return scenes

# 使用示例
if __name__ == "__main__":
    # 示例数据
    output_dir = r"data_process\behaviour\data"
    frame_data_path = os.path.join(output_dir, "all_frame_data.pkl")
    scenes_output_path = os.path.join(output_dir, "all_scenes_data.pkl")
    
    # 加载帧数据
    frame_data = load_frame_data(frame_data_path)
    
    # 划分场景
    scenes = divide_into_scenes(frame_data)
    
    # 输出处理后的场景数量
    total_scenes = sum(len(scenes[file_key]) for file_key in scenes)
    print(f"一共统计的场景数量: {total_scenes}")
    
    # 保存场景数据
    save_scenes_data(scenes, scenes_output_path)
    
    # 示例：如何读取保存的数据
    loaded_scenes = load_scenes_data(scenes_output_path)
    
    # 验证加载的数据是否正确
    assert total_scenes == sum(len(loaded_scenes[file_key]) for file_key in loaded_scenes)
    print("数据验证成功！")