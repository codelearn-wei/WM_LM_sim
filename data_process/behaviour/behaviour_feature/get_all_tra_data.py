# 获得每一条的轨迹数据
from pathlib import Path
import pickle
from data_process.train_raw_data import organize_by_frame, classify_vehicles_by_frame_1
from data_process.LM_scene import LMScene
import json
import pandas as pd

def _get_vehicle_tra(json_path: str, excel_path: str):
    # 检查文件是否存在
    if not Path(json_path).exists():
        raise FileNotFoundError(f"JSON 文件未找到: {json_path}")
    if not Path(excel_path).exists():
        raise FileNotFoundError(f"Excel 文件未找到: {excel_path}")
    
    # 读取 JSON 文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"无法解析 JSON 文件: {json_path}")
    
    # 读取 Excel 文件（假设是 CSV 格式，也可以用 pd.read_excel 处理 .xlsx 文件）
    try:
        vehicle_data = pd.read_csv(excel_path)
    except Exception as e:
        raise ValueError(f"无法读取 Excel 文件: {excel_path}, 错误: {str(e)}")
    
    # 初始化 LMScene 对象
    scene = LMScene(json_path, excel_path)
    # 获取所有车辆的 ID 集合
    all_vehicle_ids = set()
    merge_vehicle_ids = set()
    for key in scene.vehicles.keys():
        all_vehicle_ids.add(scene.vehicles[key].track_id)
  
    for i in range(len(scene.merge_vehicle)):
        merge_vehicle_ids.add(scene.merge_vehicle[i].track_id)
         
    main_road_vehicle_ids = sorted(all_vehicle_ids - merge_vehicle_ids)
         
    # merge_vehicles = scene.merge_vehicle
    merge_vehicles = {i: v for i, v in enumerate(sorted(scene.merge_vehicle, key=lambda x: x.track_id))}
    # main_road_vehicles = {vid: scene.vehicles[vid] for vid in main_road_vehicle_ids}
    main_road_vehicles = {i: scene.vehicles[vid] for i, vid in enumerate(main_road_vehicle_ids)}

                 
    return main_road_vehicles, merge_vehicles
    
def get_mutli_vehicle_tra(json_path: str, dir_path: str):
    all_main_tra = {}
    all_merge_tra = {}
    for csv_file in Path(dir_path).glob("*.csv"):
        file_name = csv_file.stem
        main_road_vehicles, merge_vehicle = _get_vehicle_tra(json_path, str(csv_file))
        all_main_tra[file_name] = main_road_vehicles
        all_merge_tra[file_name] = merge_vehicle
    
    return all_main_tra, all_merge_tra

def save_data_to_pkl(all_main_tra, all_merge_tra, output_dir="processed_data"):
    """
    将车辆轨迹数据保存到pkl文件中
    
    参数:
    all_main_tra: 主路车辆轨迹数据
    all_merge_tra: 合流车辆轨迹数据
    output_dir: 输出目录，默认为processed_data
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 保存主路车辆数据
    main_road_path = output_path / "all_tra_data_main.pkl"
    with open(main_road_path, 'wb') as f:
        pickle.dump(all_main_tra, f)
    
    # 保存合流车辆数据
    merge_path = output_path / "all_tra_data_merge.pkl"
    with open(merge_path, 'wb') as f:
        pickle.dump(all_merge_tra, f)
    
    print(f"数据已保存到：{output_path}")
    print(f"主路车辆数据：{main_road_path}")
    print(f"合流车辆数据：{merge_path}")
    
def load_vehicle_data(data_dir="processed_data"):
    """
    读取保存的车辆轨迹数据
    
    参数:
    data_dir: 数据目录，默认为processed_data
    
    返回:
    all_main_tra: 主路车辆轨迹数据
    all_merge_tra: 合流车辆轨迹数据
    """
    data_path = Path(data_dir)
    
    # 读取主路车辆数据
    main_road_path = data_path / "main_road_vehicles.pkl"
    if not main_road_path.exists():
        raise FileNotFoundError(f"主路车辆数据文件未找到: {main_road_path}")
    
    with open(main_road_path, 'rb') as f:
        all_main_tra = pickle.load(f)
    
    # 读取合流车辆数据
    merge_path = data_path / "merge_vehicles.pkl"
    if not merge_path.exists():
        raise FileNotFoundError(f"合流车辆数据文件未找到: {merge_path}")
    
    with open(merge_path, 'rb') as f:
        all_merge_tra = pickle.load(f)
    
    return all_main_tra, all_merge_tra

if __name__ == "__main__":
    # 参数设置
    map_path = r"LM_data\map\DR_CHN_Merging_ZS.json"
    dir_path = r"LM_data\data\DR_CHN_Merging_ZS"
    output_dir = r"data_process\behaviour\data"  # 可以根据需要修改输出目录
    
    # 获取车辆轨迹数据
    all_main_tra, all_merge_tra = get_mutli_vehicle_tra(map_path, dir_path)
    print(f"已成功获取数据，主路车辆场景数量: {len(all_main_tra.keys())}")
    
    # 保存数据到pkl文件
    save_data_to_pkl(all_main_tra, all_merge_tra, output_dir)
    
    # 测试读取数据
    loaded_main_tra, loaded_merge_tra = load_vehicle_data(output_dir)
    print("数据读取测试成功!")
    print(f"读取到的主路车辆场景列表: {list(loaded_main_tra.keys())}")