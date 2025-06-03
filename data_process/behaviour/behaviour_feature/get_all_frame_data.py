# 获得每一帧的轨迹数据
from pathlib import Path
import pickle
import os
from data_process.LM_scene import LMScene
from data_process.train_raw_data import (
    organize_by_frame, 
    classify_vehicles_by_frame_1,
    filter_vehicles_by_x, 
)

def frame_data(data_path: str, map_path: str, output_path: str = "processed_data"):
    """
    Process frame data and save it as a pickle file.
    
    Args:
        data_path: Path to the CSV data files
        map_path: Path to the map JSON file
        output_path: Directory to save the pickle file
    
    Returns:
        Dictionary containing all frame data
    """
    all_frame_data = {}
    for csv_file in Path(data_path).glob("*.csv"):
        scene = LMScene(map_path, str(csv_file))
        frame_data = organize_by_frame(scene.vehicles)
        filtered_data = filter_vehicles_by_x(frame_data, x_threshold=1030)

        upper_bd = scene.get_upper_boundary()
        auxiliary_bd = scene.get_auxiliary_dotted_line()
        classified_data = classify_vehicles_by_frame_1(filtered_data, upper_bd, auxiliary_bd)
        file_name = csv_file.stem
        all_frame_data[file_name] = classified_data
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    # Save the data as a pickle file
    output_file = os.path.join(output_path, "all_frame_data.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(all_frame_data, f)
    
    print(f"Data successfully saved to {output_file}")
    return all_frame_data

def load_frame_data(file_path: str):
    """
    Load frame data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
    
    Returns:
        Dictionary containing the loaded frame data
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"Data successfully loaded from {file_path}")
    return data
            
if __name__ == "__main__":
    data_path = r"LM_data\data\DR_CHN_Merging_ZS"
    map_path = r"LM_data\map\DR_CHN_Merging_ZS.json"
    output_dir = r"data_process\behaviour\data"  # 可以根据需要修改输出目录
    
    # Process and save data
    all_frame_data = frame_data(data_path, map_path, output_dir)
    
    # Example of loading the data back
    loaded_data = load_frame_data(os.path.join(output_dir, "all_frame_data.pkl"))
    
    # Verify data is loaded correctly
    print(f"Number of files processed: {len(loaded_data)}")