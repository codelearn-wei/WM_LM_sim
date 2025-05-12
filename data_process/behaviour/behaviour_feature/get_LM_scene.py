# TODO: 利用找到的lane_change_info的周围帧的变道数据，进一步完善变道的博弈模型。

from get_all_scene_data import load_scenes_data
import json
import numpy as np
import os
import pickle

def find_nearest_point_on_line(vehicle_x, vehicle_y, line_x, line_y):
    """
    Find the nearest point on a line (辅道虚线) to a vehicle position
    
    Args:
        vehicle_x (float): Vehicle's x position
        vehicle_y (float): Vehicle's y position
        line_x (list): List of x coordinates for the line
        line_y (list): List of y coordinates for the line
        
    Returns:
        tuple: (nearest x, nearest y, distance, segment index)
    """
    min_dist = float('inf')
    nearest_x, nearest_y = None, None
    nearest_idx = -1
    
    # Convert line points to numpy arrays for vectorized operations
    line_x_np = np.array(line_x)
    line_y_np = np.array(line_y)
    
    for i in range(len(line_x) - 1):
        # Define line segment
        x1, y1 = line_x[i], line_y[i]
        x2, y2 = line_x[i + 1], line_y[i + 1]
        
        # Vector from point 1 to point 2
        dx = x2 - x1
        dy = y2 - y1
        
        # Length of line segment squared
        len_squared = dx**2 + dy**2
        
        # Avoid division by zero
        if len_squared == 0:
            continue
            
        # Calculate projection of vehicle point onto line segment
        t = max(0, min(1, ((vehicle_x - x1) * dx + (vehicle_y - y1) * dy) / len_squared))
        
        # Calculate nearest point on the line segment
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        # Calculate distance
        dist = np.sqrt((vehicle_x - proj_x)**2 + (vehicle_y - proj_y)**2)
        
        if dist < min_dist:
            min_dist = dist
            nearest_x, nearest_y = proj_x, proj_y
            nearest_idx = i
    
    return nearest_x, nearest_y, min_dist, nearest_idx

def detect_lane_change_points(scenes_data, map_data, distance_threshold=1.0, interaction_radius=50.0):
    """
    Detect lane change points for main vehicles and gather surrounding vehicle data
    
    Args:
        scenes_data (dict): Dictionary containing vehicle tracks data
        map_data (dict): Dictionary containing map information
        distance_threshold (float): Maximum distance threshold to consider as intersection
        interaction_radius (float): Radius to consider surrounding vehicles as interacting
    
    Returns:
        dict: Dictionary with detected lane change points and related information
    """
    # Get the auxiliary lane dashed line coordinates
    aux_lane_x = map_data['辅道虚线']['x']
    aux_lane_y = map_data['辅道虚线']['y']
    
    lane_change_info = {}
    
    # Process each file
    for file_id, file_data in scenes_data.items():
        print(f"Processing file: {file_id}")
        lane_change_info[file_id] = {}
        
        # Process each scene in the file
        for scene_id, scene_data in file_data.items():
            print(f"  Processing scene: {scene_id}")
            lane_change_info[file_id][scene_id] = []
            
            # Get frame data for this scene
            frames = scene_data
            
            # Track all vehicles across frames (both main and surrounding)
            all_vehicles = {}  # track_id -> list of (frame_idx, vehicle_data)
            main_vehicles = {}  # track_id -> list of (frame_idx, vehicle_data) for main lane-changing vehicles
            
            # First pass: collect all vehicle data
            for idx, frame_data in enumerate(frames):
                for _, vehicle_data in enumerate(frame_data):
                    track_id = vehicle_data['track_id']
                    
                    # Store all vehicles data
                    if track_id not in all_vehicles:
                        all_vehicles[track_id] = []
                    
                    all_vehicles[track_id].append((idx, vehicle_data))
                    
                    # Separately track main lane-changing vehicles
                    if vehicle_data['is_main_vehicle'] and vehicle_data['lane_type'] == '变道车辆':
                        if track_id not in main_vehicles:
                            main_vehicles[track_id] = []
                        
                        main_vehicles[track_id].append((idx, vehicle_data))
            
            # Second pass: analyze each main vehicle's trajectory
            for main_track_id, vehicle_frames in main_vehicles.items():
                # Sort frames by timestamp
                vehicle_frames.sort(key=lambda x: x[1]['timestamp'])
                
                # Check if this vehicle ever crosses the auxiliary lane line
                crossing_frames = []
                
                for frame_idx, vehicle_data in vehicle_frames:
                    veh_x, veh_y = vehicle_data['x'], vehicle_data['y']
                    
                    # Find nearest point on auxiliary lane line
                    nearest_x, nearest_y, dist, line_idx = find_nearest_point_on_line(
                        veh_x, veh_y, aux_lane_x, aux_lane_y
                    )
                    
                    # Store if the vehicle is close to the lane line
                    if dist < distance_threshold:
                        crossing_frames.append({
                            'frame_idx': frame_idx,
                            'timestamp': vehicle_data['timestamp'],
                            'vehicle_data': vehicle_data,
                            'distance_to_line': dist,
                            'nearest_point': (nearest_x, nearest_y),
                            'line_segment_idx': line_idx
                        })
                
                # If we found crossing points, add to our result
                if crossing_frames:
                    # Find the frame with minimum distance to the line (most accurate crossing point)
                    min_dist_frame = min(crossing_frames, key=lambda x: x['distance_to_line'])
                    
                    # Get timestamps for 5 seconds before and 5 seconds after
                    center_timestamp = min_dist_frame['timestamp']
                    
                    # Find first and last timestamp to define the lane change window
                    if vehicle_frames:
                        first_ts = vehicle_frames[0][1]['timestamp']
                        last_ts = vehicle_frames[-1][1]['timestamp']
                        
                        # Define lane change window (5 seconds before and after the crossing point)
                        start_ts = max(first_ts, center_timestamp - 5000)  # 5 seconds before (timestamps in milliseconds)
                        end_ts = min(last_ts, center_timestamp + 5000)    # 5 seconds after
                        
                        # Get surrounding vehicles during the lane change window
                        surrounding_vehicles = {}
                        
                        # For each surrounding vehicle, extract data within the lane change window
                        for other_track_id, other_vehicle_frames in all_vehicles.items():
                            # Skip the main vehicle
                            if other_track_id == main_track_id:
                                continue
                                
                            # Find frames within the lane change window
                            relevant_frames = []
                            for other_frame_idx, other_vehicle_data in other_vehicle_frames:
                                other_ts = other_vehicle_data['timestamp']
                                
                                # Check if this frame is within our window
                                if start_ts <= other_ts <= end_ts:
                                    # Calculate distance to main vehicle at this timestamp
                                    # Find the main vehicle position at the nearest timestamp
                                    main_pos_at_ts = None
                                    min_ts_diff = float('inf')
                                    
                                    for _, main_vehicle_data in vehicle_frames:
                                        main_ts = main_vehicle_data['timestamp']
                                        ts_diff = abs(main_ts - other_ts)
                                        
                                        if ts_diff < min_ts_diff:
                                            min_ts_diff = ts_diff
                                            main_pos_at_ts = (main_vehicle_data['x'], main_vehicle_data['y'])
                                    
                                    # Calculate distance between vehicles
                                    if main_pos_at_ts:
                                        main_x, main_y = main_pos_at_ts
                                        other_x, other_y = other_vehicle_data['x'], other_vehicle_data['y']
                                        distance = np.sqrt((main_x - other_x)**2 + (main_y - other_y)**2)
                                        
                                        # Only include vehicles within interaction radius
                                        if distance <= interaction_radius:
                                            relevant_frames.append((other_frame_idx, other_vehicle_data, distance))
                            
                            # If we found relevant frames for this vehicle, add it to our surrounding vehicles
                            if relevant_frames:
                                surrounding_vehicles[other_track_id] = relevant_frames
                        
                        # Add all data to our result
                        lane_change_info[file_id][scene_id].append({
                            'track_id': main_track_id,
                            'crossing_point': min_dist_frame,
                            'lane_change_window': {
                                'start_timestamp': start_ts,
                                'center_timestamp': center_timestamp,
                                'end_timestamp': end_ts
                            },
                            'main_vehicle_frames': vehicle_frames,  # Store all frames for the main vehicle
                            'surrounding_vehicles': surrounding_vehicles  # Store data for surrounding vehicles
                        })
    
    return lane_change_info

def main():
    data_path = r"data_process\behaviour\data\all_scenes_data.pkl"
    result_path = r"data_process\behaviour\data"
    map_path = r"LM_data\map\DR_CHN_Merging_ZS.json"
    
    # Load data
    scenes_data = load_scenes_data(data_path)
    print("Loaded scenes data. Number of files:", len(scenes_data))
    
    with open(map_path, 'r', encoding='utf-8') as f:
        map_data = json.load(f)
    print("Loaded map data.")
    
    # Create output directory if it doesn't exist
    os.makedirs(result_path, exist_ok=True)
    
    # Step 1: Detect lane change points
    print("Detecting lane change points and gathering surrounding vehicle data...")
    lane_change_info = detect_lane_change_points(scenes_data, map_data, 
                                                interaction_radius=50.0)  # 50 meters interaction radius
    
    # Count detected lane changes and surrounding vehicles
    total_lane_changes = 0
    total_surrounding_vehicles = 0
    for file_id, file_data in lane_change_info.items():
        for scene_id, scene_changes in file_data.items():
            total_lane_changes += len(scene_changes)
            for lane_change in scene_changes:
                total_surrounding_vehicles += len(lane_change['surrounding_vehicles'])
    
    total_scenes = sum(len(file_data) for file_data in lane_change_info.values())
    print(f"Detected {total_lane_changes} lane changes across {total_scenes} scenes in {len(lane_change_info)} files.")
    print(f"Included data from {total_surrounding_vehicles} surrounding vehicles.")
    
    # Save lane change information
    output_file = os.path.join(result_path, 'lane_change_info_with_surroundings.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(lane_change_info, f)
    
    print(f"Lane change information saved to {output_file}")
    print("Analysis complete.")

if __name__ == "__main__":
    main()