from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from bisect import bisect_left


class BoundaryProcessor:
    """Utility class to preprocess and query boundary data efficiently."""
    def __init__(self, boundary: List[Tuple[float, float]], segment_size: int = 10):
        self.sorted_boundary = np.array(sorted(boundary, key=lambda p: p[0]))
        self.x_coords = self.sorted_boundary[:, 0]
        self.segments = np.stack([self.sorted_boundary[:-1], self.sorted_boundary[1:]], axis=1)
        # Precompute segment ranges
        self.segment_size = segment_size
        self.num_segments = len(self.segments)
        self.range_indices = np.arange(0, self.num_segments, segment_size)


    def find_closest_index(self, x: float) -> int:
        """Find the index of the closest x-coordinate using binary search."""
        idx = bisect_left(self.x_coords, x)
        if idx == 0:
            return 0
        if idx == len(self.x_coords):
            return len(self.x_coords) - 1
        return idx - 1 if (x - self.x_coords[idx - 1]) < (self.x_coords[idx] - x) else idx
    
    def get_min_distances(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.array([])

        x_coords = points[:, 0]
        # Find approximate segment range for each point
        start_idx = np.searchsorted(self.x_coords[:-1], x_coords, side='left')
        start_idx = np.clip(start_idx, 0, self.num_segments - 1)
        # Define a window of segments to check
        window_size = 5  # Adjustable parameter
        seg_starts = np.maximum(start_idx - window_size, 0)
        seg_ends = np.minimum(start_idx + window_size, self.num_segments)

        min_dists = np.full(len(points), np.inf)
        for i, (start, end) in enumerate(zip(seg_starts, seg_ends)):
            p = points[i:i+1, np.newaxis, :]  # (1, 1, 2)
            local_segments = self.segments[start:end, :, np.newaxis, :]  # (local_n, 2, 1, 2)
            seg_start = local_segments[:, 0]
            seg_end = local_segments[:, 1]
            seg_vec = seg_end - seg_start
            start_to_p = p - seg_start
            seg_len_sq = np.sum(seg_vec ** 2, axis=2, keepdims=True)
            t = np.sum(start_to_p * seg_vec, axis=2, keepdims=True) / (seg_len_sq + 1e-9)
            t = np.clip(t, 0, 1)
            closest = seg_start + t * seg_vec
            dist = np.linalg.norm(p - closest, axis=2)
            min_dists[i] = np.min(dist)

        return min_dists

    def is_below_batch(self, points: np.ndarray) -> np.ndarray:
        """Check if points are below the boundary in batch."""
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        # Batch binary search
        indices = np.searchsorted(self.x_coords, x_coords, side='left')
        indices = np.clip(indices, 0, len(self.x_coords) - 1)
        # Handle edge cases
        left_idx = np.maximum(indices - 1, 0)
        right_idx = indices
        closer_left = (x_coords - self.x_coords[left_idx]) < (self.x_coords[right_idx] - x_coords)
        final_idx = np.where(closer_left, left_idx, right_idx)
        # Compare y coordinates
        return y_coords < self.sorted_boundary[final_idx, 1]

    def is_above_batch(self, points: np.ndarray) -> np.ndarray:
        """Check if points are above the boundary in batch."""
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        indices = np.searchsorted(self.x_coords, x_coords, side='left')
        indices = np.clip(indices, 0, len(self.x_coords) - 1)
        left_idx = np.maximum(indices - 1, 0)
        right_idx = indices
        closer_left = (x_coords - self.x_coords[left_idx]) < (self.x_coords[right_idx] - x_coords)
        final_idx = np.where(closer_left, left_idx, right_idx)
        return y_coords > self.sorted_boundary[final_idx, 1]

    def get_min_distance(self, x: float, y: float) -> float:
        """Calculate the minimum distance to the boundary segments."""
        point = np.array([x, y])
        segments = np.stack([self.sorted_boundary[:-1], self.sorted_boundary[1:]], axis=1)
        distances = np.array([
            np.linalg.norm(np.cross(segments[i, 1] - segments[i, 0], segments[i, 0] - point)) / 
            np.linalg.norm(segments[i, 1] - segments[i, 0])
            for i in range(len(segments))
        ])
        return np.min(distances)
    
def organize_by_frame(vehicles):
    """Organize vehicle data by frame, including all vehicle info per frame."""
    frame_data = defaultdict(list)
    for vehicle in vehicles.values():
        for idx, frame in enumerate(vehicle.frames):
            frame_data[frame].append({
                "track_id": vehicle.track_id,
                "timestamp": vehicle.timestamps[idx],
                "x": vehicle.x_coords[idx],
                "y": vehicle.y_coords[idx],
                "vx": vehicle.vx_values[idx],
                "vy": vehicle.vy_values[idx],
                "psi_rad": vehicle.psi_rad_values[idx],
                "length": vehicle.length,
                "width": vehicle.width,
            })
    return dict(frame_data)  # Convert defaultdict to dict for consistency


def filter_vehicles_by_x(frame_data: Dict[int, List[dict]], x_threshold: float = 1055) -> Dict[int, List[dict]]:
    """Filter vehicles based on an x-coordinate threshold."""
    return {
        frame: [vehicle for vehicle in vehicles if vehicle["x"] > x_threshold]
        for frame, vehicles in frame_data.items()
        if any(vehicle["x"] > x_threshold for vehicle in vehicles)
    }
    
def filter_boundary_points(points, x_threshold=1055):
    """
    过滤边界点集，保留 x >= x_threshold 的点。

    参数:
        points (list or np.ndarray): 边界点集，格式为 [[x1, y1], [x2, y2], ...]
        x_threshold (float): x 坐标的阈值，默认值为 1055

    返回:
        np.ndarray: 过滤后的点集
    """
    # 将输入转换为 NumPy 数组
    points = np.array(points)

    # 检查输入是否为二维数组，且每点包含 x 和 y 坐标
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points 必须是一个形状为 (N, 2) 的二维数组")

    # 使用布尔索引过滤出 x >= x_threshold 的点
    filtered_points = points[points[:, 0] >= x_threshold]

    return filtered_points

def filter_all_boundaries(upper_bd, auxiliary_bd, lower_bd, x_threshold=1055):
    """
    过滤所有边界点集，保留 x >= x_threshold 的点。

    参数:
        upper_bd (list or np.ndarray): 上边界点集
        auxiliary_bd (list or np.ndarray): 辅助边界点集
        lower_bd (list or np.ndarray): 下边界点集
        x_threshold (float): x 坐标的阈值，默认值为 1055

    返回:
        tuple: 过滤后的 upper_bd, auxiliary_bd, lower_bd
    """
    # 对每个边界点集分别进行过滤
    filtered_upper_bd = filter_boundary_points(upper_bd, x_threshold)
    filtered_auxiliary_bd = filter_boundary_points(auxiliary_bd, x_threshold)
    filtered_lower_bd = filter_boundary_points(lower_bd, x_threshold)

    return filtered_upper_bd, filtered_auxiliary_bd, filtered_lower_bd

def classify_vehicles(
    frame_data: Dict[int, List[dict]],
    upper_boundary: List[Tuple[float, float]],
    auxiliary_boundary: List[Tuple[float, float]],
) -> Dict[int, List[dict]]:
    upper_processor = BoundaryProcessor(upper_boundary)
    aux_processor = BoundaryProcessor(auxiliary_boundary)

    for frame, vehicles in frame_data.items():
        if not vehicles:
            continue

        coords = np.array([[v["x"], v["y"]] for v in vehicles])
        # Batch process all conditions
        below_upper = upper_processor.is_below_batch(coords)
        above_aux = aux_processor.is_above_batch(coords)

        # Assign lane types
        for i, vehicle in enumerate(vehicles):
            if below_upper[i] and above_aux[i] :
                vehicle["lane_type"] = "变道车辆"
            else:
                vehicle["lane_type"] = "主道车辆"

    return frame_data

def classify_vehicles_by_frame_1(
    frame_data: Dict[int, List[dict]],
    upper_boundary: List[Tuple[float, float]],
    auxiliary_boundary: List[Tuple[float, float]],
) -> Dict[int, List[dict]]:
    # 创建边界处理器
    upper_processor = BoundaryProcessor(upper_boundary)
    aux_processor = BoundaryProcessor(auxiliary_boundary)

    # 获取按时间顺序排序的帧
    frames = sorted(frame_data.keys())
    if not frames:
        return frame_data

    # 记录已处理的车辆ID及其类型
    processed_vehicles = {}

    # 处理所有帧
    for frame in frames:
        vehicles = frame_data[frame]
        if not vehicles:
            continue

        # 获取当前帧中未处理过的车辆
        new_vehicles = [v for v in vehicles if v["track_id"] not in processed_vehicles]
        if new_vehicles:
            coords = np.array([[v["x"], v["y"]] for v in new_vehicles])
            below_upper = upper_processor.is_below_batch(coords)
            above_aux = aux_processor.is_above_batch(coords)

            # 处理新出现的车辆
            for i, vehicle in enumerate(new_vehicles):
                if below_upper[i] and above_aux[i]:
                    vehicle["lane_type"] = "变道车辆"
                    processed_vehicles[vehicle["track_id"]] = "变道车辆"
                else:
                    vehicle["lane_type"] = "主道车辆"
                    processed_vehicles[vehicle["track_id"]] = "主道车辆"

        # 为已处理过的车辆分配之前确定的车道类型
        for vehicle in vehicles:
            if vehicle["track_id"] in processed_vehicles:
                vehicle["lane_type"] = processed_vehicles[vehicle["track_id"]]

    return frame_data


