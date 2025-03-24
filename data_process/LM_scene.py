import json
import numpy as np
import pandas as pd

#! 提取数据集中的merge信息，用于分析和可视化
class LMScene:
    def __init__(self, json_file , excel_file):
        """
        初始化场景类，从 JSON 文件读取地图数据
        :param json_file: JSON 文件路径
        """
        with open(json_file, 'r', encoding='utf-8') as f:  # 确保编码为 UTF-8
            self.map_data = json.load(f)

        # 提取地图信息
        self.upper_boundary = self._extract_boundary("上沿边界")
        self.auxiliary_dotted_line = self._extract_boundary("辅道虚线")
        self.main_lower_boundary = self._extract_boundary("主道下边界")
        self.restricted_area = self._extract_boundary("辅道限制加入区域")
        self.road_boundary = (997, 1146, 957, 974)  # 道路边界
        
        # 整合地图信息
        self.map_dict = self._map_dict()
        
        # 提取车辆信息
        self.all_vehicles = self._read_vehicle_data(excel_file)
        
        #  过滤后的车辆信息
        self.vehicles = self._filter_vehicles()
        # 汇入区域的阈值（车辆到上沿边界的距离）——需要依据实际情况进行调整
        self.threshold = 3
        self.merge_vehicle = self._get_merge_vehicles()
        
        self.good = 1
        
    def _map_dict(self):
        """
        将地图数据重新整合为字典，方便后续可视化绘制
        :return: 包含地图信息的字典
        """
        map_dict = {
            'upper_boundary': self.upper_boundary,
            'auxiliary_dotted_line': self.auxiliary_dotted_line,
            'main_lower_boundary': self.main_lower_boundary,
            'restricted_area': self.restricted_area,
            'road_boundary': self.road_boundary,  # 直接保存边界数据
        }
        return map_dict
       
    def _read_vehicle_data(self, csv_file):
        """
        从 CSV 文件中读取车辆信息，并按 track_id 分组
        :param csv_file: CSV 文件路径
        :return: 包含所有车辆信息的字典，键为 track_id，值为 Vehicle 对象
        """
        # 读取 CSV 文件
        df = pd.read_csv(csv_file)

        # 按 track_id 分组
        grouped = df.groupby("track_id")

        # 存储所有车辆信息
        vehicles = {}
        for track_id, group in grouped:
            # 提取车辆的状态信息（时间序列）
            frames = group["frame_id"].tolist()
            timestamps = group["timestamp_ms"].tolist()
            x_coords = group["x"].tolist()
            y_coords = group["y"].tolist()
            vx_values = group["vx"].tolist()
            vy_values = group["vy"].tolist()
            psi_rad_values = group["psi_rad"].tolist()

            # 创建 Vehicle 对象
            vehicle = self.Vehicle(
                track_id=track_id,
                frames=frames,
                timestamps=timestamps,
                x_coords=x_coords,
                y_coords=y_coords,
                vx_values=vx_values,
                vy_values=vy_values,
                psi_rad_values=psi_rad_values,
                length=group["length"].iloc[0],  # 车辆长度（假设不变）
                width=group["width"].iloc[0]    # 车辆宽度（假设不变）
            )
            vehicles[track_id] = vehicle

        return vehicles

    class Vehicle:
        """
        车辆信息类，用于存储每辆车的详细信息
        """
        def __init__(self, track_id, frames, timestamps, x_coords, y_coords, vx_values, vy_values, psi_rad_values, length, width):
            self.track_id = track_id  # 车辆 ID
            self.frames = frames  # 帧 ID 列表
            self.timestamps = timestamps  # 时间戳列表（毫秒）
            self.x_coords = x_coords  # 车辆 x 坐标列表
            self.y_coords = y_coords  # 车辆 y 坐标列表
            self.vx_values = vx_values  # 车辆 x 方向速度列表
            self.vy_values = vy_values  # 车辆 y 方向速度列表
            self.psi_rad_values = psi_rad_values  # 车辆航向角列表（弧度）
            self.length = length  # 车辆长度
            self.width = width  # 车辆宽度

        def __repr__(self):
            """
            返回车辆信息的字符串表示
            """
            return (f"Vehicle(track_id={self.track_id}, "
                    f"frames={self.frames}, timestamps={self.timestamps}, "
                    f"x_coords={self.x_coords}, y_coords={self.y_coords}, "
                    f"vx_values={self.vx_values}, vy_values={self.vy_values}, "
                    f"psi_rad_values={self.psi_rad_values}, "
                    f"length={self.length}, width={self.width})")
    

    def _extract_boundary(self, key):
        """
        提取边界点
        :param key: 边界名称（如 "上沿边界"）
        :return: 边界点，形状为 (N, 2) 的 NumPy 数组
        """
        return np.array(list(zip(self.map_data[key]["x"], self.map_data[key]["y"])))

    def get_upper_boundary(self):
        """获取上沿边界"""
        return self.upper_boundary

    def get_auxiliary_dotted_line(self):
        """获取辅道虚线"""
        return self.auxiliary_dotted_line

    def get_main_lower_boundary(self):
        """获取主道下边界"""
        return self.main_lower_boundary

    def get_restricted_area(self):
        """获取辅道限制加入区域"""
        return self.restricted_area
    
    
    # 判断车辆是否在上沿边界以下
    def _is_below_upper_boundary(self, x, y):
        """
        判断车辆是否在上沿边界以下
        :param x: 车辆的 x 坐标
        :param y: 车辆的 y 坐标
        :return: 如果车辆在上沿边界以下返回 True，否则返回 False
        """
        # 获取上沿边界的坐标点
        upper_boundary = self.get_upper_boundary()
        
        # 找到与车辆 x 坐标最接近的上沿边界点
        closest_point = min(upper_boundary, key=lambda point: abs(point[0] - x))
        
        # 判断车辆的 y 坐标是否小于上沿边界的 y 坐标
        return y < closest_point[1]
    
    # 判断车辆是否在下沿边界以上
    def _is_above_lower_boundary(self, x, y):
        """
        判断车辆是否在下沿边界以上
        :param x: 车辆的 x 坐标
        :param y: 车辆的 y 坐标
        :return: 如果车辆在下沿边界以上返回 True，否则返回 False
        """
        # 获取下沿边界的坐标点
        lower_boundary = self.get_main_lower_boundary()
        
        # 找到与车辆 x 坐标最接近的下沿边界点
        closest_point = min(lower_boundary, key=lambda point: abs(point[0] - x))
        
        # 判断车辆的 y 坐标是否大于下沿边界的 y 坐标
        return y > closest_point[1]
    
    def _is_above_auxiliary_boundary(self, x, y):
        """
        判断车辆是否在下沿边界以上
        :param x: 车辆的 x 坐标
        :param y: 车辆的 y 坐标
        :return: 如果车辆在下沿边界以上返回 True，否则返回 False
        """
        # 获取下沿边界的坐标点
        lower_boundary = self.get_auxiliary_dotted_line()
        
        # 找到与车辆 x 坐标最接近的下沿边界点
        closest_point = min(lower_boundary, key=lambda point: abs(point[0] - x))
        
        # 判断车辆的 y 坐标是否大于下沿边界的 y 坐标
        return y > closest_point[1]
    
    import numpy as np

    def _is_close_to_auxiliary_boundary(self, x, y, threshold=2.5):
        """
        判断车辆是否足够接近辅道虚线边界的最近点。
        :param x: 车辆的 x 坐标
        :param y: 车辆的 y 坐标
        :param threshold: 判断接近的距离阈值，默认 1.0
        :return: 如果车辆与虚线的最近点的距离小于阈值，则返回 True，否则返回 False
        """
        # 获取辅道虚线的坐标点
        lower_boundary = self.get_auxiliary_dotted_line()

        # 找到车辆到虚线的最近距离
        min_distance = float('inf')
        
        # 遍历每一段虚线，计算车辆到该线段的最短距离
        for i in range(len(lower_boundary) - 1):
            x1, y1 = lower_boundary[i]
            x2, y2 = lower_boundary[i + 1]

            # 计算点到线段的最短距离
            dx1 = x - x1
            dy1 = y - y1
            dx2 = x2 - x1
            dy2 = y2 - y1

            # 计算点到线段的投影系数 t
            t = ((dx1 * dx2 + dy1 * dy2) / (dx2**2 + dy2**2))

            # 限制 t 的范围在 [0, 1] 之间，确保最近点在线段上
            t = max(0, min(1, t))

            # 计算最近点的坐标
            closest_x = x1 + t * dx2
            closest_y = y1 + t * dy2

            # 计算车辆到最近点的距离
            dist = np.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
            min_distance = min(min_distance, dist)

        # 判断最小距离是否小于阈值
        return min_distance <= threshold

    
    # 滤除非汇流场景的车辆   
    def _filter_vehicles(self):
        """
        过滤车辆，只保留在上沿边界以下且在下沿边界以上的车辆
        :return: 过滤后的车辆字典
        """
        filtered_vehicles = {}

        for track_id, vehicle in self.all_vehicles.items():
            # 获取车辆的初始位置（第一帧的位置）
            initial_x = vehicle.x_coords[0]
            initial_y = vehicle.y_coords[0]

            # 检查车辆是否在上沿边界以下且在下沿边界以上
            if self._is_below_upper_boundary(initial_x, initial_y) and self._is_above_lower_boundary(initial_x, initial_y):
                filtered_vehicles[track_id] = vehicle

        return filtered_vehicles
    
    def _get_merge_vehicles(self):
        """
        获取所有汇入车辆
        :return: 汇入车辆列表
        """
        merge_vehicles = []
        for vehicle in self.vehicles.values():
            # 获取车辆的初始位置（第一帧的位置）
            initial_x = vehicle.x_coords[0]
            initial_y = vehicle.y_coords[0]

            # 依据车辆到上边界的距离判断是否在汇入区域内
            #if self._is_inside_merge_area(initial_x, initial_y):
            if self._is_below_upper_boundary(initial_x, initial_y) and self._is_above_auxiliary_boundary(initial_x, initial_y)and self._is_close_to_auxiliary_boundary(initial_x, initial_y):    
                merge_vehicles.append(vehicle)

        return merge_vehicles

 

    def _get_vehicle_frame_info(self, vehicle, frame):
        """
        获取某辆车在指定帧下的相关信息。
        :param vehicle: 车辆对象
        :param frame: 帧编号
        :return: 该车辆在指定帧的相关信息（如时间戳、位置、速度等）
        """
        # 查找车辆在该帧的信息
        idx = vehicle.frames.index(frame)
        
        vehicle_frame_info = {
            'timestamp': vehicle.timestamps[idx],
            'id': vehicle.track_id,
            'x': vehicle.x_coords[idx],
            'y': vehicle.y_coords[idx],
            'vx': vehicle.vx_values[idx],
            'vy': vehicle.vy_values[idx],
            'psi_rad': vehicle.psi_rad_values[idx],
            'length': vehicle.length,
            'width': vehicle.width,
        }
        
        return vehicle_frame_info
    
    def get_scene(self):
        """
        按照规则进行场景切分，切分规则如下：
        1. 每个场景以一辆汇入车辆（merge_vehicle）为主车
        2. 该主车在其帧时间范围内的其他车辆作为背景车
        :return: 场景切片列表
        """
        scenes = []

        # 1. **预处理阶段**
        # 建立一个字典，key=帧编号, value=该帧下的所有车辆
        frame_to_vehicles = {}
        for vehicle in self.vehicles.values():
            for frame in vehicle.frames:
                if frame not in frame_to_vehicles:
                    frame_to_vehicles[frame] = []
                frame_to_vehicles[frame].append(vehicle)

        # 场景序号初始化
        scene_id_counter = 1

        # 2. **遍历所有汇入车辆**
        for vehicle in self.merge_vehicle:
            # 获取当前汇入车辆的时间范围
            min_frame = min(vehicle.frames)
            max_frame = max(vehicle.frames)
            valid_frames = set(vehicle.frames)  # 主车的所有帧

            # 创建一个场景对象
            scene = {
                'scene_id': scene_id_counter,  # 添加场景序号
                'main_vehicle': vehicle,       # 主车是当前的汇入车辆
                'frames': {},                  # 存放帧数据，每一帧包含主车和环境车信息
            }

            # 3. **遍历主车的所有时间帧**
            for frame in valid_frames:
                # 获取当前帧的主车信息
                vehicle_frame_info = self._get_vehicle_frame_info(vehicle, frame)

                # 直接从 `frame_to_vehicles` 取出该帧的所有车辆
                frame_background_vehicles = []
                for other_vehicle in frame_to_vehicles.get(frame, []):
                    if other_vehicle.track_id != vehicle.track_id:
                        frame_background_vehicles.append(self._get_vehicle_frame_info(other_vehicle, frame))

                # 记录该帧的数据
                scene['frames'][frame] = {
                    'main_vehicle': vehicle_frame_info,
                    'background_vehicles': frame_background_vehicles
                }

            # 4. **添加到场景列表**
            scenes.append(scene)
            scene_id_counter += 1  # 递增场景序号

        return scenes

                
# 示例使用
if __name__ == "__main__":
    # 从 JSON 文件加载地图
    LM_scene = LMScene("get_scence/merge_map/DR_CHN_Merging_ZS.json","interaction_dataset/recorded_trackfiles/DR_CHN_Merging_ZS/vehicle_tracks_000.csv")    
    # 处理并划分场景
    Map = LM_scene.map_dict
    Scenes = LM_scene.get_scene()
    
    
    

    
    
    
    
