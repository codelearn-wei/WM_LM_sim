import cv2
import numpy as np
import math
import os

class BEVGenerator:
    def __init__(self, image_size=(400, 400), resolution=0.1, range_m=40, road_color=(192, 192, 192), non_road_color=(64, 64, 64), default_center=(0, 0)):
        """
        初始化 BEV 生成器。

        参数:
            image_size (tuple): 图像尺寸 (height, width)，默认 (400, 400) 像素
            resolution (float): 每像素代表的实际距离（米/像素），默认 0.1 米
            range_m (float): BEV 视图在每个方向上的覆盖范围（米），默认 20 米
            road_color (tuple): 道路区域的背景颜色（BGR），默认浅灰色 (192, 192, 192)
            non_road_color (tuple): 非道路区域的背景颜色（BGR），默认深灰色 (64, 64, 64)
            default_center (tuple): 未设置自车时 BEV 视图的默认中心坐标，默认 (0, 0)
        """
        self.image_size = image_size
        self.resolution = resolution
        self.range_m = range_m
        self.road_color = road_color
        self.non_road_color = non_road_color
        self.default_center = default_center  # 保存默认中心
        self.ego_vehicle = None  # 自车信息，初始为 None
        # 自动计算覆盖范围（单边距离）
        self.range_m_x = (image_size[1] * resolution) / 2  # 水平方向覆盖范围
        self.range_m_y = (image_size[0] * resolution) / 2  # 垂直方向覆盖范围

    def set_ego_vehicle(self, ego_vehicle=None):
        """
        设置自车信息。

        参数:
            ego_vehicle (dict or None): 自车信息字典，包含 'x', 'y' 等键值；若为 None，则不设置自车
        """
        self.ego_vehicle = ego_vehicle

    def world_to_local(self, x, y):
        """
        将全局坐标转换为局部坐标。

        参数:
            x (float): 全局 x 坐标
            y (float): 全局 y 坐标

        返回:
            tuple: 局部坐标 (x_local, y_local)
        """
        # 根据是否设置自车选择中心点
        if self.ego_vehicle is not None:
            x_center, y_center = self.ego_vehicle['x'], self.ego_vehicle['y']
        else:
            x_center, y_center = self.default_center
        x_local = x - x_center
        y_local = y - y_center
        return x_local, y_local

    def local_to_image(self, x_local, y_local):
        """坐标转换（修正范围计算）"""
        u = int((x_local + self.range_m_x) / self.resolution)
        v = int((-y_local + self.range_m_y) / self.resolution)  # 反转y轴
        return u, v
    

    def world_to_image(self, x, y):
        """
        将全局坐标直接转换为图像坐标。

        参数:
            x (float): 全局 x 坐标
            y (float): 全局 y 坐标

        返回:
            tuple: 图像坐标 (u, v)
        """
        x_local, y_local = self.world_to_local(x, y)
        u, v = self.local_to_image(x_local, y_local)
        return u, v
    
    def set_scene_center(self, upper_bd, lower_bd):
        """
        根据道路边界计算场景的中心点，并设置为 default_center。

        参数:
            upper_bd (list or np.ndarray): 上边界点列表或数组
            lower_bd (list or np.ndarray): 下边界点列表或数组
        """
        # 合并上下边界点
        all_points = np.array(upper_bd + lower_bd)  # 确保转换为 NumPy 数组

        # 计算所有点的平均中心
        if all_points.size > 0:  # 检查数组是否为空
            x_center = np.mean(all_points[:, 0])/2  # 计算 x 坐标的平均值
            y_center = np.mean(all_points[:, 1])/2  # 计算 y 坐标的平均值
            self.default_center = (x_center, y_center)
        else:
            self.default_center = (0, 0)  # 如果没有点，使用默认中心

    def draw_vehicles(self, vehicles, image):
        """绘制所有车辆"""
        for vehicle in vehicles:
            x, y = vehicle['x'], vehicle['y']
            psi_rad = vehicle['psi_rad']
            length, width = vehicle['length'], vehicle['width']
            lane_type = vehicle['lane_type']

            # 计算车辆的四个角点（以车辆中心为原点）
            half_length = length / 2
            half_width = width / 2
            offsets = [
                (half_length, half_width),
                (half_length, -half_width),
                (-half_length, -half_width),
                (-half_length, half_width)
            ]

            # 根据航向角旋转并平移到全局坐标
            corners = []
            for dx, dy in offsets:
                x_offset = dx * math.cos(psi_rad) - dy * math.sin(psi_rad)
                y_offset = dx * math.sin(psi_rad) + dy * math.cos(psi_rad)
                corners.append((x + x_offset, y + y_offset))

            # 转换为图像坐标
            image_corners = [self.world_to_image(cx, cy) for cx, cy in corners]

            # 检查是否超出图像范围
            if any(u < 0 or u >= self.image_size[1] or v < 0 or v >= self.image_size[0] for u, v in image_corners):
                continue

            # 根据车道类型选择颜色（BGR格式）
            if lane_type == '主道车辆':
                color = (0, 0, 255)  # 蓝色
            elif lane_type == '变道车辆':
                color = (255, 0, 0)  # 红色
            else:
                color = (0, 255, 0)  # 绿色（其他类型）

            # 绘制车辆矩形
            pts = np.array(image_corners, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [pts], color)

    def draw_roads(self, upper_bd, auxiliary_bd, lower_bd, image):
        """绘制道路边界和变道虚线"""
        def draw_polyline(points, color, is_dashed=False, thickness=2):
            """绘制折线，is_dashed决定是否为虚线"""
            img_pts = [self.world_to_image(x, y) for x, y in points]
            if is_dashed:
                for i in range(len(img_pts) - 1):
                    if i % 2 == 0:  # 每隔一段绘制
                        cv2.line(image, img_pts[i], img_pts[i + 1], color, thickness)
            else:
                cv2.polylines(image, [np.array(img_pts, np.int32)], isClosed=False, color=color, thickness=thickness)

        # 绘制上边界：灰色实线
        draw_polyline(upper_bd, (128, 128, 128))
        # 绘制变道虚线：黄色虚线
        draw_polyline(auxiliary_bd, (0, 255, 255), is_dashed=True)
        # 绘制下边界：灰色实线
        draw_polyline(lower_bd, (128, 128, 128))

    def draw_road_area(self, upper_bd, lower_bd, image):
        """绘制道路区域背景"""
        # 将上下边界点集转换为图像坐标
        upper_img_pts = [self.world_to_image(x, y) for x, y in upper_bd]
        lower_img_pts = [self.world_to_image(x, y) for x, y in lower_bd]

        # 创建道路区域的多边形（连接上下边界）
        road_polygon = np.array(upper_img_pts + lower_img_pts[::-1], np.int32).reshape((-1, 1, 2))

        # 填充道路区域
        cv2.fillPoly(image, [road_polygon], self.road_color)

    def generate_bev(self, vehicles, upper_bd, auxiliary_bd, lower_bd, frame_id,output_dir = "bev_images"):
            """
            生成 BEV 图像。

            参数:
                vehicles (list): 车辆信息列表
                upper_bd (list): 上边界点列表
                auxiliary_bd (list): 辅助边界点列表
                lower_bd (list): 下边界点列表
                frame_id (int): 帧编号
            """
            # 初始化图像为非道路背景颜色
            image = np.full((self.image_size[0], self.image_size[1], 3), self.non_road_color, dtype=np.uint8)

            # 绘制道路区域背景、边界和车辆（假设这些方法已实现）
            self.draw_road_area(upper_bd, lower_bd, image)
            self.draw_roads(upper_bd, auxiliary_bd, lower_bd, image)
            self.draw_vehicles(vehicles, image)

            # 保存图像
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_id:04d}.png"), image)

