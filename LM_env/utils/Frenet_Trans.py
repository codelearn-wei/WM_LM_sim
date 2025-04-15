#! /usr/bin/env python3
# _*_ coding: utf-8 _*_

import math
import numpy as np
from scipy.spatial import KDTree
M_PI = 3.141593


def NormalizeAngle(angle_rad):
    # to normalize an angle to [-pi, pi]
    a = math.fmod(angle_rad + M_PI, 2.0 * M_PI)
    if a < 0.0:
        a = a + 2.0 * M_PI
    return a - M_PI
    # return angle_rad   

def Dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class PathPoint:
    def __init__(self, pp_list):
        # pp_list: from CalcRefLine, [rx, ry, rs, rtheta, rkappa, rdkappa] x y 路程 角度 角度变化量/路程变化量 (角度变化量/路程变化量)/路程变化量
        self.rx = pp_list[0]
        self.ry = pp_list[1]
        self.rs = pp_list[2]
        self.rtheta = pp_list[3]
        self.rkappa = pp_list[4]
        self.rdkappa = pp_list[5]

class TrajPoint:
    def __init__(self, tp_list):
        # tp_list: from sensors, [x, y, v, a, theta, kappa]
        self.x = tp_list[0]
        self.y = tp_list[1]
        self.v = tp_list[2]
        self.a = tp_list[3]
        self.theta = tp_list[4]
        self.kappa = tp_list[5]

    def MatchPath(self, path_points,tree):
        '''
        Find the closest/projected point on the reference path using KDTree.
        '''
        # Extract coordinates from path_points

        
        # Query the closest point
        dist, index = tree.query((self.x, self.y))
        
        # Get the closest point
        path_point_min = path_points[index]
        self.matched_point = path_point_min
        return self.matched_point

    def LimitTheta(self, theta_thr=M_PI / 6):
        # limit the deviation of traj_point.theta from the matched path_point.rtheta within theta_thr
        if self.theta - self.matched_point.rtheta > theta_thr:
            self.theta = NormalizeAngle(self.matched_point.rtheta + theta_thr)  # upper limit of theta
        elif self.theta - self.matched_point.rtheta < -theta_thr:
            self.theta = NormalizeAngle(self.matched_point.rtheta - theta_thr)  # lower limit of theta
        else:
            pass  # maintained, actual theta should not deviate from the path rtheta too much





def CartesianToFrenet(path_point, traj_point):
    ''' from Cartesian to Frenet coordinate, to the matched path point
    copy Apollo cartesian_frenet_conversion.cpp'''
    rx, ry, rs, rtheta, rkappa, rdkappa = path_point.rx, path_point.ry, path_point.rs, \
                                          path_point.rtheta, path_point.rkappa, path_point.rdkappa
    x, y, v, a, theta, kappa = traj_point.x, traj_point.y, traj_point.v, \
                               traj_point.a, traj_point.theta, traj_point.kappa
    s_condition = np.zeros(3)
    d_condition = np.zeros(3)
    dx = x - rx
    dy = y - ry
    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d_condition[0] = math.copysign(math.sqrt(dx ** 2 + dy ** 2), cross_rd_nd)
    delta_theta = theta - rtheta
    tan_delta_theta = math.tan(delta_theta)
    cos_delta_theta = math.cos(delta_theta)
    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    d_condition[1] = one_minus_kappa_r_d * tan_delta_theta
    kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
    d_condition[2] = -kappa_r_d_prime * tan_delta_theta + one_minus_kappa_r_d / (cos_delta_theta ** 2) * \
                    (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa)
    s_condition[0] = rs
    s_condition[1] = v * cos_delta_theta / one_minus_kappa_r_d
    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
    s_condition[2] = (a * cos_delta_theta - s_condition[1] ** 2 * \
                      (d_condition[1] * delta_theta_prime - kappa_r_d_prime)) / one_minus_kappa_r_d
    return s_condition, d_condition

def FrenetToCartesian(path_point, s_condition, d_condition):
    ''' from Frenet to Cartesian coordinate
    copy Apollo cartesian_frenet_conversion.cpp'''
    rx, ry, rs, rtheta, rkappa, rdkappa = path_point.rx, path_point.ry, path_point.rs, \
                                          path_point.rtheta, path_point.rkappa, path_point.rdkappa
    if math.fabs(rs - s_condition[0]) >= 1.0e-6:
        pass
        # print("the reference point s and s_condition[0] don't match")
    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)
    x = rx - sin_theta_r * d_condition[0]
    y = ry + cos_theta_r * d_condition[0]
    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    tan_delta_theta = d_condition[1] / one_minus_kappa_r_d
    delta_theta = math.atan2(d_condition[1], one_minus_kappa_r_d)
    cos_delta_theta = math.cos(delta_theta)
    theta = NormalizeAngle(delta_theta + rtheta)
    kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]
    kappa = ((d_condition[2] + kappa_r_d_prime * tan_delta_theta) * cos_delta_theta ** 2 / one_minus_kappa_r_d \
             + rkappa) * cos_delta_theta / one_minus_kappa_r_d
    d_dot = d_condition[1] * s_condition[1]
    v = math.sqrt((one_minus_kappa_r_d * s_condition[1]) ** 2 + d_dot ** 2)
    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
    a = s_condition[2] * one_minus_kappa_r_d / cos_delta_theta + s_condition[1] ** 2 / cos_delta_theta * \
        (d_condition[1] * delta_theta_prime - kappa_r_d_prime)
    tp_list = [x, y, v, a, theta, kappa]
    return tp_list

def unwrap_angles(rtheta):
    '''
    Unwrap angles to make them continuous.
    '''
    rtheta_unwrapped = np.zeros_like(rtheta)
    rtheta_unwrapped[0] = rtheta[0]  # Initialize the first value

    for i in range(1, len(rtheta)):
        delta = rtheta[i] - rtheta[i - 1]
        if delta > np.pi:
            delta -= 2 * np.pi  # Correct for positive jump
        elif delta < -np.pi:
            delta += 2 * np.pi  # Correct for negative jump
        rtheta_unwrapped[i] = rtheta_unwrapped[i - 1] + delta

    return rtheta_unwrapped

def CalcRefLine(cts_points):
    '''
    Deal with reference path points 2d-array
    to calculate rs/rtheta/rkappa/rdkappa according to cartesian points.
    '''
    rx = cts_points[0]  # the x values
    ry = cts_points[1]  # the y values
    n = len(rx)  # number of points
    rs = np.zeros(n)  # distance traveled
    rtheta = np.zeros(n)  # angle
    rkappa = np.zeros(n)  # curvature
    rdkappa = np.zeros(n)  # derivative of curvature

    # Calculate rs, rtheta, and rkappa in a single loop
    for i in range(n):
        if i > 0:
            # Calculate distance traveled (rs)
            dx = rx[i] - rx[i - 1]
            dy = ry[i] - ry[i - 1]
            rs[i] = rs[i - 1] + math.sqrt(dx**2 + dy**2)

        if i < n - 1:
            # Calculate angle (rtheta) using atan2
            dx = rx[i + 1] - rx[i]
            dy = ry[i + 1] - ry[i]
            rtheta[i] = math.atan2(dy, dx)  # Directly compute the angle

        if i > 0 and i < n - 1:
            # Calculate curvature (rkappa) using three points
            x1, y1 = rx[i - 1], ry[i - 1]
            x2, y2 = rx[i], ry[i]
            x3, y3 = rx[i + 1], ry[i + 1]

            # Area of the triangle formed by the three points
            A = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            A = abs(A) / 2.0

            # Lengths of the sides of the triangle
            a = math.sqrt((x2 - x3)**2 + (y2 - y3)**2)
            b = math.sqrt((x1 - x3)**2 + (y1 - y3)**2)
            c = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            # Radius of the circumscribed circle
            if A == 0:
                R = float('inf')  # Straight line, curvature is 0
            else:
                R = (a * b * c) / (4 * A)

            # Curvature
            if R == 0:
                rkappa[i] = 0
            else:
                rkappa[i] = 1.0 / R

    # Unwrap angles to make them continuous
    rtheta = unwrap_angles(rtheta)
    # Handle edge cases for rtheta and rkappa
    if n > 2:
        rtheta[-1] = rtheta[-2]  # Last point's angle equals the second last
        rkappa[0] = rkappa[1]  # First point's curvature equals the second
        rkappa[-1] = rkappa[-2]  # Last point's curvature equals the second last

        # Calculate derivative of curvature (rdkappa)
        rdkappa[:-1] = np.diff(rkappa) / np.diff(rs)
        rdkappa[-1] = rdkappa[-2]  # Last point's derivative equals the second last

    # Generate path points
    path_points = []
    for i in range(n):
        path_points.append(PathPoint([rx[i], ry[i], rs[i], rtheta[i], rkappa[i], rdkappa[i]]))

    return path_points

class VehicleInfo:
    def __init__(self, x, y, v,  a,heading,yaw_rate,length, width):
        self.x = x
        self.y = y
        self.v = v
        self.a =a
        self.length = length
        self.width = width
        self.yaw = heading
        self.heading = heading
        self.yaw_rate = yaw_rate


class Frenet_trans():
    def __init__(self,ref_xy):
  
        self.ref_xy = ref_xy
        rx =self.ref_xy[:,0] 
        ry = self.ref_xy[:,1]
        cts_points = np.array([rx, ry])
        self.path_points = CalcRefLine(cts_points)
        points = [(p.rx, p.ry) for p in self.path_points]
        
        # Build KDTree
        self.tree = KDTree(points)
        self.rs_pp_all = []  # the rs value of all the path points
        for path_point in self.path_points:
            self.rs_pp_all.append(path_point.rs)
        self.rs_pp_all = np.array(self.rs_pp_all)

    def trans_sl(self,x,y,v,a,yaw):
        # obs 障碍物车辆字典，carid  下面有xyvayaw  length width
        # 主车和障碍物位置信息
        x_ego, y_ego, v_ego,a_ego, heading_ego = x, y, v,a, yaw
        heading_ego = NormalizeAngle(heading_ego)
        tp_list = [x_ego, y_ego, v_ego,a_ego , heading_ego, 0]  # from sensor actually, an example here
        traj_point = TrajPoint(tp_list)  ### [x, y, v, a, theta, kappa]
        traj_point.MatchPath(self.path_points,self.tree)  # matching once is enough 将traj_point(单点)与path_points(序列)中最近的点匹配
        s_cond_init, d_cond_init = CartesianToFrenet(traj_point.matched_point,traj_point)  ### 转化坐标系 s d分别速度方向和垂直于速度方向
        return s_cond_init[0],d_cond_init[0]

    
    def trans_xy(self,s,l):
        index_min = np.argmin(np.abs(self.rs_pp_all - s))
        path_point_min = self.path_points[index_min]
        tp_list_c=FrenetToCartesian(path_point_min, [s,0,0], [l,0,0])

        return tp_list_c[0],tp_list_c[1]
    
