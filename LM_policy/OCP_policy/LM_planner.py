#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import lib

#! 后面考虑使用acados替代（如果实时性差）
import numpy as np
from utils.observation import Observation
import math
from . import DMS_planner



class LMOCPPLANNER():
    def __init__(self):
        self.dt = 0.1
        self.Np = 20  # 预测步长
        self.planner = DMS_planner.dms_planner(self.dt, self.Np)  # 初始化求解器

      
    def act(self, observation: Observation):
        
        
        
        
        
        # TODO：获取主车和环境状态，建模OCP规划问题
        self.planner.build_model(
            coeffi, observation.ego_info.length/1.7, Q, R, maxA, minA, maxDeltaRate)
        self.planner.generate_constraint(
            state_ini, state_final, state_cons, obs_collide, self.w_opt, d_safe, car_following_list, follow_distance, maxDelta)
        
        S_opt, L_opt, Theta_opt, V_opt, self.A_opt, self.delta_opt, self.w_opt, self.ocp_state = self.planner.solve()
        act_acc = np.clip(self.A_opt[1], -9, 9)
        act_delta = np.clip(self.delta_opt[1], -0.65, 0.65)


        return [act_acc, act_delta]