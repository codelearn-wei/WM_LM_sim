
# 动作空间设置1：
# MAIN_INTERACTIVE两种动作：
# 1、选择主道前车以IDM跟车
# 2、选择辅道车以IDM跟车

# 辅道车辆的两种动作：
# 1、选择辅道前车IDM跟车
# 2、辅道规划一条换道轨迹，并沿着这个轨迹IDM更主道前车

from typing import Dict
from enum import Enum


class ActionType(Enum):
    IDM_ZV_2_LZV = 0      # IDM follow main road front vehicle
    IDM_ZV_2_FV = 1         # IDM follow auxiliary road vehicle (SUBJECT)
    IDM_FV_2_LFV = 2   # IDM follow auxiliary road front vehicle
    IDM_FV_MERGE_1 = 3
    IDM_FV_MERGE_2 = 4
    IDM_FV_MERGE_3 = 5
    IDM_FV_MERGE_4 = 6
    IDM_FV_MERGE_5 = 7
    # Plan lane change and IDM follow main road front vehicle

class VehicleRole(Enum):
    FV = 0
    ZV = 1   

class ActionSpace:
    def __init__(self, params: Dict):
        self.params = params
        self.action_space = self._define_action_space()

    def _define_action_space(self):
        action_space = {}
        action_space[VehicleRole.FV] = [
            {'type': ActionType.IDM_FV_2_LFV, 'value': None},
            {'type': ActionType.IDM_FV_MERGE_1, 'value': None},
            {'type': ActionType.IDM_FV_MERGE_2, 'value': None},
            {'type': ActionType.IDM_FV_MERGE_3, 'value': None},
            {'type': ActionType.IDM_FV_MERGE_4, 'value': None},
            {'type': ActionType.IDM_FV_MERGE_5, 'value': None}
        ]
        action_space[VehicleRole.ZV] = [
            {'type': ActionType.IDM_ZV_2_LZV, 'value': None},
            {'type': ActionType.IDM_ZV_2_FV, 'value': None}
        ]
        return action_space

    def get_actions(self, role: VehicleRole):
        return self.action_space[role]
 
 
  
    
# 动作空间设置2：
# MAIN_INTERACTIVE多种动作：
# 1、采样一系列的加速的轨迹（加速，减速，维持IDM跟车）基于逆强化学习


# 辅道车辆的多种动作：
# 1、规划一条换道轨迹
# 2、采样一系列的加速度轨迹
# 3、维持IDM跟主道前车
