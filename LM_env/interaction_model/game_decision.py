from typing import Dict
from enum import Enum
from typing import Dict
import numpy as np

from LM_env.interaction_model.compute_reward import *
class ActionType(Enum):
    # 主车让行行为
    IDM_ZV_2_LZV = 0     
    IDM_ZV_2_FV = 1  
    
    # 辅道车辆行为定义
    IDM_FV_2_LFV = 2   
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
      


# TODO：需要简化收益函数
class PayoffCalculator:
    def __init__(self, params: Dict):
        self.params = params

    def calculate(self, fv, zv, f_future, z_future):
        # 计算主道主车的payoff
        zv_payoff = self._calculate_zv_payoff(fv, zv , f_future, z_future)
        
        # 计算辅道车辆payoff
        fv_payoff = self._calculate_fv_payoff(fv, zv ,f_future, z_future)
        
        return zv_payoff, fv_payoff
    
    def _calculate_zv_payoff(self, fv, zv , f_future, z_future):
        zv_pay_off = 0
        main_future_trajectory = z_future
        aux_future_trajectory = f_future
        auxiliary_vehicle = fv
        main_vehicle = zv
        
        zv_pay_off = calculate_main_rewards(self.params,
                                      main_future_trajectory , 
                                      aux_future_trajectory)
        return zv_pay_off

    def _calculate_fv_payoff(self, fv, zv ,f_future, z_future):
        fv_pay_off = 0
        main_future_trajectory = z_future
        aux_future_trajectory = f_future
        
        fv_pay_off = calculate_aux_rewards(self.params,
                                      main_future_trajectory , 
                                      aux_future_trajectory )   
        return fv_pay_off

def reorganize_trajectory_data(trajectory_data):
    trajectorys = []
    for i, point in enumerate(trajectory_data):
        x = point['position'][0]
        y = point['position'][1]
        v = point['velocity']
        heading = point['heading']
        acc = point['acceleration']
        reorganized_point = [x, y, v, heading, acc]
        trajectorys.append(reorganized_point)
        
    return trajectorys

class GameModel:
    def __init__(self, params: Dict , predict_state: Dict):
        self.params = params
        self.future_state = predict_state
        self.beta_values = {
            VehicleRole.FV: params.get('Rationality_aux', 0.1),
            VehicleRole.ZV: params.get('Rationality_main', 0.8)
        }
        self.action_space = ActionSpace(params)
        self.payoff_calculator = PayoffCalculator(params)
        self.action_set = self._set_action_combination_dict()
        
    def _set_action_combination_dict(self):
        ego_states = [
            self.future_state['ego']['follow_main_front'],  # index 0
            self.future_state['ego']['follow_auxiliary_0']    # index 1
        ]
        
        aux_states = [
            self.future_state['auxiliary'][0]['idm_driving'],           # index 2
            self.future_state['auxiliary'][0]['sampled_trajectory_1'],  # index 3
            self.future_state['auxiliary'][0]['sampled_trajectory_2'],  # index 4
            self.future_state['auxiliary'][0]['sampled_trajectory_3'],  # index 5
            self.future_state['auxiliary'][0]['sampled_trajectory_4'],  # index 6
            self.future_state['auxiliary'][0]['sampled_trajectory_5']   # index 7
        ]

        action_combination_dict = {}

        for i, ego_state in enumerate(ego_states):
            for j, aux_state in enumerate(aux_states, start=2):
                key = (i, j)
                action_combination_dict[key] = {
                    "ego_state": ego_state,
                    "aux_state": aux_state
                }

        return action_combination_dict

              
    def generate_payoff_matrices(self, vehicles: Dict):
        fv = vehicles[VehicleRole.FV]
        zv = vehicles[VehicleRole.ZV]
        
        fv_actions = self.action_space.get_actions(VehicleRole.FV)
        zv_actions = self.action_space.get_actions(VehicleRole.ZV)
        
        P_fv = np.zeros((len(fv_actions), len(zv_actions)))
        P_zv = np.zeros((len(fv_actions), len(zv_actions)))
        
        # Todo: 获取不同动作的未来状态，并计算收益矩阵
        for i, fv_action in enumerate(fv_actions):
            for j, zv_action in enumerate(zv_actions):
                # 解包动作值
                _fv_action = list(fv_action.values())[0]
                fv_action_value = _fv_action.value 
                _zv_action = list(zv_action.values())[0]
                zv_action_value = _zv_action.value 

                action_key = (zv_action_value , fv_action_value)
                if action_key in self.action_set:
                    
                    # 将动作空间纳入到博弈框架中
                    fv_future_state = self.action_set[action_key]['aux_state']
                    zv_future_state = self.action_set[action_key]['ego_state']
                    
                    # 重新组织轨迹数据，利用重新组织的数据计算收益
                    fv_future_state = reorganize_trajectory_data(fv_future_state)
                    zv_future_state = reorganize_trajectory_data(zv_future_state)
                    
                    #! TODO: 需要修改收益的计算，按照博弈的方法
                    zv_payoff, fv_payoff = self.payoff_calculator.calculate(fv, zv, fv_future_state, zv_future_state)
                    P_fv[i, j] = fv_payoff
                    P_zv[i, j] = zv_payoff
                
        return P_fv, P_zv

    def stackelberg_equilibrium(self, P_zv , P_fv):
        fv_actions = self.action_space.get_actions(VehicleRole.FV)
        zv_actions = self.action_space.get_actions(VehicleRole.ZV)
        
        best_fv_strategy = np.zeros(len(fv_actions))
        best_zv_responses = np.zeros((len(fv_actions), len(zv_actions)))
        
        for i in range(len(fv_actions)):
            zv_qre = self._quantal_response(P_zv[i, :], self.beta_values[VehicleRole.ZV])
            best_zv_responses[i] = zv_qre
            best_fv_strategy[i] = np.sum(zv_qre * P_fv[i, :])
        
        fv_qre = self._quantal_response(best_fv_strategy, self.beta_values[VehicleRole.FV])
        expected_zv_strategy = np.zeros(len(zv_actions))
        for i in range(len(fv_actions)):
            expected_zv_strategy += fv_qre[i] * best_zv_responses[i]
        
        return {
            'fv_strategy': fv_qre,
            'zv_strategy': expected_zv_strategy,
            'P_fv': P_fv,
            'P_zv': P_zv
        }

    def _quantal_response(self, payoffs, beta):
        numerators = np.exp(payoffs / beta)
        denominator = np.sum(numerators)
        return numerators / denominator if denominator > 0 else np.ones_like(numerators) / len(numerators)

    def select_action(self, vehicle_role, strategy_probs):
        actions = self.action_space.get_actions(vehicle_role)
        action_idx = np.random.choice(len(actions), p=strategy_probs)
        selected_action = actions[action_idx].copy()
        if isinstance(selected_action['value'], list):
            selected_action['value'] = np.random.choice(selected_action['value'])
        return selected_action
    
    def print_actions_with_probabilities(self, vehicle_role, strategy_probs):
        actions = self.action_space.get_actions(vehicle_role)
        
        print(f"\n{vehicle_role}角色可能的动作及概率:")
        for i, (action, prob) in enumerate(zip(actions, strategy_probs)):
            action_type = action['type'].name if hasattr(action['type'], 'name') else action['type']
            action_value = action['value']
            print(f"动作 {i+1}: 类型={action_type}, 值={action_value}, 概率={prob:.4f}")      