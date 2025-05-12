import numpy as np
from typing import Dict
from data_process.behaviour.game_behaviour_model.vehicle import Vehicle
from data_process.behaviour.game_behaviour_model.game.game_action import *
from data_process.behaviour.game_behaviour_model.game.game_pay_off import *
from data_process.behaviour.game_behaviour_model.IDM import *
from typing import Dict

class GameModel:
    def __init__(self, params: Dict , predict_state: Dict , pay_off_info: Dict):
        self.params = params
        self.future_state = predict_state
        self.pay_off_info = pay_off_info
        self.beta_values = {
            VehicleRole.FV: params.get('beta_fv', 0.01),
            VehicleRole.ZV: params.get('beta_zv', 0.01)
        }
        self.action_space = ActionSpace(params)
        self.payoff_calculator = PayoffCalculator(params)
        
    
    def generate_payoff_matrices(self, vehicles: Dict):
        fv = vehicles[VehicleRole.FV]
        zv = vehicles[VehicleRole.ZV]
        
        fv_actions = self.action_space.get_actions(VehicleRole.FV)
        zv_actions = self.action_space.get_actions(VehicleRole.ZV)
        
        P_fv = np.zeros((len(fv_actions), len(zv_actions)))
        P_zv = np.zeros((len(fv_actions), len(zv_actions)))
        
     
        for i, fv_action in enumerate(fv_actions):
            for j, zv_action in enumerate(zv_actions):
                # 解包动作值
                _fv_action = list(fv_action.values())[0]
                fv_action_value = _fv_action.value 
                _zv_action = list(zv_action.values())[0]
                zv_action_value = _zv_action.value 
                action_key = (zv_action_value , fv_action_value)
                if action_key in self.future_state:
                    
                    fv_future_state = self.future_state[action_key]['fv']
                    zv_future_state = self.future_state[action_key]['zv']
                    
                    # TODO:设计博弈收益函数，建模主道车辆决策问题
                    fv_payoff, zv_payoff = self.payoff_calculator.calculate(fv, zv, fv_future_state, zv_future_state,self.pay_off_info)
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


