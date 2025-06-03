
import numpy as np
from LM_env.interaction_model.game_decision import *
from LM_env.interaction_model.hunman_predictor import predict_trajectories , plot_trajectories

def get_roi_vehicles(vehicle_state, reference_line):
    x_ego, y_ego = vehicle_state['position']
    neighbors = vehicle_state.get('neighbors', [])
    ref_x = reference_line[:, 0]
    ref_y = reference_line[:, 1]
    
    roi_categories = {
        '环境自车':[],
        '主道前车': [],
        '主道后车': [],
        '辅道车辆': []
    }
    min_front_distance = float('inf')
    min_rear_distance = float('inf')
    closest_front_vehicle = None
    closest_rear_vehicle = None
    for neighbor in neighbors:
        x_neighbor = neighbor['position'][0]
        y_neighbor = neighbor['position'][1]
        relative_position = np.array([x_neighbor - x_ego, y_neighbor - y_ego])
        idx = np.argmin(np.abs(ref_x - x_neighbor))
        ref_y_at_x = ref_y[idx]
        if y_neighbor > ref_y_at_x + 1:
            roi_categories['辅道车辆'].append(neighbor)
        else:
            distance = abs(relative_position[0])
            if x_neighbor < x_ego:
                if distance < min_front_distance:
                    min_front_distance = distance
                    closest_front_vehicle = neighbor
            else:
                if distance < min_rear_distance:
                    min_rear_distance = distance
                    closest_rear_vehicle = neighbor
    if closest_front_vehicle:
        roi_categories['主道前车'].append(closest_front_vehicle)
    if closest_rear_vehicle:
        roi_categories['主道后车'].append(closest_rear_vehicle)
    if 'neighbors' in vehicle_state:
        vehicle_state.pop('neighbors')
    roi_categories['环境自车'].append(vehicle_state)
    return roi_categories  
   

        
# 建立主车的博弈决策模型
def game_theroy(Roi_categories , predict_state , params):
    

    game_model = GameModel(params, predict_state)
    
    vehicles = {
        VehicleRole.FV: Roi_categories['辅道车辆'][0],
        VehicleRole.ZV: Roi_categories['环境自车'][0]
    }
    

    P_FV, P_main = game_model.generate_payoff_matrices(vehicles)
      
    equilibrium = game_model.stackelberg_equilibrium(P_main, P_FV)
    
    return equilibrium



# TODO:先用简单的距离判断，后续可以引入交通流量、航向角和纵向距离等修正系数
# TODO：辅道车只能被一辆车选为跟车对象
def is_game_with_aux(Roi_categories):
    """
    判断是否参与博弈
    """
    ego_vehicles = Roi_categories['环境自车']
    aux_vehicles = Roi_categories['辅道车辆']
    dis_threshold = 5
    for  aux_vehicle in aux_vehicles:
        if abs(aux_vehicle['relative_position'][0]) < dis_threshold and aux_vehicle['relative_position'][0] < 0:
            return True
        
    return False

# TODO：这里需要体现选取的随机性，且策略变化不能过大
def select_action_by_probability(prob_distribution):
    """
    根据给定的概率分布选择一个动作。
    
    """
    # 确保概率分布是numpy数组
    prob_dist = np.array(prob_distribution)
    
    # 确保概率之和为1
    if not np.isclose(np.sum(prob_dist), 1.0):
        prob_dist = prob_dist / np.sum(prob_dist)
    
    # 使用numpy的random.choice根据概率选择动作
    action_index = np.random.choice(len(prob_dist), p=prob_dist)
    
    return action_index    

def select_max_probability_action(prob_distribution):
    """
    选择概率分布中概率最大的动作。
    
    参数:
        prob_distribution: 列表或numpy数组，表示动作的概率分布
        
    返回:
        action_index: 概率最大的动作的索引
    """
    # 确保概率分布是numpy数组
    prob_dist = np.array(prob_distribution)
    
    # 返回概率最大的动作索引
    action_index = np.argmax(prob_dist)
    
    return action_index  
    
    
# 简单的采样预测（这是人的预测，要靠近人的机理去预测）
def predict_tra(Roi_categories , reference_line , aux_reference_line):
    """
    预测车辆状态
    
    参数:
        Roi_categories: 关注区域车辆
    
    返回:
        predict_state: 预测的目标车辆状态
    """
    # 初始化车辆状态
    ego_vehicle = Roi_categories['环境自车'][0]
    aux_vehicles = Roi_categories['辅道车辆']
    leader_vehicle = Roi_categories['主道前车']

    predictions = 0
    # 预测的轨迹能够纳入博弈的动作空间
    predictions = predict_trajectories(
        ego_vehicle=ego_vehicle,
        main_front_vehicle=leader_vehicle,
        auxiliary_vehicles=aux_vehicles,
        reference_path=reference_line,
        auxiliary_reference_path=aux_reference_line[::-1],
    )
    
    # # 可视化预测轨迹
    # plot_trajectories( ego_vehicle=ego_vehicle,
    #                     main_front_vehicle=leader_vehicle,
    #                     auxiliary_vehicles=aux_vehicles,
    #                     reference_path=reference_line,
    #                     aux_reference_path=aux_reference_line[::-1],
    #                     predicted_trajectories=predictions
    # )
       
    # print('ego_vehicle:', predictions)
    
    return predictions
    
    
    

    
    
    
    



    
    
    
       