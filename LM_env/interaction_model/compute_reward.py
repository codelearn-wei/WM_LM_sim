import math
from LM_env.interaction_model.env_config.env_config import TrajectoryConfig

def parse_trajectory_simple(traj):
    x = [p[0] for p in traj]
    y = [p[1] for p in traj]
    vel = [p[2] for p in traj]
    acc = [p[4] for p in traj]
    heading = [p[3] for p in traj]
  
    return x, y, acc, vel

def is_merging_trajectory(x, y):
    if not y:
        return False
    min_y = min(y)
    return min_y < TrajectoryConfig.Y_MERGE_THRESHOLD

def is_lane_changing_trajectory(y):
    if len(y) < 3:
        return False
    
    y_changes = [abs(y[i] - y[-1]) for i in range(1, len(y))]
    total_y_change = sum(y_changes)
    max_y_change = max(y_changes) if y_changes else 0
    
    return  max_y_change > 1.0

def detect_lane_overlap(y1, y2, i):
    """检测两车在时刻i是否在同一车道或存在横向重叠风险"""
    config = TrajectoryConfig
    
    y_diff = abs(y1[i] - y2[i])
    
    if y_diff <= config.VEHICLE_RADIUS * 2:
        return True
    elif y_diff <= config.COLLISION_ZONE_THRESHOLD:
        return True
    
    return False

def calculate_collision_risk(x1, y1, x2, y2, i):
    """计算特定时刻的碰撞风险"""
    config = TrajectoryConfig
    
    if i >= len(x1) or i >= len(x2):
        return 0.0
    
    x_diff = abs(x1[i] - x2[i])
    y_diff = abs(y1[i] - y2[i])
    
    if not detect_lane_overlap(y1, y2, i):
        return 0.0
    
    if x_diff <= config.VEHICLE_RADIUS * 2 and y_diff <= config.VEHICLE_RADIUS * 2:
        return 1.0
    
    distance = math.sqrt(x_diff**2 + y_diff**2)
    safe_distance = config.MIN_SAFE_DISTANCE
    
    if distance < safe_distance:
        risk = 1.0 - (distance / safe_distance)
        return max(0.0, min(1.0, risk))
    
    return 0.0

def advanced_safety_score(x_traj, y_traj, acc, vel, other_x=None, other_y=None, other_vel=None):
    """改进的安全性评分，重点关注换道碰撞风险"""
    config = TrajectoryConfig
    
    acc_violations = sum(1 for a in acc if abs(a) > config.SAFE_ACC_LIMIT)
    acc_safety = 1.0 - (acc_violations / len(acc)) if acc else 1.0
    
    interaction_safety = 1.0
    if other_x and other_y:
        collision_risks = []
        
        for i in range(min(len(x_traj), len(other_x))):
            risk = calculate_collision_risk(x_traj, y_traj, other_x, other_y, i)
            collision_risks.append(risk)
        
        if collision_risks:
            max_risk = max(collision_risks)
            interaction_safety = 1.0 - max_risk
    
    total_safety = min(acc_safety, interaction_safety)
    return max(0.0, total_safety)

def comfort_score_simple(acc):
    if not acc:
        return 1.0
    acc_avg = sum(abs(a) for a in acc) / len(acc)
    return max(0.0, 1.0 - acc_avg / TrajectoryConfig.SAFE_ACC_LIMIT)

def efficiency_score_simple(vel, target_speed):
    if not vel:
        return 1.0
    avg_speed = sum(vel) / len(vel)
    return max(0.0, 1.0 - abs(avg_speed - target_speed) / target_speed)

def calculate_merge_pressure(x_traj, y_traj):
    config = TrajectoryConfig
    if not x_traj:
        return 0.0
    
    min_merge_dist = min(abs(x - config.MERGE_POINT_X) for x in x_traj)
    
    if min_merge_dist > config.MERGE_PRESSURE_DISTANCE: 
        return 0.0
    
    is_merging = is_merging_trajectory(x_traj, y_traj)
    
    if is_merging:
        base_pressure = 1.0 - (min_merge_dist / config.MERGE_PRESSURE_DISTANCE)
        return max(0.0, base_pressure * 0)
    else:
        base_pressure = 1.0 - (min_merge_dist / config.MERGE_PRESSURE_DISTANCE)
        return max(0.0, base_pressure)



def calculate_main_rewards(params, main_traj, aux_traj=None):
    config = TrajectoryConfig
    
    x, y, acc, vel = parse_trajectory_simple(main_traj)
    
    other_x, other_y, other_vel = None, None, None
    if aux_traj:
        other_x, other_y, _, other_vel = parse_trajectory_simple(aux_traj)
    
    # 主道车自身收益
    comfort = comfort_score_simple(acc)
    efficiency = efficiency_score_simple(vel, config.MAIN_TARGET_SPEED)
    safety = advanced_safety_score(x, y, acc, vel, other_x, other_y, other_vel)
    
    # 利己部分
    self_reward = (params['main_comfort_weight'] * comfort + 
                   params['main_efficiency_weight'] * efficiency + 
                   params['main_safety_weight'] * -safety)
    
    # 利他部分：考虑辅道车的收益
    altruistic_reward = 0.0
    if aux_traj:
        aux_reward = calculate_aux_rewards(params, aux_traj, main_traj)
        altruistic_reward = params.get('main_altruistic_weight', 0.1) * aux_reward
    
    total_reward = self_reward + altruistic_reward
    return total_reward

def calculate_aux_rewards(params, aux_traj, main_traj=None):
    config = TrajectoryConfig
    
    x, y, acc, vel = parse_trajectory_simple(aux_traj)
    
    other_x, other_y, other_vel = None, None, None
    if main_traj:
        other_x, other_y, _, other_vel = parse_trajectory_simple(main_traj)
    
    # 辅道车自身收益
    comfort = comfort_score_simple(acc)
    efficiency = efficiency_score_simple(vel, config.AUX_TARGET_SPEED)
    safety = advanced_safety_score(x, y, acc, vel, other_x, other_y, other_vel)
    merge_pressure = calculate_merge_pressure(x, y)
    
    # 利己部分
    self_reward = (params['aux_comfort_weight'] * comfort + 
                   params['aux_efficiency_weight'] * efficiency + 
                   params['aux_safety_weight'] * -safety +
                   params['aux_pressure_weight'] * merge_pressure)
    
    # 利他部分：考虑主道车的收益
    altruistic_reward = 0.0
    if main_traj:
        main_reward = calculate_main_rewards_self_only(params, main_traj, aux_traj)
        altruistic_reward = params.get('aux_altruistic_weight', 0.1) * main_reward
    
    total_reward = self_reward + altruistic_reward
    return total_reward

def calculate_main_rewards_self_only(params, aux_traj, main_traj=None):
    """计算辅道车的纯自身收益（避免循环调用）"""
    config = TrajectoryConfig
    
    x, y, acc, vel = parse_trajectory_simple(main_traj)
    
    other_x, other_y, other_vel = None, None, None
    if aux_traj:
        other_x, other_y, _, other_vel = parse_trajectory_simple(aux_traj)
    
    # 主道车自身收益
    comfort = comfort_score_simple(acc)
    efficiency = efficiency_score_simple(vel, config.MAIN_TARGET_SPEED)
    safety = advanced_safety_score(x, y, acc, vel, other_x, other_y, other_vel)
    
    # 利己部分
    return (params['main_comfort_weight'] * comfort + 
                   params['main_efficiency_weight'] * efficiency + 
                   params['main_safety_weight'] * -safety)

def calculate_aux_rewards_self_only(params, aux_traj, main_traj=None):
    """计算辅道车的纯自身收益（避免循环调用）"""
    config = TrajectoryConfig
    
    x, y, acc, vel = parse_trajectory_simple(aux_traj)
    
    other_x, other_y, other_vel = None, None, None
    if main_traj:
        other_x, other_y, _, other_vel = parse_trajectory_simple(main_traj)
    
    comfort = comfort_score_simple(acc)
    efficiency = efficiency_score_simple(vel, config.AUX_TARGET_SPEED)
    safety = advanced_safety_score(x, y, acc, vel, other_x, other_y, other_vel)
    merge_pressure = calculate_merge_pressure(x, y)
    
    return (params['aux_comfort_weight'] * comfort + 
            params['aux_efficiency_weight'] * efficiency + 
            params['aux_safety_weight'] * -safety +
            params['aux_pressure_weight'] * merge_pressure)

def detailed_collision_analysis(traj1, traj2):
    """详细的碰撞风险分析"""
    x1, y1, _, vel1 = parse_trajectory_simple(traj1)
    x2, y2, _, vel2 = parse_trajectory_simple(traj2)
    
    analysis = {
        'max_collision_risk': 0.0,
        'collision_moments': [],
        'lane_overlap_moments': []
    }
    
    for i in range(min(len(x1), len(x2))):
        risk = calculate_collision_risk(x1, y1, x2, y2, i)
        overlap = detect_lane_overlap(y1, y2, i)
        
        if risk > analysis['max_collision_risk']:
            analysis['max_collision_risk'] = risk
        
        if risk > 0.3:
            analysis['collision_moments'].append({
                'time_step': i,
                'risk': risk,
                'x1': x1[i], 'y1': y1[i],
                'x2': x2[i], 'y2': y2[i]
            })
        
        if overlap:
            analysis['lane_overlap_moments'].append(i)
    
    return analysis

if __name__ == "__main__":
    main_trajectory = [
        [1112.9539227169691, 966.4079784270298, 0.10015999999999994, -2.9730448582316615, 0.0],
        [1112.9248134040224, 966.4031004222314, 0.29515199999999997, -2.9755599324494013, 1.94992],
        [1112.8774328979823, 966.39517101854, 0.4803944, -2.9757735184798344, 1.852424],
        [1112.8126939221513, 966.3843477928748, 0.65637468, -2.9759420986801732, 1.7598028000000001],
        [1112.731457545883, 966.3708165278676, 0.823555946, -2.976541363682337, 1.67181266]
    ]
    
    aux_straight_trajectory = [
        [1100.0, 980.0, 0.1, -1.5, 0.0],
        [1105.0, 980.2, 0.15, -1.2, 1.0],
        [1110.0, 980.1, 0.2, -1.0, 2.0],
        [1115.0, 980.3, 0.25, -0.8, 3.0],
        [1120.0, 980.0, 0.3, -0.5, 4.0]
    ]
    
    aux_lane_change_trajectory = [
        [1100.0, 980.0, 0.1, -1.5, 0.0],
        [1105.0, 975.0, 0.15, -1.2, 1.0],
        [1110.0, 970.0, 0.2, -1.0, 2.0],
        [1115.0, 967.0, 0.25, -0.8, 3.0],
        [1120.0, 966.5, 0.3, -0.5, 4.0]
    ]
    
    aux_dangerous_trajectory = [
        [1110.0, 968.0, 0.1, -2.0, 0.0],
        [1111.5, 967.5, 0.15, -1.8, 1.0],
        [1112.8, 967.0, 0.2, -1.5, 2.0],
        [1113.5, 966.8, 0.25, -1.2, 3.0],
        [1114.0, 966.6, 0.3, -1.0, 4.0]
    ]
    
    # 测试参数 - 包含利他权重
    test_params = {
        # 主道车参数
        'main_comfort_weight': 0.3,
        'main_efficiency_weight': 0.3,
        'main_safety_weight': 0.3,
        'main_altruistic_weight': 0.1,  # 主道车对辅道车的利他权重
        
        # 辅道车参数
        'aux_comfort_weight': 0.2,
        'aux_efficiency_weight': 0.3,
        'aux_safety_weight': 0.3,
        'aux_pressure_weight': 0.2,
        'aux_altruistic_weight': 0.15,  # 辅道车对主道车的利他权重
    }
    
    print("=== 博弈论收益计算测试（利己+利他行为） ===")
    
    scenarios = [
        ("直行轨迹", aux_straight_trajectory),
        ("换道轨迹", aux_lane_change_trajectory),
        ("危险接近轨迹", aux_dangerous_trajectory)
    ]
    
    for name, aux_traj in scenarios:
        print(f"\n--- {name} ---")
        
        main_reward = calculate_main_rewards(test_params, main_trajectory, aux_traj)
        aux_reward = calculate_aux_rewards(test_params, aux_traj, main_trajectory)
        
        # 分别计算利他部分
        main_self_reward = calculate_main_rewards_self_only(test_params, main_trajectory, aux_traj)
        aux_self_reward = calculate_aux_rewards_self_only(test_params, aux_traj, main_trajectory)
        aux_altruistic_part = calculate_aux_rewards(test_params, aux_traj, main_trajectory) - aux_self_reward
        main_altruistic_part = main_reward - main_self_reward
        
        print(f"主道车总收益: {main_reward:.4f}")
        print(f"  - 主道车利他部分: {main_altruistic_part:.4f}")
        print(f"辅道车总收益: {aux_reward:.4f}")
        print(f"  - 辅道车利他部分: {aux_altruistic_part:.4f}")
        
        collision_analysis = detailed_collision_analysis(main_trajectory, aux_traj)
        print(f"最大碰撞风险: {collision_analysis['max_collision_risk']:.4f}")
        
        _, y_aux, _, _ = parse_trajectory_simple(aux_traj)
        is_changing = is_lane_changing_trajectory(y_aux)
        print(f"是否换道: {is_changing}")
        
    print("\n=== 利他权重影响测试 ===")
    # 测试不同利他权重的影响
    altruistic_weights = [0.0, 0.1, 0.2, 0.3]
    test_scenario = aux_lane_change_trajectory
    
    for weight in altruistic_weights:
        params_test = test_params.copy()
        params_test['main_altruistic_weight'] = weight
        params_test['aux_altruistic_weight'] = weight
        
        main_reward = calculate_main_rewards(params_test, main_trajectory, test_scenario)
        aux_reward = calculate_aux_rewards(params_test, test_scenario, main_trajectory)
        
        print(f"利他权重 {weight}: 主道车收益={main_reward:.4f}, 辅道车收益={aux_reward:.4f}")