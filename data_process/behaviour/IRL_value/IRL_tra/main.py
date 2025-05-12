# IRL可以捕捉价值取向和社会偏好

# 优先实现用一条人类轨迹，来捕捉换道的人类价值观

# ! IRL捕捉人类决策价值，需要设定环境计算采样轨迹的特征和真实人类数据特征做对比分析，因此在引入前还需做以下工作
# TODO : 1. 结合数据集，确定采样轨迹应该如何生成；
# TODO : 2. 需要在博弈和强化学习模型中，实现基于聚类的IRL轨迹选择。————先聚类，基于人的数据，价值参数辨识（参考一些社会价值取向的模型）————SVO的文章

import torch
import numpy as np
from reward import LinearRewardFunction,NeuralNetworkRewardFunction
from feature_cal import TrajectoryFeatures
from data_processes import extract_features
from IRL import InverseReinforcementLearning

# TODO: 建模并选择匝道汇流轨迹

def main():    
    
    gauss_data = GaussData('./guass_progress_data/sample_trajectory.mat')
    trajectories = gauss_data.compute_trajectory_data()
    trajectories = trajectories[0:10,:,:]

    # 选择一个人类驾驶轨迹，这是需要去捕捉的驾驶价值观
    person_trajectory = trajectories[6]

    # 计算人类驾驶轨迹和采样或者预测规则的特征
    features_calculator = TrajectoryFeatures(person_trajectory, trajectories)
    features_IRL = features_calculator.get_IRL_features()
    human_feature = features_calculator.calculate_human_features(person_trajectory)
    feature_num = features_calculator.feature_num

    theta_initial = np.random.rand(feature_num)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    linear_reward = LinearRewardFunction(theta_initial, device=device)
     
    # nn_reward = NeuralNetworkRewardFunction(input_dim=feature_num, device=device)

    # 创建并运行逆强化学习实例
    irl = InverseReinforcementLearning(features_IRL, human_feature, linear_reward)
    final_theta = irl.run()

    # 输出结果
    print("Final theta from IRL:", final_theta)

if __name__ == "__main__":
    main()