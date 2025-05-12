import numpy as np

class TrajectoryFeatures:
    def __init__(self, human_traj_data, generated_traj_data):
        self.human_traj_data = human_traj_data
        self.generated_traj_data = generated_traj_data
        self.features = []
        self.FDEs = []
        self.IRL_features = []  
        self.feature_num = 0

    def compute_features(self):
        speed_threshold = 20  
        for trajectory in self.generated_traj_data:
            x = trajectory[:, 0]
            v = trajectory[:, 1]
            a = trajectory[:, 2]
            jerk = np.gradient(a)  

            feature_list = [
                np.max(x), np.min(x), np.max(v), np.min(v),
                np.max(a), np.min(a) 
            ]
            self.feature_num = len(feature_list)
            self.features.append(feature_list)

    def get_features(self):
        return self.features

    def get_feature_num(self):
        return self.feature_num

    def calculate_human_likeness(self) -> list:
        """
        计算所有生成轨迹与人类轨迹的FDE。
        """
        self.FDEs = []
        for generated_traj in self.generated_traj_data:
            if self.human_traj_data.shape[0] != generated_traj.shape[0]:
                raise ValueError("轨迹长度必须相同。")
            FDE = np.linalg.norm(self.human_traj_data[-1] - generated_traj[-1])
            self.FDEs.append(FDE)
        return self.FDEs

    def get_IRL_features(self):
        self.compute_features()
        self.calculate_human_likeness()
        for feature, FDE in zip(self.features, self.FDEs):
            self.IRL_features.append([feature, FDE])
        return self.IRL_features

    def calculate_human_features(self, trajectory):


        # 提取每一列的数据
        x = trajectory[:, 0]
        v = trajectory[:, 1]
        a = trajectory[:, 2]
        jerk = np.gradient(a)  # 计算冲击

        # 计算统计特征
        feature_list = [
            np.max(x), np.min(x), np.max(v), np.min(v),
            np.max(a), np.min(a) # 计算速度超过阈值的时间比例
        ]
        return feature_list