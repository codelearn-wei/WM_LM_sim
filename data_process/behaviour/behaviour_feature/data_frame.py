
from get_all_frame_data import load_frame_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns





if __name__ == "__main__":
    
    
    data_path = r"data_process\behaviour\data\all_frame_data.pkl"
    frame_data = load_frame_data(data_path)
    

    
    print("数据读取与绘图完成!")