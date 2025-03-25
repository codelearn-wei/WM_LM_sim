import argparse
from scripts.generate_data import generate_training_data
from scripts.train import train_model

# 运行命令为 python LM_wm/main.py --train --mode feature
# 运行命令为 python LM_wm/main.py --generate_data --mode feature
# 运行命令为 python LM_wm/main.py --train --mode image
def parse_args():
    parser = argparse.ArgumentParser(description='BEV Prediction Training')
    parser.add_argument('--generate_data', action='store_true', help='生成训练数据')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--all', action='store_true', help='执行完整流程（生成数据并训练）')
    parser.add_argument('--mode', type=str, default='feature', choices=['feature', 'image'],
                      help='训练模式: feature prediction 或 image generation')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 生成数据
    if args.generate_data or args.all:
        print("开始生成训练数据...")
        generate_training_data()
        print("数据生成完成！")
    
    # 训练模型
    if args.train or args.all:
        print(f"开始训练模型，模式: {args.mode}")
        train_model(mode=args.mode)
        print("训练完成！")

if __name__ == "__main__":
    main() 