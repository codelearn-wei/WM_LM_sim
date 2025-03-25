import argparse
from scripts.generate_data import generate_training_data
from scripts.train import train_model

def parse_args():
    parser = argparse.ArgumentParser(description='LM_wm 训练流程')
    parser.add_argument('--generate_data', action='store_true', help='生成训练数据')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--all', action='store_true', help='执行完整流程（生成数据并训练）')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.generate_data or args.all:
        generate_training_data()
    
    if args.train or args.all:
        train_model()

if __name__ == "__main__":
    main() 