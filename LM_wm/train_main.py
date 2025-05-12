

import argparse
import os
from scripts.generate_data import generate_training_data
from scripts.train import train_model

# 尝试导入预测函数 - 处理多种导入路径情况
try:
    from scripts.predict import predict_main_func
except ImportError:
    try:
        from LM_wm.scripts.predict import predict_main_func
    except ImportError:
        print("警告: 无法导入预测模块，预测功能将不可用")
        predict_main_func = None

# 运行命令示例:
# 生成数据: python LM_wm/train_main.py --generate_data 
# 训练模型: python LM_wm/train_main.py --train --mode image
# 预测:    python LM_wm/train_main.py --predict --sample_idx 5
# 全流程:  python LM_wm/train_main.py --all --mode image
def parse_args():
    parser = argparse.ArgumentParser(description='BEV轨迹预测模型 - 集成工具')
    
    # 主要功能选择
    parser.add_argument('--generate_data', action='store_true', help='生成训练数据')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--predict', action='store_true', help='使用训练好的模型进行预测')
    parser.add_argument('--all', action='store_true', help='执行完整流程（生成数据并训练）')
    
    # 训练相关参数
    parser.add_argument('--mode', type=str, default='image', choices=['feature', 'image'],
                      help='训练模式: feature (特征预测) 或 image (图像生成)')
    
    # 预测相关参数
    parser.add_argument('--checkpoint', type=str, default=None, 
                      help='模型检查点路径，默认使用最佳模型')
    parser.add_argument('--data_dir', type=str, default=None,
                      help='测试数据目录，默认使用配置中的数据目录')
    parser.add_argument('--output_dir', type=str, default="LM_wm/predictions",
                      help='预测结果输出目录')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='预测批次大小')
    parser.add_argument('--sample_idx', type=int, default=None,
                      help='指定预测单个样本的索引，如果不指定则进行批量预测')
    
    return parser.parse_args()

def main():
    """
    主函数，集成数据生成、模型训练和预测功能
    """
    args = parse_args()
    
    # 显示标题
    print("\n" + "="*60)
    print("BEV轨迹预测模型 - 集成工具")
    print("="*60 + "\n")
    
    # 生成数据
    if args.generate_data or args.all:
        print("\n" + "-"*30)
        print("步骤 1: 生成训练数据")
        print("-"*30)
        generate_training_data()
        print("数据生成完成！")
    
    # 训练模型
    if args.train or args.all:
        print("\n" + "-"*30)
        print(f"步骤 2: 训练模型 (模式: {args.mode})")
        print("-"*30)
        train_model(mode=args.mode)
        print(f"模型训练完成！模式: {args.mode}")
    
    # 预测
    if args.predict or (args.all and predict_main_func is not None):
        print("\n" + "-"*30)
        print("步骤 3: 模型预测")
        print("-"*30)
        
        if predict_main_func is None:
            print("错误: 无法导入预测模块。请确保 'scripts/predict.py' 文件存在并正确配置。")
        else:
            # 预测模式
            checkpoint_path = args.checkpoint
            data_dir = args.data_dir
            output_dir = args.output_dir
            
            # 创建输出目录（如果不存在）
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 运行预测
            predict_main_func(
                checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                output_dir=output_dir,
                batch_size=args.batch_size,
                sample_idx=args.sample_idx
            )
            
            print(f"预测完成！结果保存在: {output_dir}")
    
    # 如果没有指定任何操作，显示帮助信息
    if not (args.generate_data or args.train or args.predict or args.all):
        print("请指定要执行的操作:")
        print("  --generate_data : 生成训练数据")
        print("  --train         : 训练模型")
        print("  --predict       : 使用训练好的模型进行预测")
        print("  --all           : 执行完整流程（生成数据并训练）")
        print("\n使用 --help 查看更多选项")
    
    print("\n" + "="*60)
    print("处理完成!")
    print("="*60)

if __name__ == "__main__":
    main() 