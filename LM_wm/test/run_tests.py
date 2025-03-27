#!/usr/bin/env python3
"""
区域权重与掩码生成测试脚本
用法:
    python run_tests.py [选项]

选项:
    --all          运行所有测试
    --region       仅运行区域检测测试
    --loss         仅运行加权损失函数测试
    --integration  仅运行掩码和损失集成测试
    --help         显示帮助信息
"""

import os
import sys
import importlib
from pathlib import Path
import argparse

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="区域权重与掩码生成测试")
    
    # 添加命令行选项
    parser.add_argument('--all', action='store_true', help='运行所有测试')
    parser.add_argument('--region', action='store_true', help='仅运行区域检测测试')
    parser.add_argument('--loss', action='store_true', help='仅运行加权损失函数测试')
    parser.add_argument('--integration', action='store_true', help='仅运行掩码和损失集成测试')
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果未提供任何选项，则默认运行所有测试
    if not (args.all or args.region or args.loss or args.integration):
        args.all = True
        
    return args

def main():
    """主函数，根据命令行参数运行测试"""
    args = parse_args()
    
    # 确保结果目录存在
    os.makedirs('LM_wm/test/results', exist_ok=True)
    
    # 如果运行所有测试，则执行综合测试脚本
    if args.all:
        print("运行所有测试...")
        all_tests = importlib.import_module('LM_wm.test.run_all_tests')
        all_tests.run_all_tests()
        return
    
    # 否则，根据命令行选项运行特定测试
    if args.region:
        print("运行区域检测测试...")
        region_detection = importlib.import_module('LM_wm.test.test_region_detection')
        region_detection.test_region_detection()
        
    if args.loss:
        print("运行加权损失函数测试...")
        weighted_loss = importlib.import_module('LM_wm.test.test_weighted_loss')
        weighted_loss.test_weighted_loss()
        
    if args.integration:
        print("运行掩码和损失集成测试...")
        mask_loss = importlib.import_module('LM_wm.test.test_mask_and_loss')
        mask_loss.main()
    
    print("测试完成。结果保存在 LM_wm/test/results 目录。")

if __name__ == "__main__":
    main() 