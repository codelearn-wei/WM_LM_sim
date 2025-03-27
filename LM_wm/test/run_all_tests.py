#!/usr/bin/env python3
"""
综合测试脚本 - 运行所有测试项目并生成报告
"""

import os
import sys
import time
import importlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

def run_all_tests():
    """运行所有测试并生成综合报告"""
    print("=" * 60)
    print("开始执行区域权重与掩码生成综合测试")
    print("=" * 60)
    
    # 创建结果目录
    results_dir = Path('LM_wm/test/results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 准备报告数据
    report_data = {
        "测试项目": [],
        "执行结果": [],
        "耗时(秒)": [],
        "相关指标": []
    }
    
    # 测试1: 区域检测测试
    print("\n>>> 执行区域检测测试")
    try:
        start_time = time.time()
        region_detection = importlib.import_module('LM_wm.test.test_region_detection')
        boundary_iou, vehicle_iou, road_iou = region_detection.test_region_detection()
        
        report_data["测试项目"].append("区域检测")
        report_data["执行结果"].append("通过")
        report_data["耗时(秒)"].append(time.time() - start_time)
        report_data["相关指标"].append(f"边界IoU={boundary_iou:.4f}, 车辆IoU={vehicle_iou:.4f}, 道路IoU={road_iou:.4f}")
        print(f"区域检测测试完成，用时 {report_data['耗时(秒)'][-1]:.2f} 秒")
    except Exception as e:
        report_data["测试项目"].append("区域检测")
        report_data["执行结果"].append("失败")
        report_data["耗时(秒)"].append(time.time() - start_time)
        report_data["相关指标"].append(f"错误: {str(e)}")
        print(f"区域检测测试失败: {e}")
    
    # 测试2: 加权损失函数测试
    print("\n>>> 执行加权损失函数测试")
    try:
        start_time = time.time()
        weighted_loss = importlib.import_module('LM_wm.test.test_weighted_loss')
        weighted_loss.test_weighted_loss()
        
        report_data["测试项目"].append("加权损失函数")
        report_data["执行结果"].append("通过")
        report_data["耗时(秒)"].append(time.time() - start_time)
        report_data["相关指标"].append("详见图表输出")
        print(f"加权损失函数测试完成，用时 {report_data['耗时(秒)'][-1]:.2f} 秒")
    except Exception as e:
        report_data["测试项目"].append("加权损失函数")
        report_data["执行结果"].append("失败")
        report_data["耗时(秒)"].append(time.time() - start_time)
        report_data["相关指标"].append(f"错误: {str(e)}")
        print(f"加权损失函数测试失败: {e}")
    
    # 测试3: 现有的掩码与损失测试
    print("\n>>> 执行掩码与损失集成测试")
    try:
        start_time = time.time()
        mask_loss = importlib.import_module('LM_wm.test.test_mask_and_loss')
        mask_loss.main()
        
        report_data["测试项目"].append("掩码与损失集成")
        report_data["执行结果"].append("通过")
        report_data["耗时(秒)"].append(time.time() - start_time)
        report_data["相关指标"].append("详见测试输出")
        print(f"掩码与损失集成测试完成，用时 {report_data['耗时(秒)'][-1]:.2f} 秒")
    except Exception as e:
        report_data["测试项目"].append("掩码与损失集成")
        report_data["执行结果"].append("失败")
        report_data["耗时(秒)"].append(time.time() - start_time)
        report_data["相关指标"].append(f"错误: {str(e)}")
        print(f"掩码与损失集成测试失败: {e}")
    
    # 生成测试报告
    generate_test_report(report_data)
    
    print("\n" + "=" * 60)
    print(f"所有测试已完成，报告已保存到 {results_dir / 'test_report.html'}")
    print("=" * 60)

def generate_test_report(report_data):
    """生成HTML测试报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 创建HTML报告
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>区域权重与掩码生成测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333366; }}
        h2 {{ color: #666699; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        .summary {{ margin: 20px 0; padding: 10px; background-color: #f8f8f8; border-left: 5px solid #666699; }}
        .image-gallery {{ display: flex; flex-wrap: wrap; margin-top: 20px; }}
        .image-container {{ margin: 10px; text-align: center; }}
        .image-container img {{ max-width: 500px; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <h1>区域权重与掩码生成测试报告</h1>
    <p>生成时间: {now}</p>
    
    <div class="summary">
        <h2>测试摘要</h2>
        <p>总测试项目: {len(report_data["测试项目"])}</p>
        <p>通过项目: {report_data["执行结果"].count("通过")}</p>
        <p>失败项目: {report_data["执行结果"].count("失败")}</p>
        <p>总耗时: {sum(report_data["耗时(秒)"]):.2f} 秒</p>
    </div>
    
    <h2>测试详情</h2>
    <table>
        <tr>
            <th>测试项目</th>
            <th>执行结果</th>
            <th>耗时(秒)</th>
            <th>相关指标</th>
        </tr>
"""

    # 添加测试结果行
    for i in range(len(report_data["测试项目"])):
        result_class = "pass" if report_data["执行结果"][i] == "通过" else "fail"
        html_content += f"""
        <tr>
            <td>{report_data["测试项目"][i]}</td>
            <td class="{result_class}">{report_data["执行结果"][i]}</td>
            <td>{report_data["耗时(秒)"][i]:.2f}</td>
            <td>{report_data["相关指标"][i]}</td>
        </tr>"""

    # 添加测试图片
    html_content += """
    </table>
    
    <h2>测试结果可视化</h2>
    <div class="image-gallery">
"""

    # 列出并包含结果目录中的所有图像
    results_dir = Path('LM_wm/test/results')
    image_files = [f for f in results_dir.glob("*.png")]
    
    for img_file in image_files:
        img_name = img_file.name
        rel_path = f"results/{img_name}"  # 相对路径，用于HTML
        html_content += f"""
        <div class="image-container">
            <h3>{img_name.replace('_', ' ').replace('.png', '')}</h3>
            <img src="{rel_path}" alt="{img_name}">
        </div>"""

    # 完成HTML文件
    html_content += """
    </div>
</body>
</html>
"""

    # 保存HTML报告
    with open(results_dir / 'test_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    run_all_tests() 