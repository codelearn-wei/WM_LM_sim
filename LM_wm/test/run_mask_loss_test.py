#!/usr/bin/env python3
"""
运行掩码和损失函数测试的命令行脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

if __name__ == "__main__":
    from LM_wm.test.test_mask_and_loss import main
    main() 