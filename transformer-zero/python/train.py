#!/usr/bin/env python3
"""
python/train.py - 调 C 库 + 画训练曲线

教学项目，因为 C 端用数值微分算梯度，
每次 forward 都很慢，所以 Python 端只做辅助：
  1. 调 C .so 训练
  2. 解析日志画 loss 曲线
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)

    print("=" * 60)
    print("Transformer 从零实现 - 训练脚本")
    print("=" * 60)

    # 编译
    print("\n[1/3] 编译 C 代码...")
    subprocess.run(["bash", "build.sh"], check=True)

    # 训练
    print("\n[2/3] 训练模型...")
    result = subprocess.run(
        ["./build/transformer", "train"],
        capture_output=True, text=True
    )
    print(result.stdout)

    # 推理
    print("\n[3/3] 推理生成...")
    result = subprocess.run(
        ["./build/transformer", "generate"],
        capture_output=True, text=True
    )
    print(result.stdout)

    print("\n完成！")


if __name__ == "__main__":
    main()