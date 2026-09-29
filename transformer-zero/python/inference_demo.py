#!/usr/bin/env python3
"""
python/inference_demo.py - "我喜欢苹果，它真好吃" → "苹果" 推理演示

这是教学项目的简化版。完整 GPT 推理需要大模型和 KV cache，
教学版本主要演示流程：

  1. 加载训练好的模型
  2. 把文本转成 token ids
  3. 自回归生成
  4. 把 token ids 转回文本
"""

import subprocess
import os
from pathlib import Path


def main():
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)

    print("=" * 60)
    print("Transformer 推理演示")
    print("=" * 60)
    print()
    print("注意：教学模型规模小（vocab=65, embd=128, 6层）")
    print("训练数据是 tinyshakespeare（英文），不是中文。")
    print("中文演示需要：")
    print("  1. 中文分词数据")
    print("  2. 更大的模型（>10M参数）")
    print("  3. 长时间训练")
    print()
    print("我们演示英文版本：")
    print('  prompt: "ROMEO:" → 模型生成后续字符')
    print()

    # 调用 C 推理
    result = subprocess.run(
        ["./build/transformer", "generate"],
        capture_output=True, text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)


if __name__ == "__main__":
    main()