#!/usr/bin/env python3
"""
python/visualize.py - 可视化训练日志和 attention

用法:
    python visualize.py --log /tmp/train.log
    python visualize.py --attention
"""

import argparse
import re
import sys
from pathlib import Path


def parse_log(path):
    """解析训练日志，返回 (steps, losses)"""
    steps, losses = [], []
    for line in Path(path).read_text(errors='ignore').splitlines():
        m = re.match(r"^step\s+(\d+):\s+loss\s+=\s+([\d.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return steps, losses


def plot_loss(steps, losses, save_path=None):
    """画 loss 曲线"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, 'b-o', markersize=4)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Transformer Training Loss")
    plt.grid(True, alpha=0.3)

    if losses:
        plt.axhline(y=min(losses), color='r', linestyle='--', alpha=0.5,
                    label=f'best: {min(losses):.4f}')
        plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"图已保存: {save_path}")
    else:
        plt.show()


def visualize_attention(att_matrix, tokens, layer=0, head=0, save_path=None):
    """画 attention 热力图"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy 未安装，跳过绘图")
        return

    plt.figure(figsize=(8, 8))
    plt.imshow(att_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='attention weight')

    n = len(tokens)
    plt.xticks(range(n), tokens, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(n), tokens, fontsize=8)

    plt.title(f"Layer {layer}, Head {head}")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"图已保存: {save_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", help="训练日志路径")
    p.add_argument("--save", help="保存图为 PNG")
    args = p.parse_args()

    if args.log:
        steps, losses = parse_log(args.log)
        if not steps:
            print(f"无法从 {args.log} 解析 loss")
            return
        print(f"解析到 {len(steps)} 个 steps")
        print(f"初始 loss: {losses[0]:.4f}")
        print(f"最终 loss: {losses[-1]:.4f}")
        if losses[0] > 0:
            print(f"下降比例: {(1 - losses[-1]/losses[0])*100:.1f}%")
        plot_loss(steps, losses, args.save)


if __name__ == "__main__":
    main()