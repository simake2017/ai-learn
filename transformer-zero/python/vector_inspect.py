#!/usr/bin/env python3
"""
vector_inspect.py - 详细展示两个词的向量值和差别

用法:
    python3 python/vector_inspect.py apple pear
    python3 python/vector_inspect.py man woman
    python3 python/vector_inspect.py        # 交互式
"""
import numpy as np
import sys
from pathlib import Path

DEFAULT_BIN = "/home/magene/runtime/ai/llm.c/gpt2_124M.bin"

bin_path = sys.argv[1] if len(sys.argv) > 1 and Path(sys.argv[1]).exists() else DEFAULT_BIN
# 检查是否是默认 bin
if Path(bin_path).exists():
    print(f"加载 {bin_path}...")
else:
    bin_path = DEFAULT_BIN
    print(f"加载 {bin_path}...")

with open(bin_path, 'rb') as f:
    f.read(256 * 4)
    wte = np.frombuffer(f.read(50304 * 768 * 4), dtype=np.float32).reshape(50304, 768)

print(f"wte 形状: {wte.shape}")
print()

import tiktoken
enc = tiktoken.get_encoding("gpt2")


def vec(word):
    ids = enc.encode_ordinary(word)
    if len(ids) == 0:
        return None
    return wte[ids].mean(axis=0)


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def show(word1, word2, top=10):
    v1 = vec(word1)
    v2 = vec(word2)
    if v1 is None or v2 is None:
        print(f"无法编码 '{word1}' 或 '{word2}'")
        return
    sim = cos_sim(v1, v2)
    d = v1 - v2
    abs_d = np.abs(d)

    print(f"\n{'=' * 70}")
    print(f"对比: '{word1}' vs '{word2}'")
    print(f"余弦相似度: {sim:.4f}")
    print(f"{'=' * 70}")

    # 前 10 维
    print(f"\n前 10 维数值:")
    print(f"{'维度':<6}{word1:>14}{word2:>14}{'差':>14}")
    for i in range(10):
        print(f"{i:<6}{v1[i]:>14.4f}{v2[i]:>14.4f}{d[i]:>14.4f}")

    # 最不同
    top_indices = np.argsort(abs_d)[-top:][::-1]
    print(f"\n最不同的 {top} 个维度（差距大）:")
    print(f"{'维度':<6}{word1:>14}{word2:>14}{'差':>14}")
    for i in top_indices:
        print(f"{i:<6}{v1[i]:>14.4f}{v2[i]:>14.4f}{d[i]:>14.4f}")

    # 最相似
    bot_indices = np.argsort(abs_d)[:top]
    print(f"\n最相似的 {top} 个维度（差距小）:")
    print(f"{'维度':<6}{word1:>14}{word2:>14}{'差':>14}")
    for i in bot_indices:
        print(f"{i:<6}{v1[i]:>14.4f}{v2[i]:>14.4f}{d[i]:>14.4f}")


print("=" * 70)
print("向量详细查看器")
print("=" * 70)
print()
print("用法:")
print("  交互: 输入两个英文词，查看它们的向量差异")
print("  退出: Ctrl+C 或 Ctrl+D")
print()

# 命令行参数模式
if len(sys.argv) >= 3:
    show(sys.argv[1], sys.argv[2])
elif len(sys.argv) == 2 and not Path(sys.argv[1]).exists():
    # 单个词可能不存在于默认路径
    pass

# 交互式
while True:
    try:
        print()
        w1 = input("词1> ").strip()
        if not w1:
            continue
        w2 = input("词2> ").strip()
        if not w2:
            continue
        show(w1, w2)
    except (EOFError, KeyboardInterrupt):
        print("\n再见!")
        break