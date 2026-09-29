#!/usr/bin/env python3
"""
vector_explorer.py - 探索 GPT-2 训练后的真实词向量

用法:
    python3 python/vector_explorer.py
    python3 python/vector_explorer.py /path/to/gpt2_124M.bin

交互式输入任意英文词，显示最相似的 10 个词
"""
import numpy as np
import sys

DEFAULT_BIN = "/home/magene/runtime/ai/llm.c/gpt2_124M.bin"

bin_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BIN

print(f"加载 {bin_path}...")

try:
    with open(bin_path, 'rb') as f:
        f.read(256 * 4)  # header
        wte = np.frombuffer(f.read(50304 * 768 * 4), dtype=np.float32).reshape(50304, 768)
except FileNotFoundError:
    print(f"找不到 {bin_path}")
    print(f"请下载 gpt2_124M.bin (498MB) 到该路径")
    print(f"或运行: ./dev/download_starter_pack.sh")
    sys.exit(1)

print(f"wte 形状: {wte.shape}")
print()

try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
except ImportError:
    print("需要 pip install tiktoken")
    sys.exit(1)


def vec(word):
    """拿到一个词的向量（多 token 平均）"""
    ids = enc.encode_ordinary(word)
    if len(ids) == 0:
        return None
    return wte[ids].mean(axis=0)


def cos_sim(a, b):
    """余弦相似度"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# 候选词库
VOCAB = [
    # 水果
    'apple', 'pear', 'banana', 'orange', 'grape', 'peach', 'mango',
    'strawberry', 'cherry', 'lemon', 'watermelon', 'pineapple',
    # 性别
    'man', 'woman', 'boy', 'girl', 'king', 'queen', 'prince', 'princess',
    'father', 'mother', 'son', 'daughter', 'brother', 'sister',
    # 动物
    'cat', 'dog', 'tiger', 'lion', 'horse', 'cow', 'pig', 'sheep',
    'elephant', 'monkey', 'rabbit', 'mouse', 'bear', 'wolf',
    # 情感
    'happy', 'sad', 'angry', 'joyful', 'excited', 'bored', 'calm',
    'good', 'bad', 'great', 'terrible', 'awful', 'wonderful',
    # 形容词
    'big', 'small', 'hot', 'cold', 'fast', 'slow', 'high', 'low',
    'old', 'young', 'beautiful', 'ugly', 'rich', 'poor',
    # 国家/城市
    'China', 'Japan', 'USA', 'India', 'France', 'Germany', 'Italy',
    'Beijing', 'Tokyo', 'NewYork', 'Paris', 'London', 'Berlin',
    # 动词
    'walk', 'run', 'jump', 'swim', 'eat', 'drink', 'sleep', 'work',
    'love', 'hate', 'think', 'know', 'see', 'hear', 'speak',
    # 颜色
    'red', 'blue', 'green', 'yellow', 'black', 'white', 'pink',
    # 数字
    'one', 'two', 'three', 'four', 'five', 'ten', 'hundred',
    # 常见词
    'happy', 'love', 'time', 'life', 'world', 'good', 'bad', 'big', 'small',
    'hot', 'cold', 'fast', 'slow', 'high', 'low',
]


def find_closest(target_vec, exclude_words, top_k=10):
    """找 top-k 最相似的词"""
    results = []
    for word in VOCAB:
        if word in exclude_words:
            continue
        v = vec(word)
        if v is None:
            continue
        sim = cos_sim(target_vec, v)
        results.append((word, sim))
    results.sort(key=lambda x: -x[1])
    return results[:top_k]


def vec_arithmetic():
    """词向量算术：king - man + woman ≈ ?"""
    print("\n=== 词向量算术实验 ===")
    print("king - man + woman ≈ ?\n")

    king = vec("king")
    man = vec("man")
    woman = vec("woman")

    if king is None or man is None or woman is None:
        print("某个词无法编码")
        return

    result = king - man + woman
    print("'king - man + woman' 最接近的词:")
    for w, sim in find_closest(result, ['king', 'man', 'woman'], top_k=10):
        bar = "█" * int((sim + 0.2) * 30)
        print(f"  {w:<12} {sim:>7.4f}  {bar}")
    print()


def main():
    print("=" * 50)
    print("GPT-2 真实词向量探索器")
    print("=" * 50)
    print()
    print("用法:")
    print("  输入英文词 → 显示最相似的 10 个词")
    print("  输入 'vec' → 词向量算术实验 (king - man + woman)")
    print("  输入 'q' → 退出")
    print()

    # 先跑算术实验
    vec_arithmetic()

    while True:
        try:
            word = input("\n词> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not word:
            continue
        if word == 'q':
            print("再见!")
            break
        if word == 'vec':
            vec_arithmetic()
            continue

        v = vec(word)
        if v is None:
            print(f"无法编码 '{word}'")
            continue

        print(f"\n'{word}' 的最相似词:")
        for w, sim in find_closest(v, [word]):
            bar = "█" * int((sim + 0.2) * 30)
            print(f"  {w:<12} {sim:>7.4f}  {bar}")


if __name__ == "__main__":
    main()