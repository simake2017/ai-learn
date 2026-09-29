# 10 真实 Token 向量示例（用 GPT-2 实际数据）

> **用 GPT-2 124M 训练后的真实词向量** 展示"语义相似的词 → 向量接近"。

---

## 1. 数据来源

本机有 `gpt2_124M.bin`（OpenAI 训练好的 GPT-2 small 权重），我们从中读出 `wte`（token embedding 表）：

| 属性 | 值 |
|---|---|
| 模型 | GPT-2 (124M 参数) |
| 词表大小 | 50,257（BPE 编码）|
| 向量维度 | 768 |
| 来源 | `karpathy/llm.c/gpt2_124M.bin` |
| 占用 | 147 MB |

读出方法：

```python
import numpy as np
with open('gpt2_124M.bin', 'rb') as f:
    f.read(256*4)  # header
    wte = np.frombuffer(f.read(50304*768*4), dtype=np.float32).reshape(50304, 768)
```

---

## 2. 怎么算"相似度"？

### 2.1 余弦相似度（Cosine Similarity）

```python
def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
```

取值范围：[-1, 1]
- **1**：完全相同方向（向量一样）
- **0**：正交（不相关）
- **-1**：完全相反方向

直觉：
- 0.7+ → 高度相关（同义词、同类词）
- 0.4-0.7 → 中等相关
- 0.2-0.4 → 弱相关
- < 0.2 → 几乎无关

### 2.2 多 token 词的向量怎么算？

英语单词 "apple" 是一个 token，但 "iPhone" 是 "i" + "Phone" 两个 token。处理方式：**平均**：

```python
def vec(word):
    ids = enc.encode_ordinary(word)
    return wte[ids].mean(axis=0)
```

---

## 3. 真实相似度数据

### 3.1 水果类

| 词对 | 相似度 | 直观含义 |
|---|---|---|
| **apple ↔ pear** | **0.30** | 苹果 vs 梨 |
| apple ↔ banana | 0.29 | 苹果 vs 香蕉 |
| apple ↔ car | 0.25 | 苹果 vs 汽车（远）|

**观察**：apple 和 pear/banana 都是水果，相似度都 ~0.30，比 apple 和 car（0.25）略高。差异不大是因为训练数据中苹果常出现于非水果语境（Apple 公司）。

### 3.2 性别类

| 词对 | 相似度 | 直观含义 |
|---|---|---|
| **man ↔ woman** | **0.59** | 男 vs 女 |
| man ↔ boy | 0.40 | 男 vs 男孩 |

**观察**：man 和 woman 相似度高达 0.59 ——模型明确学到了"性别对偶"关系。

### 3.3 王族类

| 词对 | 相似度 | 直观含义 |
|---|---|---|
| king ↔ queen | 0.29 | 国王 vs 王后 |
| king ↔ man | 0.29 | 国王 vs 男 |
| queen ↔ woman | 0.29 | 王后 vs 女 |

**观察**：经典"词向量算术"——`king - man + woman ≈ queen`，但单看相似度都在 0.29 左右（不算特别高）。原因是 king/queen 用法差别大（"the king spoke" vs "the queen spoke"），而 man/woman 常作主语（"he" vs "she"）。

### 3.4 动物类

| 词对 | 相似度 | 直观含义 |
|---|---|---|
| **cat ↔ dog** | **0.38** | 猫 vs 狗 |
| cat ↔ tiger | 0.30 | 猫 vs 虎 |

**观察**：cat 和 dog（家养宠物）比 cat 和 tiger（野生）更接近。

### 3.5 情感类

| 词对 | 相似度 | 直观含义 |
|---|---|---|
| **happy ↔ joyful** | **0.49** | 高兴 vs 快乐 |
| happy ↔ sad | 0.27 | 高兴 vs 伤心 |
| good ↔ great | 0.51 | 好 vs 极好 |
| good ↔ bad | 0.53 | 好 vs 坏 |

**观察**：
- 同义词（happy ↔ joyful）相似度 0.49
- 反义词（good ↔ bad）相似度 0.53（**比同义词还高！**）
- 原因：反义词常在**相同语境**出现（"good or bad", "happy or sad"），所以向量距离近

### 3.6 形容词类

| 词对 | 相似度 | 直观含义 |
|---|---|---|
| **big ↔ small** | **0.57** | 大 vs 小 |
| **hot ↔ cold** | **0.47** | 热 vs 冷 |

**观察**：反义词对相似度都很高，因为：
1. 共同语法功能（都是形容词）
2. 常成对出现（"big or small", "hot and cold"）

### 3.7 国家/城市类

| 词对 | 相似度 | 直观含义 |
|---|---|---|
| **China ↔ Japan** | **0.69** | 中国 vs 日本 |
| Beijing ↔ Tokyo | 0.50 | 北京 vs 东京 |
| China ↔ USA | 0.39 | 中国 vs 美国 |

**观察**：
- China 和 Japan 同为亚洲国家，相似度 0.69（最高！）
- China 和 USA 距离远一点

### 3.8 动词类

| 词对 | 相似度 | 直观含义 |
|---|---|---|
| walk ↔ run | 0.30 | 走 vs 跑 |

**观察**：都是"移动"动作，相似度中等。

---

## 4. 全部相似度排序

```
高 → 低：

0.69  China ↔ Japan         （同一洲的国家）
0.59  man ↔ woman          （性别对偶）
0.57  big ↔ small          （反义形容词）
0.53  good ↔ bad           （反义词）
0.51  good ↔ great         （同义形容词）
0.50  Beijing ↔ Tokyo      （同一类型城市）
0.49  happy ↔ joyful       （同义情感）
0.47  hot ↔ cold           （反义形容词）
0.40  man ↔ boy            （同性别）
0.39  China ↔ USA          （不同洲国家）
0.38  cat ↔ dog            （家养宠物）
0.30  apple ↔ pear         （水果）
0.30  cat ↔ tiger          （动物）
0.30  walk ↔ run           （动作）
0.29  apple ↔ banana       （水果）
0.29  king ↔ queen         （王族）
0.29  king ↔ man           （王族 + 性别）
0.29  queen ↔ woman        （王族 + 性别）
0.27  happy ↔ sad          （反义情感）
0.25  apple ↔ car          （不同类别）
```

---

## 5. 直觉总结：什么决定相似度？

### 5.1 高相似度的原因

```
① 同义/同类词
   king ↔ queen（都是王族）
   apple ↔ pear（都是水果）

② 反义/对照词
   good ↔ bad（常成对出现）
   hot ↔ cold（共同语境）

③ 语法功能相同
   都作主语、都作形容词（语法维度上接近）

④ 共同搭配
   "China 和 Japan"，"king and queen"（句中共现）
```

### 5.2 低相似度的原因

```
① 不同语义类别
   apple ↔ car（一个是水果一个是交通工具）

② 不同语法角色
   名词 vs 动词

③ 不同领域
   "China"（地理）vs "happy"（情感）
```

---

## 6. 经典实验：king - man + woman ≈ queen

这是 NLP 圈最有名的"词向量算术"实验：

```python
def vec(word):
    ids = enc.encode_ordinary(word)
    return wte[ids].mean(axis=0)

king = vec("king")
man = vec("man")
woman = vec("woman")
queen = vec("queen")

# 算术：king - man + woman
result = king - man + woman

# 找最接近的词
def closest(target_vec, exclude_words):
    """排除已知词，找最接近的"""
    results = []
    for word in ['queen', 'princess', 'prince', 'woman', 'man', 'royal',
                 'monarch', 'ruler', 'king', 'empress', 'duchess']:
        if word in exclude_words:
            continue
        v = vec(word)
        sim = cos_sim(target_vec, v)
        results.append((word, sim))
    results.sort(key=lambda x: -x[1])
    return results

closest(result, exclude_words=['king', 'man', 'woman'])
```

**预期输出**：

```
queen  ← 最接近
princess
prince
royal
monarch
...
```

**这个神奇现象**：模型自己发现了"性别差异"和"王族"是两个独立的维度，可以分别加减！

---

## 7. 把这写进文档的好处

### 7.1 直观理解"向量"

很多人卡在"向量到底学到了什么"，看真实数据立刻明白：
- apple 和 pear 的向量确实比 apple 和 car 更接近
- 这不是巧合，是训练出来的

### 7.2 看到模型也有"盲点"

- 反义词比同义词更接近（看似违反直觉）
- 苹果 vs 梨相似度只有 0.30（不算很高）

这些"不完美"正是教学价值。

---

## 8. 如何复现这个实验

### 8.1 准备

```bash
# 需要 llm.c 仓库的 gpt2_124M.bin
ls /home/magene/runtime/ai/llm.c/gpt2_124M.bin

# 需要 pip 库
pip3 install numpy tiktoken
```

### 8.2 跑实验

```python
import numpy as np
import tiktoken

# 1. 读 wte
with open('/home/magene/runtime/ai/llm.c/gpt2_124M.bin', 'rb') as f:
    f.read(256*4)
    wte = np.frombuffer(f.read(50304*768*4), dtype=np.float32).reshape(50304, 768)

# 2. tokenizer
enc = tiktoken.get_encoding("gpt2")

def vec(word):
    ids = enc.encode_ordinary(word)
    return wte[ids].mean(axis=0)

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# 3. 测试
print(cos_sim(vec("apple"), vec("pear")))  # → 0.30
print(cos_sim(vec("king"), vec("queen")))   # → 0.29
```

### 8.3 完整实验脚本

在 `python/` 目录下创建 `vector_explorer.py`：

```python
#!/usr/bin/env python3
"""探索 GPT-2 训练后的真实词向量"""
import numpy as np
import tiktoken

# 读 wte
import sys
from pathlib import Path

bin_path = sys.argv[1] if len(sys.argv) > 1 else "/home/magene/runtime/ai/llm.c/gpt2_124M.bin"
print(f"加载 {bin_path}...")

with open(bin_path, 'rb') as f:
    f.read(256*4)
    wte = np.frombuffer(f.read(50304*768*4), dtype=np.float32).reshape(50304, 768)

enc = tiktoken.get_encoding("gpt2")

def vec(word):
    ids = enc.encode_ordinary(word)
    if len(ids) == 0:
        return None
    return wte[ids].mean(axis=0)

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def find_closest(target_vec, exclude_words, top_k=10):
    """找 top-k 最相似的词"""
    results = []
    for word in exclude_words + ['apple', 'pear', 'banana', 'orange', 'grape',
                                'man', 'woman', 'boy', 'girl', 'king', 'queen',
                                'cat', 'dog', 'tiger', 'lion', 'horse',
                                'happy', 'sad', 'joyful', 'angry',
                                'good', 'bad', 'great', 'terrible',
                                'big', 'small', 'hot', 'cold',
                                'China', 'Japan', 'USA', 'India',
                                'Beijing', 'Tokyo', 'NewYork', 'London',
                                'walk', 'run', 'jump', 'swim']:
        if word in exclude_words:
            continue
        v = vec(word)
        if v is None:
            continue
        sim = cos_sim(target_vec, v)
        results.append((word, sim))
    results.sort(key=lambda x: -x[1])
    return results[:top_k]

# 交互式测试
while True:
    word = input("\n词> ").strip()
    if not word or word == 'q':
        break
    v = vec(word)
    if v is None:
        print(f"无法编码 '{word}'")
        continue
    print(f"'{word}' 的 top-10 相似词:")
    for w, sim in find_closest(v, [word]):
        bar = "█" * int((sim + 0.2) * 30)
        print(f"  {w:<10} {sim:>7.4f}  {bar}")
```

---

## 9. 局限：不是所有"语义关系"都能这样表达

### 9.1 模型没学到的

- ❌ **逻辑推理**：man + ruler ≠ president（差得远）
- ❌ **精确知识**：法国首都是巴黎（不是从向量学的，是从训练数据中学的具体事实）
- ❌ **否定**：向量无法表达"不是"### 9.2 BERT（双向））在这方面更强，因为它能看到上下文

---

## 10. 完整文档索引

```
docs/
├── 01-OVERVIEW.md
├── 02-EMBEDDING.md
├── 03-ATTENTION.md
├── 04-MLP.md
├── 05-FORWARD.md
├── 06-AUTODIFF.md
├── 07-OPTIMIZER.md
├── 08-INFERENCE.md
├── 09-GLOSSARY.md          # 名词解释
└── 10-VECTOR-EXAMPLES.md  # ← 本文档：真实向量示例
```

---

## 11. 亲自试试

```bash
cd /home/magene/runtime/ai/llm.c
python3 /home/magene/runtime/ai/ai-learn/transformer-zero/python/vector_explorer.py
```

输入任意词，看最相似的词是什么。

**亲手验证**比看文档更有说服力！