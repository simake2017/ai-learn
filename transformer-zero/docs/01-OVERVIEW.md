# 01 Transformer 整体架构

> 本文档用 **"我喜欢苹果，它真好吃"** 这个例子贯穿全文，帮你建立直觉。

## 1. Transformer 是什么？

Transformer 是 2017 年 Google 在论文 *"Attention Is All You Need"* 中提出的神经网络架构。它是**现代所有大语言模型**的骨架：GPT、LLaMA、Claude、BERT 全是 Transformer 的变体。

我们要实现的是 **GPT 风格**的 Transformer（也叫 **Decoder-only Transformer**）：

```
┌─────────────────────────────────────────────────────────────┐
│  输入：一串 token ids   [I, like, apples, ,, they, are]    │
│      ↓                                                      │
│  ┌──────────────────────┐                                   │
│  │  Embedding           │  token → 向量                   │
│  │  + Position          │  + 位置信息                       │
│  └──────────┬───────────┘                                   │
│             ↓                                               │
│  ┌──────────────────────┐                                   │
│  │  Transformer Block × N│  （N=6，我们用小模型）         │
│  │   ┌────────────────┐ │                                   │
│  │   │ LayerNorm      │ │                                   │
│  │   │ Multi-Head     │ │  ← 核心：让 token 之间"交流"   │
│  │   │ Self-Attention │ │                                   │
│  │   │ + 残差          │ │                                   │
│  │   ├────────────────┤ │                                   │
│  │   │ LayerNorm      │ │                                   │
│  │   │ MLP (FFN)      │ │  ← 每个 token 独立"思考"         │
│  │   │ + 残差          │ │                                   │
│  │   └────────────────┘ │                                   │
│  └──────────┬───────────┘                                   │
│             ↓                                               │
│  ┌──────────────────────┐                                   │
│  │  LM Head             │  向量 → 词表大小的 logits            │
│  └──────────┬───────────┘                                   │
│             ↓                                               │
│  logits：每个位置上，50257 个 token 的分数                    │
└─────────────────────────────────────────────────────────────┘
```

## 2. 数据流：以 "I like apples" 为例

假设我们的词表是字符级：`vocab_size=65`，序列长度 `T=3`。

**输入**：`[I, like, apples]`（3 个 token）
**目标**：在每个位置预测下一个 token
- 看到 `I` → 预测 `like`
- 看到 `like` → 预测 `apples`
- 看到 `apples` → 预测 `<EOT>` 或下一句

### 2.1 张量形状变化

| 步骤 | 张量 | 形状 |
|---|---|---|
| 输入 | `idx` | `(B, T) = (1, 3)` |
| Token Embedding | `tok_emb` | `(B, T, C) = (1, 3, 128)` |
| Position Embedding | `pos_emb` | `(T, C) = (3, 128)` |
| Embedding 输出 | `x` | `(B, T, C) = (1, 3, 128)` |
| Block × 6 输出 | `x` | `(B, T, C) = (1, 3, 128)` |
| LM Head | `logits` | `(B, T, V) = (1, 3, 65)` |

符号：
- `B` = batch size（多少条序列并行训练）
- `T` = sequence length（每条序列多长）
- `C` = n_embd = 128（每个 token 的向量维度）
- `V` = vocab_size = 65（词表大小）
- `n_layer` = 6（Transformer block 数）
- `n_head` = 4（多头注意力的头数）

### 2.2 直观理解每个步骤做什么

**Embedding**：把离散的 token id 转成连续的向量。

```
"I"      → [0.12, -0.34, 0.56, ..., 0.78]   (128 个数)
"like"   → [0.45, 0.67, -0.89, ..., 0.12]
"apples" → [-0.23, 0.78, 0.12, ..., -0.45]
```

**Position Embedding**：加上位置信息，否则模型不知道"apples" 在第3 个位置。

**Attention**：让每个 token "看" 其他 token，提取关系。
- "apples" 看到 "I" 和 "like"，知道这是在说"我喜欢 apples"
- 计算 `Q·K^T` 得到 attention 权重，决定关注谁

**MLP**：每个 token 独立地用学到的知识"思考"。
- 把 attention 提取的关系信息加工成更抽象的表示

**LM Head**：把 128 维向量映射到 65 维（词表大小），每个维度对应一个 token 的"分数"。

```
位置 0（"I"）      → logits = [1.2, 0.3, 2.1, ..., 0.8]   (65 个)
位置 1（"like"）   → logits = [0.5, 3.4, 1.2, ..., 1.1]   (65 个)
位置 2（"apples"） → logits = [0.7, 0.2, 0.9, ..., 2.5]   (65 个)
```

**预测**：softmax 后取 argmax 就是下一个 token 的预测。

## 3. 因果语言模型（Causal LM）

GPT 只能看到**当前和之前**的 token，不能看未来。这是通过 **causal mask** 实现的：

```
位置: 0  1  2
  0:  [✓  ✗  ✗]      ← "I" 只能看自己
  1:  [✓  ✓  ✗]      ← "like" 能看 "I" 和自己
  2:  [✓  ✓  ✓]      ← "apples" 能看前面所有
```

**为什么？** 因为推理时模型是**逐个**生成 token 的：先生成 `I`，再生成 `like`，最后生成 `apples`。训练时也要模拟这个过程。

## 4. 损失函数：交叉熵

模型预测每个位置下一个 token 的概率分布 `P`，真实 token 是 `y`。loss 是交叉熵：

```
L = -1/(B*T) Σ log P(y[i] | x[:i+1])
```

直觉：如果模型对真实 token 给的概率是 `p`，loss = `-log(p)`：
- `p=0.99` → loss ≈ 0.01（几乎完美）
- `p=0.50` → loss ≈ 0.69（一般）
- `p=0.01` → loss ≈ 4.6（很差）

训练目标：**最小化 loss = 让模型对正确 token 给高概率**。

## 5. 训练 = 调参 + 监督学习

模型有 ~600K 个参数（vocab=65, embd=128, 6 层）。训练过程：

```
for iter in range(num_iters):
    batch = sample data            # 拿一批数据
    logits, loss = forward(batch)  # 算 loss
    grads = compute_grads(loss)    # 算梯度（数值微分）
    params = adamw_update(params, grads)  # 调参数
```

每次调一点点参数，让 loss 变小。

## 6. 项目文件结构

```
transformer-zero/
├── c/                  # C 核心实现
│   ├── transformer.h   # 数据结构 + API
│   ├── matrix.c        # 矩阵运算
│   ├── activations.c   # GELU, softmax
│   ├── layernorm.c     # LayerNorm
│   ├── attention.c     # 多头因果 Attention
│   ├── model.c         # 拼装 GPT
│   ├── autodiff.c      # 数值微分
│   ├── optimizer.c     # AdamW
│   └── main.c          # 训练 + 推理入口
├── python/             # Python 胶水
│   ├── train.py        # 调 C .so + 画图
│   ├── visualize.py    # attention 热力图
│   └── bindings.py     # ctypes 接口
├── data/               # tinyshakespeare
├── docs/               # 你正在看的文档
├── build.sh            # 一键编译
└── run.sh              # 一键训练
```

## 7. 推荐阅读顺序

| 顺序 | 文档 | 重点 |
|---|---|---|
| 1 | **01-OVERVIEW** (本文) | 整体架构 |
| 2 | 02-EMBEDDING | token → 向量的数学 |
| 3 | 03-ATTENTION | Q/K/V + 因果掩码 + 多头 |
| 4 | 04-MLP | FFN + GELU + 残差 |
| 5 | 05-FORWARD | 完整 forward 流程 |
| 6 | 06-AUTODIFF | 数值微分原理 |
| 7 | 07-OPTIMIZER | AdamW |
| 8 | 08-INFERENCE | "我喜欢苹果" → "苹果" 完整例子 |

**配合 C 代码读**，每个文档都会引用 `c/` 里的具体行号。

## 8. 关键概念速查

| 概念 | 一句话解释 |
|---|---|
| **Token** | 文本的最小单位（字符/词/子词） |
| **Embedding** | token → 向量 |
| **Attention** | token 之间"互相看"，提取关系 |
| **Causal Mask** | 位置 t 只能看 ≤ t 的位置 |
| **Multi-Head** | 多组 Q/K/V 并行，学不同关系 |
| **MLP / FFN** | 每个 token 独立做非线性变换 |
| **Residual** | `y = x + sublayer(x)`，梯度直道 |
| **LayerNorm** | 沿特征维度归一化 |
| **LM Head** | 隐层 → 词表大小 |
| **Loss** | 交叉熵：模型对真实 token 的 `-log(p)` |
| **Autograd / 数值微分** | 算梯度 |
| **AdamW** | 用梯度更新参数的优化器 |

下一步：[02-EMBEDDING.md](02-EMBEDDING.md) — 看 token 是怎么变成向量的。