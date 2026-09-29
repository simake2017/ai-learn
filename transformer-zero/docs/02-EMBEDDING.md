# 02 Embedding：token → 向量

> 这一步把离散的 token id 变成连续的向量，让神经网络能"理解"。

## 1. 为什么需要 Embedding？

神经网络只能处理数字（浮点数张量），不能直接处理 `"I"`、`"like"` 这种字符串。

```
原始 token:   "I"      "like"    "apples"
              ↓         ↓         ↓
token id:     12        45        78       (整数)
              ↓         ↓         ↓
embedding:    [0.1,    [0.4,    [-0.2,    (128 维浮点向量)
               -0.3,    0.6,      0.7,
               0.5,     -0.8,     0.3,
               ...]     ...]      ...]
```

**关键洞察**：相似的词应该有相似的向量（"king" 和 "queen" 的 embedding 在某些维度上接近）。

## 2. Embedding 的数学本质

**Embedding 就是一个查找表（lookup table）**：

```
token_embedding_table: shape (vocab_size, n_embd) = (65, 128)

                   128 维向量
              ┌──────────────────┐
token 0 →     │  0.12 -0.34 ...  0.78 │
token 1 →     │  0.45  0.67 ... -0.12 │
...           │  ...                  │
token 64 →    │ -0.23  0.78 ... -0.45 │
              └──────────────────┘
```

把 token id 当作索引，查表得到对应的向量。

## 3. 代码实现

### 3.1 Python 版（参考）

```python
# nanoGPT/model.py:236
tok_emb = self.transformer.wte(idx)  # (B, T) → (B, T, C)
```

PyTorch 帮你处理了查表。我们手写版本：

```python
# 伪代码
def embedding_forward(token_ids, embedding_table):
    """
    token_ids: shape (B, T) 整数
    embedding_table: shape (V, C) 浮点
    return: shape (B, T, C)
    """
    B, T = token_ids.shape
    V, C = embedding_table.shape
    out = np.zeros((B, T, C), dtype=np.float32)

    for b in range(B):
        for t in range(T):
            token_id = token_ids[b, t]
            out[b, t] = embedding_table[token_id]

    return out
```

### 3.2 C 版（我们的实现）

详见 `c/model.c:embedding_forward`。

```c
// c/model.c 里的核心循环
for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
        int token_id = inputs[b * T + t];
        // 查表：把 token_id 当索引
        for (int c = 0; c < C; c++) {
            out[(b * T + t) * C + c] = wte[token_id * C + c];
        }
    }
}
```

**核心**：3 层循环，最内层是内存复制。

## 4. 位置编码（Position Embedding）

只有 token embedding，模型不知道 token 在序列中的位置（"apples 在第几个？"）。

### 4.1 为什么需要位置信息？

考虑这两个句子：

```
"I like apples"   →  [I, like, apples]
"apples like I"   →  [apples, like, I]
```

只看 token embedding，这两个句子的表示一样。模型分不清语序。

### 4.2 GPT 的做法：可学习的位置编码

不像原始 Transformer 用固定 sin/cos 编码，GPT 用**可学习**的位置向量：

```
position_embedding_table: shape (block_size, n_embd) = (64, 128)

                  128 维向量
              ┌──────────────────┐
位置 0 →      │  0.01  0.02 ...  0.03 │
位置 1 →      │  0.04 -0.01 ...  0.02 │
...           │  ...                  │
位置 63 →     │  0.05  0.07 ... -0.01 │
              └──────────────────┘
```

训练过程中，模型自己学"位置 0 的 token 通常是一句话开头"等等规律。

### 4.3 代码

```c
// c/model.c:forward_embedding
void forward_embedding(
    float* out,           // (B, T, C) 输出
    const int* inputs,    // (B, T) token ids
    const float* wte,     // (V, C) token embedding
    const float* wpe,     // (maxT, C) position embedding
    int B, int T, int C
) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int token_id = inputs[b * T + t];
            // 输出 = token 嵌入 + 位置嵌入
            for (int c = 0; c < C; c++) {
                out[(b * T + t) * C + c] =
                    wte[token_id * C + c] +  // token embedding
                    wpe[t * C + c];          // position embedding
            }
        }
    }
}
```

## 5. 数据维度变化

假设 `B=1, T=3, C=128`：

```
输入: idx (1, 3)            int
       ↓
       查表 wte (65, 128)
       ↓
tok_emb (1, 3, 128)         float
       +
       查表 wpe (3, 128)    ← 位置 0/1/2 广播到 batch 维
       ↓
x (1, 3, 128)               float
```

## 6. 数值示例

假设：
- `vocab_size = 5`（小例子）
- `n_embd = 4`
- `T = 3`
- 输入 `idx = [1, 3, 2]`

```
wte:
  token 0: [0.0, 0.0, 0.0, 0.0]
  token 1: [0.1, 0.2, 0.3, 0.4]
  token 2: [0.5, 0.6, 0.7, 0.8]
  token 3: [0.9, 1.0, 1.1, 1.2]
  token 4: [1.3, 1.4, 1.5, 1.6]

wpe:
  pos 0:   [0.01, 0.02, 0.03, 0.04]
  pos 1:   [0.05, 0.06, 0.07, 0.08]
  pos 2:   [0.09, 0.10, 0.11, 0.12]
```

**Embedding 输出**：

```
位置 0 (token 1):
  [0.1+0.01, 0.2+0.02, 0.3+0.03, 0.4+0.04]
= [0.11, 0.22, 0.33, 0.44]

位置 1 (token 3):
  [0.9+0.05, 1.0+0.06, 1.1+0.07, 1.2+0.08]
= [0.95, 1.06, 1.17, 1.28]

位置 2 (token 2):
  [0.5+0.09, 0.6+0.10, 0.7+0.11, 0.8+0.12]
= [0.59, 0.70, 0.81, 0.92]
```

## 7. 关键概念

| 概念 | 说明 |
|---|---|
| **Embedding 是查表** | 不是神经网络层，只是查找 |
| **可学习** | wte 和 wpe 都是模型参数，会被梯度下降更新 |
| **位置信息** | 没它模型不知道 token 顺序 |
| **维度变化** | (B, T) → (B, T, C) |

## 8. 常见疑问

**Q: 为什么不直接把 token id 当 one-hot 编码？**
A: One-hot 太稀疏（V 维但只有一个 1），浪费空间。Embedding 是稠密的 C 维，且可学习。

**Q: 为什么位置编码是可学习的而不是固定的 sin/cos？**
A: 实践上可学习效果差不多略好。GPT-2 之后大家基本都用可学习版本。

**Q: n_embd = 128 怎么选？**
A: 模型越大，n_embd 越大。GPT-2 small 用 768，大模型用 4096+。我们教学用 128，训练快、效果好。

下一步：[03-ATTENTION.md](03-ATTENTION.md) — 看 Attention 是怎么让 token 之间"交流"的。