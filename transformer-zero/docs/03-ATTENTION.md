# 03 Attention：token 之间的"交流"

> Attention 是 Transformer 最核心的创新。它让每个 token 能"看"其他 token，提取关系。

## 1. 直觉：从 "I like apples" 开始

**问题**：看到 `apples` 这个词时，模型怎么知道它是"我喜欢的东西"？

**直觉**：`apples` 应该"回头看" `I` 和 `like`，因为这两个词决定了它的语义角色（宾语、被喜欢的对象）。

**Attention 就是让 token 之间互相"通信"**：
- `apples` 问："谁喜欢我？" → `I` 和 `like` 都举手
- `I` 问："我喜欢什么？" → `apples` 举手
- `like` 问："什么东西被喜欢？" → `apples` 举手

## 2. Q/K/V 三兄弟

Attention 的核心是三个向量：**Query、Key、Value**。

类比查字典：
- **Query (Q)**：你想查什么（"我想找带'水果'意思的词"）
- **Key (K)**：字典里每个词的"标签"（"我是水果"）
- **Value (V)**：字典里每个词的"实际内容"

**过程**：
1. 用 Q 和所有 K 算相似度（"你的标签和我想要的多匹配"）
2. 相似度 → 概率分布（softmax）
3. 用概率加权求和所有 V（"取回最相关的内容"）

## 3. 数学定义（单头）

设输入为 `X`（B, T, C），对应 "I like apples"，T=3。

### 3.1 线性投影得到 Q/K/V

```
Q = X @ W_q   (B, T, d)   d = C/n_head = 32
K = X @ W_k   (B, T, d)
V = X @ W_v   (B, T, d)
```

每个 token 都生成一个 Q、一个 K、一个 V。

### 3.2 计算 attention 分数

对每个 query 位置 t：

```
score[t, j] = (Q[t] · K[j]) / sqrt(d)    j = 0..T-1
```

`Q · K` 是点积，相似度越大分数越高。`/sqrt(d)` 是缩放，防止 d 大时分数太大导致 softmax 饱和。

### 3.3 因果掩码（causal mask）

GPT 只能看 ≤ 当前 位置 的 token：

```
score[t, j] = -∞   if j > t   (看不见未来)
```

`softmax(-∞) = 0`，相当于"看不到未来"。

### 3.4 Softmax → 概率

```
att[t, j] = softmax_j(score[t, :])
```

att[t, j] 表示"位置 t 的 token 看位置 j 的 token 的权重"。

### 3.5 加权求和

```
out[t] = Σ_j att[t, j] * V[j]
```

把每个 token 的 V 按权重加起来。

## 4. 完整数学公式

```
att[t, j] = softmax( (Q[t] · K[j]) / sqrt(d) + mask[j > t] )
out[t]    = Σ_j att[t, j] * V[j]
```

其中 `mask[cond]` 在 cond=True 时为 -∞。

## 5. 因果 mask 可视化

```
mask = [[0,   -∞,  -∞],
        [0,   0,   -∞],
        [0,   0,   0 ]]     (0 表示可看，-∞ 表示不可看)

softmax 后：
      ┌                ┐
      │ 1.00  0.00  0.00 │   ← 位置 0 只能看自己
      │ 0.50  0.50  0.00 │   ← 位置 1 看 0 和自己
      │ 0.33  0.33  0.33 │   ← 位置 2 看所有
      └                ┘
```

## 6. 多头注意力（Multi-Head）

**为什么多头？** 不同 head 可以学不同的"关系"：
- Head 1：学"主谓一致"（动词看主语）
- Head 2：学"指代"（代词看名词）
- Head 3：学"长距离依赖"
- Head 4：学"局部语法"

**实现**：把 C 维拆成 n_head 段，每段独立算 attention：

```
Q, K, V: (B, T, C) → reshape → (B, T, n_head, d) → transpose → (B, n_head, T, d)

其中 d = C / n_head

对每个 head：
  att = softmax(Q @ K^T / sqrt(d) + mask)
  out = att @ V

最后把 n_head 个 out 拼回去：(B, n_head, T, d) → (B, T, n_head, d) → (B, T, C)
```

## 7. 数值例子："I like apples"

设 `T=3, C=4, n_head=2, d=2`（小例子）。

### 7.1 输入 X（已加 embedding）

```
X[0] = "I":      [0.1, 0.2, 0.3, 0.4]
X[1] = "like":   [0.5, 0.6, 0.7, 0.8]
X[2] = "apples": [0.9, 1.0, 1.1, 1.2]
```

### 7.2 投影（简化：假设所有 W 都是单位矩阵）

```
Q[0] = [0.1, 0.2]    (head 0)  Q[0] = [0.3, 0.4]  (head 1)
Q[1] = [0.5, 0.6]              Q[1] = [0.7, 0.8]
Q[2] = [0.9, 1.0]              Q[2] = [1.1, 1.2]

K/V 类似
```

### 7.3 算 attention scores（head 0）

```
score[0, 0] = Q[0]·K[0]/sqrt(2) = (0.1·0.1 + 0.2·0.2)/1.41 ≈ 0.035
score[0, 1] = 未来 → -∞
score[0, 2] = 未来 → -∞
softmax([0.035, -∞, -∞]) = [1.0, 0.0, 0.0]

score[1, 0] = (0.5·0.1 + 0.6·0.2)/1.41 ≈ 0.121
score[1, 1] = (0.5·0.5 + 0.6·0.6)/1.41 ≈ 0.432
score[1, 2] = -∞
softmax([0.121, 0.432, -∞]) ≈ [0.18, 0.82, 0.0]

score[2, 0] = (0.9·0.1 + 1.0·0.2)/1.41 ≈ 0.214
score[2, 1] = (0.9·0.5 + 1.0·0.6)/1.41 ≈ 0.745
score[2, 2] = (0.9·0.9 + 1.0·1.0)/1.41 ≈ 1.282
softmax([0.214, 0.745, 1.282]) ≈ [0.06, 0.27, 0.67]
```

**解读**：位置 2（"apples"）最关注位置 2 自己（67%），但也看位置 1（27%，即"like"）。

## 8. 代码实现

### 8.1 C 核心

详见 `c/attention.c`。

```c
void attention_forward(
    float* out,          // (B, T, C) 输出
    float* preatt,       // (B, NH, T, T) attention 分数（中间值，反向用）
    float* att,          // (B, NH, T, T) attention 概率
    const float* inp,    // (B, T, 3*C) 拼好的 Q/K/V
    int B, int T, int C, int NH
) {
    int C3 = C * 3;          // Q/K/V 拼起来
    int hs = C / NH;         // 每个 head 维度
    float scale = 1.0f / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                // Step 1: Q·K^T / sqrt(d)
                float maxval = -1e9f;
                for (int t2 = 0; t2 <= t; t2++) {  // causal: 只算 ≤ t
                    float val = 0;
                    for (int i = 0; i < hs; i++) {
                        val += inp[b*T*C3 + t*C3 + h*hs + i]
                             * inp[b*T*C3 + t2*C3 + h*hs + C + i];
                    }
                    val *= scale;
                    preatt[b*NH*T*T + h*T*T + t*T + t2] = val;
                    if (val > maxval) maxval = val;
                }

                // Step 2: softmax (subtract max for stability)
                float sum = 0;
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        float v = expf(preatt[b*NH*T*T + h*T*T + t*T + t2] - maxval);
                        att[b*NH*T*T + h*T*T + t*T + t2] = v;
                        sum += v;
                    } else {
                        att[b*NH*T*T + h*T*T + t*T + t2] = 0;
                    }
                }
                for (int t2 = 0; t2 <= t; t2++) {
                    att[b*NH*T*T + h*T*T + t*T + t2] /= sum;
                }

                // Step 3: weighted sum of V
                for (int i = 0; i < hs; i++) {
                    float o = 0;
                    for (int t2 = 0; t2 <= t; t2++) {
                        o += att[b*NH*T*T + h*T*T + t*T + t2]
                           * inp[b*T*C3 + t2*C3 + h*hs + 2*C + i];
                    }
                    out[b*T*C + t*C + h*hs + i] = o;
                }
            }
        }
    }
}
```

**4 个步骤的拆分**：
1. 算 Q·K^T，找最大值（数值稳定）
2. 算 exp(x - max)
3. softmax 归一化（顺便实现 causal mask）
4. 加权求和 V

### 8.2 Q/K/V 合并的实现

为什么不分别算 Q、K、V？**省参数 + 快**：

```c
// 一次 matmul 输出 3C 维
float* qkv = matmul(x, qkvw, qkvb, B, T, C, 3*C);

// 拆成 3 部分
// qkv[:, :, 0:C]     = Q
// qkv[:, :, C:2C]    = K
// qkv[:, :, 2C:3C]   = V
```

权重矩阵 `qkvw` 形状 `(3C, C)`，是三个独立矩阵拼起来。

## 9. 关键概念

| 概念 | 公式 | 含义 |
|---|---|---|
| Q/K/V 投影 | `Q = X @ W_q` 等 | 每个 token 三个角色 |
| Attention 分数 | `Q · K / sqrt(d)` | 相似度 |
| Causal mask | `j > t → -∞` | 不能看未来 |
| Softmax | `exp/sum` | 概率分布 |
| 加权求和 | `Σ att · V` | 取回信息 |
| 多头 | 拆 C 为 NH × d | 多角度学关系 |

## 10. 关键超参数

| 参数 | 值 | 作用 |
|---|---|---|
| `n_head` | 4 | 多少个头 |
| `head_size = n_embd / n_head` | 32 | 每个头维度 |
| `scale = 1/sqrt(head_size)` | 0.177 | 缩放防饱和 |
| 因果 mask | 上三角 -∞ | 单向信息流 |

## 11. 直觉总结

Attention 让每个 token "投票"：

```
位置 0 ("I"):       "我只关心我自己"        att = [1, 0, 0]
位置 1 ("like"):    "我看 'I' 和我自己"     att = [0.2, 0.8, 0]
位置 2 ("apples"):  "我看前面所有"          att = [0.1, 0.3, 0.6]
```

每个 token 的输出 = Σ 关注权重 × 那个 token 的 Value。

**这就是 Attention 的全部**。Attention 没有参数可学（Q/K/V 投影的参数是分开的），它只是一个加权平均的"软寻址"机制。

下一步：[04-MLP.md](04-MLP.md) — 看 Attention 之后每个 token 怎么"思考"。