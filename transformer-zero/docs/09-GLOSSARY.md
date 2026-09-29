# 09 名词解释大全 + 应用场景 + 具体参数

> **从最基础开始**，每个名词都用人话解释 + 代码例子 + 应用场景 + 我们的具体参数值。

---

## 阅读指南

如果你看到一个名词不懂，按 Ctrl+F 搜索：

| 你 | | 推荐先看 |
|---|---|---|
| 完全没听过这些词 | → | 从第 1 节"Embedding"开始顺序读 |
| 有 ML 基础但没做过 Transformer | → | 直接看第 4-7 节（核心算法）|
| 看代码时遇到变量不懂 | → | 看第 9-12 节（张量形状）|
| 想知道实际能做什么 | → | 每个名词都有"应用场景"小节 |

---

## 1. Embedding（嵌入 / 向量化）

### 一句话解释

**把"东西"（token、词、句子、图片）变成一串数字**（向量），让计算机能算。

### 类比

```
人认识世界:
  苹果 → 脑子里浮现"红色、圆形、甜、水果"
  香蕉 → 浮现"黄色、长条形、甜、水果"

AI 认识世界:
  苹果 → [0.3, -0.2, 0.8, 0.1, ...,0.5]  (128 个数字)
  香蕉 → [0.4, -0.1, 0.7, 0.2, ..., 0.6]  (和苹果很像)
  飞机 → [-0.7, 0.9, -0.5, -0.3, ..., 0.1] (和苹果很远)
```

### 在我们代码里

```c
// c/model.c: embedding_forward
// 把 token id 变成 128 维向量
out[(b * T + t) * C + c] = wte[token_id * C + c] + wpe[t * C + c];
```

### 应用场景

- 🎯 **推荐系统**：用户和商品都转成向量，算相似度
- 🎯 **搜索**：query 和文档都向量化
- 🎯 **图像识别**：图片转成向量（CNN 最后一层）
- 🎯 **语音识别**：音频转成向量

### 具体参数

| 参数 | 含义 | 我们的值 |
|---|---|---|
| `vocab_size` | 词表大小 | 65 |
| `n_embd` | 向量维度 | 128 |

---

## 2. WTE（Word Token Embedding / 词嵌入）

### 一句话解释

**一个查找表**，把每个 token id 映射成一个向量。

### 类比

```
人类字典:
  苹果 → "red, round, sweet, fruit" (描述)

AI 的"字典":
  wte[5] = [0.3, -0.2, 0.8, ...,0.5]  (128 个数字)
```

### 怎么造这个字典？**训练出来的**！

```c
// 一开始（随机）:
wte[5] = [0.001, -0.002, 0.003, ..., 0.005]  ← 随机

// 训练后（学到了）:
wte[5] = [0.345, -0.123, 0.789, ...,0.234]  ← 有意义
```

### 应用场景

- 🎯 **任何 NLP 任务的基础**（翻译、情感、问答...）
- 🎯 **情感分析**：把词向量相加 = 句子向量
- 🎯 **词性标注**：用词向量做特征
- 🎯 **命名实体识别**：找向量模式

### 具体参数

| 参数 | 含义 | 我们的值 |
|---|---|---|
| `V` | vocab size | 65 |
| `C` | embed dim | 128 |
| 总元素 | `V × C` | 8,320 |

---

## 3. WPE（Word Position Embedding / 位置嵌入）

### 一句话解释

**另一个查找表**，把"位置编号"（0, 1, 2, ..., T-1）映射成向量。

### 为什么要它？

```
没有位置信息:
  "狗咬人" 和 "人咬狗" → token embedding 一样（模型分不清顺序）

有位置信息:
  "狗" 在位置 0 → 加 pos 0 向量
  "咬" 在位置 1 → 加 pos 1 向量  ← 这些不一样
  "人" 在位置 2 → 加 pos 2 向量
```

### 怎么造？也是训练！

```c
wpe[0]  = [random 128个数字]  ← 位置 0 的"角色"
wpe[1]  = [random 128个数字]
...
wpe[63] = [random 128个数字]
// 训练后，相邻位置的 wpe 会接近（因为角色类似）
```

### 对比固定 vs 可学习

| | **sin/cos 编码（固定）** | **可学习（我们用）** |
|---|---|---|
| 生成 | 公式 `sin(pos/10000^(2i/d))` | 训练学 |
| 公式 | 有 | 没有 |
| 优点 | 无参数 | 效果略好 |
| 缺点 | 训练后不能改 | 需要训练数据 |

### 应用场景

- 🎯 **所有 Transformer** 都需要
- 🎯 序列长度不同的任务（句子长度可变）

### 具体参数

| 参数 | 含义 | 我们的值 |
|---|---|---|
| `T` | 最大序列长度 | 64 |
| `C` | 维度 | 128 |
| 总元素 | `T × C` | 8,192 |

---

## 4. Q, K, V（Query, Key, Value）

### 一句话解释

把每个 token 的向量**复制成三份**，扮演不同角色：
- **Q**（查询）：我在找什么信息？
- **K**（键）：我有什么信息？（"我的标签"）
- **V**（值）：具体内容是什么？

### 类比

```
你在图书馆找"机器学习"的书:
  你的需求（Q）: "我要机器学习的书"
  书的标签（K）: 每本书侧面的标签
  书的内容（V）: 书里面的实际内容

 步骤 1: Q 和每本书的 K 比相似度
  步骤 2: 相似度高的书 → 给高权重
  步骤 3: 用权重把 V 加起来 → "找到的资料"
```

### 在代码里

```c
// c/model.c 的 block_forward
// 一次 matmul 输出 3C 维
matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
// 然后拆成 3 部分
// qkv[:, :, 0:C]    = Q  (查询)
// qkv[:, :, C:2C]   = K  (键)
// qkv[:, :, 2C:3C]  = V  (值)
```

### 应用场景

- 🎯 **搜索/推荐**：Q 是用户，K/V 是商品
- 🎯 **问答系统**：Q 是问题，K/V 是文档
- 🎯 **图像检索**：Q 是查询图，K/V 是数据库

### 具体参数

| 参数 | 含义 | 我们的值 |
|---|---|---|
| Q/K/V 维度 | 单头维度 d | 32 (= 128/4) |
| 多头拼接 | 总维度 | 128 (= 32 × 4) |

---

## 5. Self-Attention / Attention（自注意力 / 注意力）

### 一句话解释

**让每个 token 决定"关注其他哪些 token"**，加权汇总信息。

### 类比

```
班级讨论：
  老师问"谁负责这件事？"
  A 同学举手: "我" (高权重)
  B 同学: 低权重
  C 同学: 中权重

  A 同学的回答被"重点听"
```

### 数学公式

```python
score[i, j] = (Q[i] · K[j]) / sqrt(d)    # i 关注 j 的分数
att[i, j]   = softmax(score[i, :])         # 归一化（和为 1）
out[i]      = sum_j att[i, j] * V[j]       # 加权求和
```

### 在代码里

```c
// c/attention.c: attention_forward
// 4 个步骤（详见 attention.c 注释）
// 1. Q·K^T / sqrt(d) + max for stability
// 2. exp(x - max)
// 3. normalize (softmax)
// 4. weighted sum of V
```

### 应用场景

- 🎯 **机器翻译**：源语言 token 关注目标语言 token
- 🎯 **文本摘要**：关注关键句子
- 🎯 **语音识别**：关注相关音频帧

### 具体参数

| 参数 | 含义 | 我们的值 |
|---|---|---|
| `n_head` | 多头注意力头数 | 4 |
| `head_size = d` | 单头维度 | 32 (= 128/4) |
| `scale` | 缩放因子 | 1/√32 ≈ 0.177 |

---

## 6. Causal Mask（因果掩码）

### 一句话解释

**限制 attention 只能看到当前位置和之前的 token**，不能看未来。

### 为什么需要？

```
语言模型是"自回归"的:
  生成 "I like apples" 的时候:
  - 生成 "I" 时，只能看 "I"（之前没有）
  - 生成 "like" 时，能看 "I"和 "like"
  - 生成 "apples" 时，能看 "I", "like", "apples"

不能作弊看未来（训练时不知道答案）

mask 矩阵:
       I  like  apples
  I   [ ✓   ✗    ✗ ]
  like[ ✓   ✓    ✗ ]
  apples[✓   ✓    ✓ ]
```

### 在代码里

```c
// c/attention.c
for (int t2 = 0; t2 <= t; t2++) {  // ← 只算 ≤ t 的位置
    // ... 计算 attention
}
// 其他位置设 0（在 softmax 后）
```

### 应用场景

- 🎯 **所有生成式语言模型**（GPT 类）
- 🎯 **音乐生成**：不能看后面音符
- 🎯 **代码生成**：不能看后面代码

### 具体参数

| 参数 | 含义 | 我们的值 |
|---|---|---|
| mask 形状 | (T, T) | (64, 64) |
| 类型 | 上三角 -∞ | |

---

## 7. Multi-Head Attention（多头注意力）

### 一句话解释

**同时做多组（4 头）self-attention**，每组学不同的"关注模式"，最后拼起来。

### 类比

```
读一段话，找不同方面的关系：
  头 1: 关注"主谓一致"（动词看主语）
  头 2: 关注"指代关系"（代词看名词）
  头 3: 关注"长距离依赖"（开头看结尾）
  头 4: 关注"局部语法"（相邻词）
```

### 在代码里

```c
// c/attention.c
int hs = C / NH;  // hs = 128/4 = 32
for (int h = 0; h < NH; h++) {  // 4 个头
    // 每个头独立算 attention
    // Q[h], K[h], V[h] 来自同一输入的不同子空间
    // 输出形状 (B, T, hs)
}
// 4 个头拼起来 → (B, T, C) = (B, T, 128)
```

### 应用场景

- 🎯 **所有 Transformer 类模型**
- 🎯 **多视角分析文本**

### 具体参数

| 参数 | 含义 | 我们的值 |
|---|---|---|
| `n_head` | 头数 | 4 |
| `head_size` | 每头维度 | 32 |
| 总维度 | 拼接后 | 128 |

---

## 8. LayerNorm（层归一化）

### 一句话解释

**对每个 token 的128 维向量单独归一化**：均值 0、方差 1，让训练稳定。

### 为什么需要？

```
没有归一化:
  token 1 的向量: [100, -50, 200, ...]     ← 数值爆炸
  token 2 的向量: [0.001, 0.002, 0.003]   ← 数值消失

  → 后续层不知道该怎么处理

有归一化:
  token 1: [-0.5, 0.3, -0.2, ..., 0.8]    ← 都接近 0
  token 2: [0.1, -0.7, 0.4, ..., -0.3]   ← 都接近 0
  → 后续层处理得很舒服
```

### 公式

```python
mean = x.mean()
var  = ((x - mean)**2).mean()
rstd = 1 / sqrt(var + eps)
out  = (x - mean) * rstd * gamma + beta
```

`gamma, beta` 是可学习参数。

### vs BatchNorm

| | LayerNorm | BatchNorm |
|---|---|---|
| 归一化维度 | 沿特征维（C 维）| 沿 batch 维 |
| 适用 | 序列长度可变 | 固定 batch |
| Transformer 标配 | ✅ | ❌ |

### 应用场景

- 🎯 **所有 Transformer**（GPT, BERT, LLaMA...）
- 🎯 **深度网络训练稳定性**

### 具体参数

| 参数 | 含义 | 我们的值 |
|---|---|---|
| `eps` | 数值稳定 | 1e-5 |
| `gamma` 形状 | (C,) | (128,) |
| `beta` 形状 | (C,) | (128,) |

---

## 9. MLP / FFN（前馈网络 / 多层感知机）

### 一句话解释

**两层全连接**：升维（128 → 512）→ 激活 → 降维（512 → 128）。

### 结构

```
输入 (128 维)
   ↓
Linear 128 → 512  (4 倍升维)
   ↓
GELU 激活
   ↓
Linear 512 → 128  (降回原维度)
   ↓
输出 (128 维)
```

### 为什么是 4 倍？

```
Transformer 标配升到 4 倍:
  2x: 表达空间不够，效果差
 4x: 最佳（论文实验得出）
  8x: 收益递减

直觉: 先扩展"思维空间"再做非线性
```

### 在代码里

```c
// c/model.c: block_forward 的 MLP 部分
matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
gelu_forward(l_fch, l_fch, B*T*4*C);
matmul_forward(l_x, l_fch, l_fcprojw, l_fcprojb, B, T, 4*C, C);
```

### 应用场景

- 🎯 **几乎所有深度学习模型都有 MLP**
- 🎯 **"思考"每个 token**

### 具体参数

| 参数 | 含义 | 我们的值 |
|---|---|---|
| `n_embd` | 输入维度 | 128 |
| `4 * n_embd` | 升维后 | 512 |
| 权重1 形状 | (4C, C) | (512, 128) |
| 权重2 形状 | (C, 4C) | (128, 512) |

---

## 10. GELU（高斯误差线性单元）

### 一句话解释

**激活函数**，让神经网络能拟合非线性关系。

### 公式

```python
GELU(x) = x * Φ(x)
       ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

`Φ(x)` 是标准正态 CDF。

### vs ReLU

```
        GELU              ReLU
    ╱‾‾‾‾‾          ╱‾‾‾‾
   ╱              ╱
──┼──────    ──+──────
   │              │
   ╲             (硬截断到0)
    ╲
```

- **GELU**：平滑，对负值保留少量信号
- **ReLU**：硬截断，负值直接变 0

### 在代码里

```c
// c/activations.c
float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x*x*x)));
}
```

### 应用场景

- 🎯 **Transformer 标配**（GPT, BERT, LLaMA）
- 🎯 **任何需要平滑激活的地方**

---

## 11. Residual Connection（残差连接）

### 一句话解释

**把"子层输入"直接加到"子层输出"**：`y = x + sublayer(x)`。

### 为什么？

```
没有残差 (深层网络):
  输入 → 层1 → 层2 → ... → 层 12 → 输出
  ∂Loss/∂输入 = 很多矩阵相乘 → 容易消失（梯度变小）

有残差:
  y = x + sublayer(x)
  ∂y/∂x = 1 + ∂sublayer/∂x  ← 至少是 1，梯度不会消失
```

### 在代码里

```c
// c/model.c: block_forward
residual_forward(l_residual2, l_x, l_att_out, B*T*C);
residual_forward(l_x, l_residual2, l_fcproj, B*T*C);
```

### 应用场景

- 🎯 **所有深度网络**（CNN, ResNet, Transformer）
- 🎯 **训练几十/几百层网络**的必需品

---

## 12. Softmax

### 一句话解释

**把一组任意实数转成"概率分布"**（和为 1，每个 ≥ 0）。

### 公式

```python
softmax(x_i) = exp(x_i) / sum(exp(x_j))

#)
```

数值稳定版：`softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`

### 在代码里

```c
// c/activations.c
// 减最大值防 exp 溢出
float maxval = -1e30f;
for (int v = 0; v < V; v++)
    if (logits[v] > maxval) maxval = logits[v];

float sum = 0;
for (int v = 0; v < V; v++) {
    probs[v] = expf(logits[v] - maxval);
    sum += probs[v];
}
for (int v = 0; v < V; v++)
    probs[v] /= sum;
```

### 应用场景

- 🎯 **分类任务最后一步**
- 🎯 **多分类 / 多选项问题**
- 🎯 **语言模型的概率输出**

---

## 13. Cross-Entropy Loss（交叉熵损失）

### 一句话解释

**衡量"模型预测的概率分布"和"真实标签"的差距**。越小越好。

### 公式

```python
loss = -log(P(true_class))
```

直觉：
- 真实 token 是 5，模型 P(5) = 0.99 → loss = 0.01（很好）
- 模型 P(5) = 0.5 → loss = 0.69（一般）
- 模型 P(5) = 0.01 → loss = 4.6（很差）

### 在代码里

```c
// c/activations.c: crossentropy_forward
float p = probs_bt[target];
if (p < 1e-30f) p = 1e-30f;  // 防 log(0)
losses[b * T + t] = -logf(p);
```

### 应用场景

- 🎯 **所有分类任务**（图片分类、文本分类...）
- 🎯 **语言模型的训练目标**

---

## 14. AdamW 优化器

### 一句话解释

**更新参数的算法**，比 SGD 聪明，比 Adam 更适合 Transformer。

### 完整公式

```python
m = β1 * m + (1 - β1) * g       # 一阶矩（梯度的平均）
v = β2 * v + (1 - β2) * g²      # 二阶矩（梯度平方的平均）
m_hat = m / (1 - β1^t)            # 偏差修正
v_hat = v / (1 - β2^t)

θ ← θ - lr * (m_hat / (√v̂ + ε) + λ * θ)
```

### 为什么比 SGD 好？

| | SGD | AdamW |
|---|---|---|
| 学习率 | 固定 | 每个参数自适应 |
| 动量 | 无 | 有 |
| L2 正则 | 与 lr 耦合 | 解耦 |
| 适用 | 凸问题 | 神经网络 |

### 在代码里

```c
// c/optimizer.c
opt->lr = 1e-3f;
opt->beta1 = 0.9f;
opt->beta2 = 0.999f;
opt->eps = 1e-8f;
opt->weight_decay = 0.01f;
```

### 应用场景

- 🎯 **所有 Transformer 训练**
- 🎯 **几乎所有现代深度学习**

---

## 15. 数值微分（Numerical Differentiation）

### 一句话解释

**用"做两次实验"算梯度**：

```
∂L/∂θ ≈ (L(θ + ε) - L(θ - ε)) / (2ε)
```

### 对比三种算梯度方法

| | 数值微分 | 解析微分（手写） | 自动微分（autograd）|
|---|---|---|---|
| 速度 | 慢（2N 次 forward）| 快 | 中 |
| 代码量 | 5 行 | 600 行 | 0 行（库） |
| 精度 | 受 ε 限制 | 机器精度 | 机器精度 |
| 教学性 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

### 在代码里

```c
// c/autodiff.c
for (int i = 0; i < num_params; i++) {
    float saved = params[i];
    
    params[i] = saved + eps;
    gpt_forward(model, inputs, targets);
    float L_plus = model->mean_loss;
    
    params[i] = saved - eps;
    gpt_forward(model, inputs, targets);
    float L_minus = model->mean_loss;
    
    grads[i] = (L_plus - L_minus) / (2 * eps);
    params[i] = saved;
}
```

### 应用场景

- 🎯 **教学**（最直观）
- 🎯 **单元测试**（验证解析梯度是否正确）
- ❌ 生产训练（太慢）

---

## 16. 自回归生成（Autoregressive Generation）

### 一句话解释

**一个 token 一个 token 地生成文本**。

### 流程

```
输入: "I like"
   ↓ 模型
  预测下一个 token (概率分布)
   ↓ 采样（如选 "apples"）
新输入: "I like apples"
   ↓ 模型（重新跑整段！）
  预测下一个 token
   ↓ 采样
新输入: "I like apples,"
   ↓ ... 直到 <EOT>
```

### 为什么叫"自回归"？

```
自 (self) + 回归 (regression):
  用自己之前的输出，作为下一步的输入
```

### 在代码里

```c
// c/main.c: run_generate
for (int step = 0; step < max_new_tokens; step++) {
    gpt_forward(&model, tokens, NULL);  // 跑整段 forward
    float* probs = last_position_probs;
    int next = sample_token(probs, vocab_size, temperature);
    tokens[prompt_len + step] = next;   // 加到末尾
}
```

### 应用场景

- 🎯 **ChatGPT** 对话生成
- 🎯 **翻译**
- 🎯 **代码补全**（GitHub Copilot）
- 🎯 **写作辅助**

---

## 17. Temperature（温度）

### 一句话解释

**控制生成文本的"随机程度"**：
- T → 0：稳定、可预测（贪心）
- T = 1：标准
- T → ∞：完全随机

### 公式

```python
logits = logits / T
probs = softmax(logits)
```

### 直观

```
T = 0.1: [0.99, 0.005, 0.005]  ← 几乎确定选第一个
T = 1.0: [0.5,  0.3,   0.2]    ← 标准
T = 2.0: [0.33, 0.33,  0.34]   ← 均匀
```

### 在代码里

```c
// c/main.c: sample_token
if (temperature <= 0.0f) {
    // Greedy（贪心）
} else {
    // 按概率采样
}
```

### 应用场景

- 🎯 **代码生成**：低 T（要确定性）
- 🎯 **创意写作**：高 T（要多样性）
- 🎯 **客服对话**：中 T

---

## 18. BPE Tokenizer（字节对编码）

### 一句话解释

**把文本切成"子词"**（不是字符也不是词，是中间粒度）。

### 例子

```
英文 "unhappiness":
  字符级: ['u', 'n', 'h', 'a', 'p', 'p', 'i', 'n', 'e', 's', 's']
  词级: ['unhappiness']
  BPE:  ['un', 'happiness']  ← 平衡

中文 "我喜欢苹果":
  字符级: ['我', '喜', '欢', '苹', '果']
  词级: ['我', '喜欢', '苹果']
  BPE:  ['我', '喜欢', '苹果']  (中文常用词级)
```

### 在代码里

我们**不用 BPE**，用更简单的字符级（vocab=65）。

### 应用场景

- 🎯 **GPT-2/3/4** 用 BPE
- 🎯 **LLaMA** 用 SentencePiece
- 🎯 **中文** 常用词级或字符级

---

## 19. Autoregressive vs Autoencoding

### 自回归（GPT 类）

```
任务: 给定前文，预测下一个
方向: 单向（左 → 右）
例: GPT, LLaMA, Claude
```

### 自编码（BERT 类）

```
任务: 给定上下文，预测被 mask 的词
方向: 双向
例: BERT, RoBERTa
```

### 在我们项目里

我们是**自回归**（GPT 类），所以用 causal mask。

---

## 20. Pre-Norm vs Post-Norm

### Pre-Norm（我们用）

```python
x = x + sublayer(layer_norm(x))    # 先 norm 再算子
```

### Post-Norm（原始 Transformer）

```python
x = layer_norm(x + sublayer(x))    # 算完再 norm
```

### 对比

| | Pre-Norm | Post-Norm |
|---|---|---|
| 训练稳定性 | ✅ 深层稳定 | ❌ 深层要小心 |
| 最终性能 | 略差 | 略好 |
| GPT-2 用法 | ✅ | ❌ |

---

## 21. Vocab Size（词表大小）

### 一句话解释

**模型认识多少个不同的 token**。

### 对比

| | 字符级 | BPE 级 | 词级 |
|---|---|---|---|
| vocab | 几十 | 几万 | 几十万 |
| 优点 | 简单 | 平衡 | 直观 |
| 缺点 | 序列长 | 复杂 | OOV 问题 |

### 在我们代码里

```c
c->vocab_size = 65;  // 字符级
```

---

## 22. Block Size（序列长度）

### 一句话解释

**模型一次看多长的序列**。

### 在我们代码里

```c
c->block_size = 64;  // 最长 64 个 token
```

### 应用

- 短（64-256）：小模型、教学
- 中（512-2048）：GPT-2 / GPT-3
- 长（4096+）：GPT-4、LLaMA 2

---

## 23. n_layer（层数）

### 一句话解释

**多少个 Transformer Block 堆叠**。

### 在我们代码里

```c
c->n_layer = 6;  // 6 层 Block
```

### 对比

| 模型 | 层数 |
|---|---|
| 我们的教学模型 | 6 |
| GPT-2 small | 12 |
| GPT-2 large | 36 |
| GPT-3 | 96 |
| LLaMA-7B | 32 |

---

## 24. n_head（注意力头数）

### 一句话解释

**Multi-Head Attention 有多少个头**。

### 在我们代码里

```c
c->n_head = 4;
c->n_embd / n_head = 128 / 4 = 32  // 每头 32 维
```

---

## 25. n_embd（嵌入维度）

### 一句话解释

**每个 token 向量有多少维**。

### 在我们代码里

```c
c->n_embd = 128;
```

### 对比

| 模型 | n_embd |
|---|---|
| 我们的 | 128 |
| GPT-2 small | 768 |
| GPT-3 175B | 12288 |

---

## 26. 完整参数清单

我们模型（vocab=65, n_embd=128, n_layer=6, n_head=4）的参数：

| 张量 | 形状 | 元素数 |
|---|---|---|
| wte | (65, 128) | 8,320 |
| wpe | (64, 128) | 8,192 |
| 每层（×6 层）| | |
| ln1w/b | (128) | 128 × 2 |
| qkvw | (3×128, 128) | 49,152 |
| qkvb | (3×128) | 384 |
| projw | (128, 128) | 16,384 |
| projb | (128) | 128 |
| ln2w/b | (128) | 128 × 2 |
| fcw | (4×128, 128) | 65,536 |
| fcb | (4×128) | 512 |
| fcprojw | (128, 4×128) | 65,536 |
| fcprojb | (128) | 128 |
| 每层小计 | | 198,144 |
| 6 层 | | 1,188,864 |
| lnfw/lnfb | (128) | 256 |
| **总计** | | **1,205,632** |

约 **120万参数**。

---

## 27. 张量形状速查

训练时形状变化：

| 步骤 | 张量 | 形状 |
|---|---|---|
| 输入 | idx | (B=4, T=64) |
| Embedding | x | (B=4, T=64, C=128) |
| LayerNorm | ln1 | (B=4, T=64, C=128) |
| QKV | qkv | (B=4, T=64, 3C=384) |
| Attention scores | preatt | (B=4, NH=4, T=64, T=64) |
| Attention probs | att | (B=4, NH=4, T=64, T=64) |
| Attention out | att_out | (B=4, T=64, C=128) |
| MLP intermediate | fch | (B=4, T=64, 4C=512) |
| After GELU | fch_gelu | (B=4, T=64, 4C=512) |
| After 6 Blocks | x | (B=4, T=64, C=128) |
| Final LN | lnf | (B=4, T=64, C=128) |
| LM Head | logits | (B=4, T=64, V=65) |

---

## 28. 应用场景总结

按行业分：

### 内容创作

- ✅ 文章写作（GPT-4, Claude）
- ✅ 营销文案
- ✅ 诗歌、小说

### 编程

- ✅ 代码补全（Copilot）
- ✅ 代码生成（Devin）
- ✅ Bug 修复

### 教育

- ✅ 题目讲解
- ✅ 个性化辅导
- ✅ 语言学习

### 客服/对话

- ✅ 智能客服
- ✅ 心理咨询
- ✅ 角色扮演

### 翻译/多语言

- ✅ 100+ 语言翻译
- ✅ 跨语言检索

### 数据分析

- ✅ 文本分类（情感、垃圾）
- ✅ 信息抽取
- ✅ 实体识别

### 科研

- ✅ 文献总结
- ✅ 假设生成
- ✅ 代码辅助

---

## 29. 学习路径

如果你想**真正搞懂**这些概念：

1. 📖 读 `docs/01-OVERVIEW.md`（整体）
2. 📖 读 `docs/02-EMBEDDING.md`（向量基础）
3. 📖 读 `docs/03-ATTENTION.md`（核心算法）
4. 📖 读 `docs/04-MLP.md`（另一个核心）
5. 📖 读 `docs/05-FORWARD.md`（拼起来）
6. 📖 读 `docs/06-AUTODIFF.md`（怎么学）
7. 📖 读 `docs/07-OPTIMIZER.md`（AdamW）
8. 📖 读 `docs/08-INFERENCE.md`（怎么用）
9. 📖 读本文 `docs/09-GLOSSARY.md`（查名词）

---

## 30. 推荐外部资源

| 资源 | 说明 |
|---|---|
| 3Blue1Brown 神经网络视频 | 直观理解 |
| Andrej Karpathy "Let's build GPT" | 手把手实现 |
| "Attention Is All You Need" 论文 | 原始论文 |
| nanoGPT 仓库 | PyTorch 简洁实现 |
| llm.c 仓库 | C 实现 |

---

**祝你学得开心！** 🎓