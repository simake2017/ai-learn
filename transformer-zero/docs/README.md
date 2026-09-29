# 从零实现 Transformer 教学项目

> 用 C 手写前向 + 数值微分 + AdamW + Python 胶水 + 可视化，帮你**真正理解** Transformer。

## 项目目标

让一个完全没接触过 LLM 的人读完代码 + 跑一遍实验后，能回答：

1. Token id 怎么变成向量？
2. Q/K/V 到底是什么？Attention 在算什么？
3. MLP / 残差 / LayerNorm 都是干什么的？
4. 损失函数在量什么？
5. 梯度是什么？怎么算？
6. AdamW 和 SGD 的区别？
7. 模型怎么"生成"下一个 token？

## 文件结构

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
│   └── inference_demo.py # "我喜欢苹果" 例子
├── data/               # tinyshakespeare
├── docs/               # 教学文档
│   ├── 01-OVERVIEW.md
│   ├── 02-EMBEDDING.md
│   ├── 03-ATTENTION.md
│   ├── 04-MLP.md
│   ├── 05-FORWARD.md
│   ├── 06-AUTODIFF.md
│   ├── 07-OPTIMIZER.md
│   └── 08-INFERENCE.md
├── build.sh            # 一键编译
└── run.sh              # 一键训练
```

## 推荐阅读顺序

### 第一步：先读文档（30 分钟）

按顺序读 `01 → 08`，建立直觉。

### 第二步：看代码（2 小时）

```
c/transformer.h     # 数据结构（最先看）
c/matrix.c          # 最简单
c/activations.c     # GELU, softmax
c/layernorm.c       # 归一化
c/attention.c       # 核心！
c/model.c           # 拼装
c/autodiff.c        # 数值微分（教学重点）
c/optimizer.c       # AdamW
c/main.c            # 训练循环
```

### 第三步：跑起来（30 分钟）

```bash
cd /home/magene/runtime/ai/ai-learn/transformer-zero
bash build.sh       # 编译
bash run.sh train   # 训练（CPU 几十分钟）
```

### 第四步：推理（10 分钟）

```bash
python python/inference_demo.py
# 输入: "I like apples, they are"
# 输出: 模型生成的字符
```

### 第五步：可视化（10 分钟）

```bash
python python/visualize_attention.py
# 输出: attention 热力图 PNG
```

## 核心算法概览

### Transformer Block

```
输入 x (B, T, C)
  ↓
[ LayerNorm 1 ]
  ↓
[ Multi-Head Causal Self-Attention ]
  ↓
[ + 残差（加回原 x）]
  ↓
[ LayerNorm 2 ]
  ↓
[ MLP: C → 4C → C，GELU 激活 ]
  ↓
[ + 残差 ]
  ↓
输出 y (B, T, C)
```

### 完整 GPT

```
token ids (B, T)
  ↓
[ Embedding (token + position) ]
  ↓
[ n_layer × Transformer Block ]
  ↓
[ Final LayerNorm ]
  ↓
[ LM Head: C → V ]
  ↓
logits (B, T, V)
```

### 训练循环

```python
for step in range(num_steps):
    batch = get_batch()               # 取数据
    logits, loss = forward(batch)     # 算 loss
    grads = numerical_grad(loss)      # 数值微分
    params = adamw(params, grads)     # 更新
```

### 推理

```python
while not done:
    logits = forward(prompt)         # 跑整段 forward
    probs = softmax(logits[-1])       # 取最后一个位置
    next_token = sample(probs)        # 采样
    prompt.append(next_token)
```

## 关键技术决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 语言 | **C + Python 胶水** | C 看底层，Python 看图 |
| Token | **字符级** | vocab=65，最简 |
| 数据 | **tinyshakespeare** | 1MB，训练快 |
| 反向 | **数值微分** | 最容易理解梯度 |
| 优化器 | **AdamW** | 工业标配 |
| 模型规模 | **vocab=65, embd=128, 6层, 4头** | CPU 30分钟能训练出可读结果 |

## 不做的事（避免分心）

- ❌ **Flash Attention**：用朴素版本，看清楚
- ❌ **KV Cache**：教学不需要性能
- ❌ **混合精度**：fp32 最简单
- ❌ **多卡训练**：单 GPU/CPU 足够
- ❌ **学习率调度**：固定 lr
- ❌ **复杂 tokenizer**：字符级

## 故障排查

| 问题 | 原因 | 解决 |
|---|---|---|
| 编译报错 | 缺依赖 | `apt install gcc make` |
| 训练 loss 不下降 | 数值微分 ε 不对 | 调整 eps=1e-3 |
| 训练太慢 | 参数太多 | 减小 n_embd/n_layer |
| 推理乱码 | vocab 不匹配 | 确认 meta.pkl 对应 |

## 与 llm_c / nanoGPT 关系

| 项目 | 语言 | 反向 | 性能 |
|---|---|---|---|
| **本项目** | C + Python | **数值微分** | 慢但清晰 |
| llm.c | C/CUDA | 手写解析梯度 | 快 |
| nanoGPT | Python (PyTorch) | autograd | 中 |

读完本项目后，看 llm.c / nanoGPT 会"恍然大悟"。

---

## 立即开始

```bash
cd /home/magene/runtime/ai/ai-learn/transformer-zero
ls
cat README.md
```