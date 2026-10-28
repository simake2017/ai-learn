# Transformer 从零实现 - 教学项目

> **用 C 手写 Transformer + 数值微分算梯度 + Python 画图**，从最底层理解大语言模型。

## 🎯 项目目标

> 📌 **下一步做什么？看 [ROADMAP.md](ROADMAP.md)** —— 从"能转的发动机"到"能跑的车"。

让你**完整实现**一个微型 GPT（600K 参数），并理解：

1. ✅ Token id 怎么变成向量
2. ✅ Q/K/V 是什么，Attention 怎么算
3. ✅ MLP / 残差 / LayerNorm 的作用
4. ✅ 交叉熵损失怎么算
5. ✅ **梯度**是什么，**数值微分**怎么算
6. ✅ AdamW 怎么用梯度更新参数
7. ✅ 训练完后怎么"生成"下一个 token

## 📁 项目结构

```
transformer-zero/
├── c/                       # C 核心代码（~600行）
│   ├── transformer.h        # 数据结构 + API
│   ├── matrix.c             # matmul + residual
│   ├── activations.c        # GELU + softmax + crossentropy
│   ├── layernorm.c          # LayerNorm
│   ├── attention.c          # 多头因果 Attention
│   ├── model.c              # 完整 GPT forward
│   ├── dataloader.c         # 数据加载器
│   ├── autodiff.c           # 数值微分算梯度
│   ├── optimizer.c          # AdamW
│   └── main.c               # 训练 + 推理入口
├── python/                  # Python 胶水
│   ├── train.py             # 一键训练
│   ├── visualize.py         # 画 loss + attention
│   └── inference_demo.py    # "我喜欢苹果" 演示
├── data/                    # tinyshakespeare
├── docs/                    # 8 篇教学文档（建议先读！）
│   ├── 01-OVERVIEW.md
│   ├── 02-EMBEDDING.md
│   ├── 03-ATTENTION.md
│   ├── 04-MLP.md
│   ├── 05-FORWARD.md
│   ├── 06-AUTODIFF.md
│   ├── 07-OPTIMIZER.md
│   └── 08-INFERENCE.md
├── build.sh                 # 编译
├── run.sh                   # 训练 + 推理
└── README.md                # 本文件
```

## 🚀 快速开始

```bash
cd /home/magene/runtime/ai/ai-learn/transformer-zero

# 1. 编译
bash build.sh

# 2. 测试模型结构（小模型过拟合测试）
./build/transformer overfit

# 3. 训练（CPU fp32，~600K 参数，要跑很久）
./build/transformer train

# 4. 推理
./build/transformer generate
```

## 📚 学习路径

### 第一步：先读文档（30 分钟建立直觉）

按顺序读 `docs/`：

1. **01-OVERVIEW** — Transformer 整体架构和数据流
2. **02-EMBEDDING** — token id → 向量 + 位置编码
3. **03-ATTENTION** — Q/K/V + 因果掩码 + 多头
4. **04-MLP** — FFN + GELU + 残差
5. **05-FORWARD** — 完整 forward 流程
6. **06-AUTODIFF** — 数值微分原理
7. **07-OPTIMIZER** — AdamW 算法
8. **08-INFERENCE** — 自回归生成

### 第二步：看代码（2 小时）

读 `c/` 文件，按 `transformer.h → matrix.c → ... → main.c` 顺序。

### 第三步：跑起来（30 分钟）

```bash
bash build.sh
./build/transformer overfit
```

### 第四步：实验（持续）

- 修改 `c/model.c:config_init()`，试不同模型规模
- 改 `c/main.c` 的训练超参数
- 对比不同 n_layer、n_embd 的训练效果

## 🔧 技术决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 语言 | **C + Python 胶水** | C 看底层，Python 看图 |
| Token | **字符级** | vocab=65，最简 |
| 数据 | **tinyshakespeare** | 1MB，训练快 |
| 反向 | **数值微分** | 最容易理解"梯度" |
| 优化器 | **AdamW** | 工业标配 |
| 规模 | vocab=65, embd=128, 6 层 | CPU 可训练 |

## ⚠️ 已知限制

1. **数值微分极慢**：120万参数 × 2 次 forward = 每步 ~5分钟
   - 教学用小模型（28K 参数）：~5秒/步
   - 想要"真实训练"：用 nanoGPT
2. **中文不支持**：训练数据是英文 Shakespeare
3. **没有 KV Cache**：推理时每次重算整段 forward

## 🤔 关于"我喜欢苹果，它真好吃"

你想要的演示（中文）是真实的 GPT 任务，但本项目是字符级英文。

**要演示中文，需要**：
1. 中文语料（如宋词 1MB）
2. 字符级 vocab（4000+ 字）或 BPE vocab
3. 更大模型（>10M 参数）
4. 长时间训练（小时级）

**更简单的方案**：用 HuggingFace：
```python
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
```

## 📊 与其他项目对比

| 项目 | 实现 | 性能 | 教学性 |
|---|---|---|---|
| **本项目** | C + 数值微分 | ⭐ | ⭐⭐⭐⭐⭐ |
| nanoGPT | Python + autograd | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| llm.c | C/CUDA + 手写 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**建议**：
- 想理解算法 → 本项目 + nanoGPT
- 想做实际训练 → nanoGPT
- 想理解底层 → llm.c
- 想做生产 → HuggingFace + vLLM

## 🐛 故障排查

| 问题 | 解决 |
|---|---|
| 编译失败 | `apt install gcc` |
| 找不到数据 | `ls data/` 检查 train.bin |
| 训练超慢 | 用 `overfit` 模式（小模型快速测试）|
| loss 不下降 | 检查数值微分是否正确（对比解析梯度）|

## 📝 笔记

- 数据：`/home/magene/runtime/ai/ai-learn/transformer-zero/data/`
- 模型规模：`vocab=65, n_embd=128, n_layer=6, n_head=4` ≈ 120万参数
- 单步时间：~5分钟（数值微分）
- 1 epoch：~5小时（100步）

---

**祝你学得愉快！** 🎉