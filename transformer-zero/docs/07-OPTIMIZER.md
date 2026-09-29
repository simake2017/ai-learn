# 07 Optimizer：AdamW 算法

> 拿到梯度后，怎么更新参数？SGD 太简单，Adam 是工业标配，AdamW 是 Adam 的"修正版"。

## 1. 从最朴素的 SGD 开始

**SGD（Stochastic Gradient Descent）**：

```
θ ← θ - lr × ∂L/∂θ
```

直觉：loss 减小方向的反方向走一步，步长 `lr`。

**问题**：
1. 同一个 lr 不适用于所有参数（有的参数梯度大、有的梯度小）
2. 容易陷入局部最优
3. 在鞍点附近卡住

## 2. Momentum：动量

**加惯性**：

```
v ← μ × v + ∂L/∂θ      (累积速度)
θ ← θ - lr × v          (按速度更新)
```

- `v` 类似"小球的速度"
- `μ = 0.9` 是摩擦系数
- 效果：穿越鞍点、减少震荡

## 3. Adam：自适应学习率

**Adam = Momentum + RMSProp**：

```
m ← β1 × m + (1-β1) × g          (一阶矩：梯度均值)
v ← β2 × v + (1-β2) × g²         (二阶矩：梯度平方均值)

m̂ = m / (1 - β1^t)                (偏差修正)
v̂ = v / (1 - β2^t)

θ ← θ - lr × m̂ / (√v̂ + ε)
```

**核心思想**：
- 每个参数**自适应**学习率（梯度小的参数用大学习率，梯度大的用小）
- 加动量（穿越鞍点）
- 偏差修正（前几步 m, v 偏小）

## 4. AdamW：解耦的权重衰减

**Adam 的问题**：L2 正则与自适应学习率耦合，效果不好。

**AdamW 的改进**：把 L2 正则直接加到参数更新上：

```
m ← β1 × m + (1-β1) × g
v ← β2 × v + (1-β2) × g²

m̂ = m / (1 - β1^t)
v̂ = v / (1 - β2^t)

θ ← θ - lr × (m̂ / (√v̂ + ε) + λ × θ)
                            ↑ L2 项，与 Adam 完全解耦
```

**论文**：Loshchilov & Hutter, 2019 *"Decoupled Weight Decay Regularization"*

## 5. 超参数详解

| 参数 | 符号 | 典型值 | 含义 |
|---|---|---|---|
| 学习率 | `lr` | 1e-3 ~ 5e-4 | 步长 |
| β1 | `beta1` | 0.9 | 一阶矩衰减 |
| β2 | `beta2` | 0.999 | 二阶矩衰减 |
| ε | `eps` | 1e-8 | 数值稳定 |
| L2 系数 | `weight_decay` | 0.1 | 权重衰减 |

### 5.1 β1 控制什么？

`m` 是梯度的指数移动平均，β1 越大记忆越长：
- β1 = 0.9：最近 ~10 步梯度的平均
- β1 = 0.99：最近 ~100 步

### 5.2 β2 控制什么？

`v` 是梯度平方的移动平均，决定每个参数的"自适应缩放"：
- 梯度方差大的参数 → √v 大 → 分母大 → 学习率小
- 梯度方差小的参数 → √v 小 → 分母小 → 学习率大

直觉：经常变动的参数小步走，稳定的参数大步走。

### 5.3 偏差修正为什么需要？

`m, v` 初始化为 0，前几步会偏向 0：

```
t=1: m = (1-β1) × g₁ = 0.1 × g₁  (实际期望是 g₁)
t=2: m = 0.1 × g₂ + 0.9 × 0.1 × g₁  (还偏小)
```

偏差修正：`m̂ = m / (1 - β1^t)`
- t=1: m̂ = m / (1-0.9) = m / 0.1 = 10m ≈ g₁ ✓

## 6. 数值例子

设 `θ = 0.5, g = 0.1, lr = 0.01, β1=0.9, β2=0.999, λ=0.01, t=10`。

```
初始: m=0, v=0

Step 10:
m = 0.9 × 0 + 0.1 × 0.1 = 0.01
v = 0.999 × 0 + 0.001 × 0.01 = 1e-5

m̂ = 0.01 / (1 - 0.9^10) ≈ 0.01 / 0.651 = 0.0154
v̂ = 1e-5 / (1 - 0.999^10) ≈ 1e-5 / 0.00995 ≈ 1e-3

update = 0.01 × (0.0154 / (√1e-3 + 1e-8) + 0.01 × 0.5)
       = 0.01 × (0.0154 / 0.0316 + 0.005)
       = 0.01 × (0.487 + 0.005)
       = 0.01 × 0.492
       = 0.00492

θ_new = 0.5 - 0.00492 = 0.4951
```

## 7. 完整 C 代码

```c
// c/optimizer.c
typedef struct {
    float lr;            // 学习率
    float beta1;         // 一阶矩衰减
    float beta2;         // 二阶矩衰减
    float eps;           // 数值稳定
    float weight_decay;  // L2 系数
    int t;               // 当前步数
} AdamW;

void adamw_init(AdamW* opt) {
    opt->lr = 1e-3f;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->eps = 1e-8f;
    opt->weight_decay = 0.01f;
    opt->t = 0;
}

// 状态缓冲区（与参数等大）
float* m_memory;  // 一阶矩
float* v_memory;  // 二阶矩

void adamw_step(AdamW* opt, float* params, float* grads, int num_params) {
    opt->t += 1;

    // 偏差修正系数
    float bc1 = 1.0f - powf(opt->beta1, opt->t);
    float bc2 = 1.0f - powf(opt->beta2, opt->t);

    for (int i = 0; i < num_params; i++) {
        float g = grads[i];

        // 更新一阶矩
        m_memory[i] = opt->beta1 * m_memory[i] + (1.0f - opt->beta1) * g;
        // 更新二阶矩
        v_memory[i] = opt->beta2 * v_memory[i] + (1.0f - opt->beta2) * g * g;

        // 偏差修正
        float m_hat = m_memory[i] / bc1;
        float v_hat = v_memory[i] / bc2;

        // AdamW 更新
        params[i] -= opt->lr * (
            m_hat / (sqrtf(v_hat) + opt->eps) +
            opt->weight_decay * params[i]
        );
    }
}
```

## 8. AdamW vs Adam 对比

| 维度 | Adam | AdamW |
|---|---|---|
| L2 实现 | 加到 loss | 加到参数更新 |
| 与自适应学习率耦合 | ❌ 是 | ✅ 否 |
| 对 bias/LN 效果好 | 一般 | 好 |
| Transformer 默认 | ❌ | ✅ |

## 9. 学习率调度（进阶）

我们用**固定**学习率 1e-3。生产训练会用：

```
1) Warmup（线性增加）
   lr = 0 → 1e-3 over 100 steps

2) Cosine Decay（余弦衰减）
   lr = 1e-3 → 1e-4 over remaining steps

3) Constant（固定）
   lr = min_lr
```

```
lr
 ↑
 |    /\
 |   /  \
 |  /    \____
 | /         \____________
 |/                          → iter
 0  warmup  max_iters
```

教学模型省去，固定 `lr=1e-3` 即可。

## 10. 参数组（重要）

实际训练中，**不是所有参数都用 L2 正则**：

```c
// 1) 矩阵权重 (2D+) 用 weight_decay
// 2) bias 和 LayerNorm (1D) 不用 weight_decay
```

```c
void adamw_step_with_groups(AdamW* opt, GPT* model) {
    // 权重矩阵用 L2
    adamw_step_group(opt, model->params.qkvw, model->grads.qkvw, ...);
    adamw_step_group(opt, model->params.fcw, model->grads.fcw, ...);
    // ... 所有 2D 参数

    // 1D 参数不用 L2
    adamw_step_group_no_wd(opt, model->params.qkvb, model->grads.qkvb, ...);
    adamw_step_group_no_wd(opt, model->params.ln1w, model->grads.ln1w, ...);
}
```

**为什么？** bias 和 LN 是 1D 标量，对它们 L2 正则会伤害训练稳定性。论文证明分参数组效果好。

## 11. 训练循环完整流程

```c
// c/main.c
for (int step = 0; step < num_steps; step++) {
    // 1) 取数据
    get_batch(&inputs, &targets);

    // 2) Forward → loss
    gpt_forward(model, inputs, targets);
    float loss = model->mean_loss;
    printf("step %d: loss = %f\n", step, loss);

    // 3) Compute gradients (numerical)
    compute_numerical_grads(model, targets);

    // 4) Update params (AdamW)
    adamw_step(&opt, model->params_memory, model->grads_memory, num_params);
}
```

## 12. 关键概念

| 概念 | 含义 |
|---|---|
| **SGD** | 最朴素：`θ ← θ - lr × g` |
| **Momentum** | 加惯性，穿越鞍点 |
| **Adam** | 自适应学习率（per-parameter lr）|
| **AdamW** | Adam + 解耦的 L2 正则 |
| **β1, β2** | 一阶/二阶矩衰减率 |
| **偏差修正** | 前几步修正 m, v 的偏差 |
| **参数组** | 2D 用 L2，1D 不用 |

## 13. 直觉总结

AdamW = **每个参数都有自适应学习率的 SGD +**：
- 大的梯度 → 小步走（防止震荡）
- 小的梯度 → 大步走（防止停滞）
- 加 L2 防过拟合
- 加动量穿越鞍点

**这是为什么 Transformer 训练必须用 AdamW，不用 SGD**。

下一步：[08-INFERENCE.md](08-INFERENCE.md) — 看训练完怎么推理 "我喜欢苹果，它真好吃" → "苹果"。