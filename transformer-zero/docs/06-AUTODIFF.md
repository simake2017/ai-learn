# 06 数值微分（Numerical Differentiation）

> 教学项目用手写前向 + **数值微分**算梯度。最容易理解"梯度是什么"。

## 1. 什么是梯度？

**梯度**告诉我们：**调一个参数 θ，对 loss 的影响有多大**。

```
∂L/∂θ ≈ "θ 变化一点点，loss 变化多少"
```

- `∂L/∂θ > 0`：增大 θ → loss 增大 → 应该减小 θ
- `∂L/∂θ < 0`：增大 θ → loss 减小 → 应该增大 θ
- `∂L/∂θ ≈ 0`：调 θ 没用，调别的

## 2. 三种算梯度的方法

### 2.1 解析微分（Analytic）

**手写公式**：

```
y = x * 2
dy/dx = 2          (求导)

y = x²
dy/dx = 2x         (求导)

y = relu(x) = max(0, x)
dy/dx = 1 if x > 0 else 0   (分段)
```

**优点**：快（一次算所有梯度）
**缺点**：复杂函数难写、容易错

### 2.2 自动微分（Autograd / PyTorch）

**构建计算图，反向传播**：

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2              # 计算图自动记录
y.backward()             # 自动求 ∂y/∂x
print(x.grad)            # tensor([4.])  = 2x = 2*2
```

**优点**：方便
**缺点**：黑盒，看不到细节

### 2.3 数值微分（Numerical）

**用极限定义算梯度**：

```
∂L/∂θ ≈ (L(θ + ε) - L(θ - ε)) / (2ε)
```

```python
def numerical_grad(L_fn, theta, eps=1e-4):
    saved = theta.copy()
    L_plus  = L_fn(theta + eps)   # L(θ + ε)
    L_minus = L_fn(theta - eps)   # L(θ - ε)
    theta[:] = saved               # 还原
    return (L_plus - L_minus) / (2 * eps)
```

**优点**：最直观、不依赖数学推导
**缺点**：慢（每个参数要算 2 次 forward）

## 3. 为什么我们用数值微分？

**教学目的**：最容易理解"梯度"是什么。

- 不用记复杂公式
- 不用理解计算图
- 直接看出"调这个参数，loss 怎么变"

代价是**慢**：对 N 个参数算梯度需要 2N 次 forward。

我们的教学模型只有 ~600K 参数（小），所以**每次约 1.2M 次 forward**。即使每次 forward 10ms，总共 3.3 小时能跑完 1000 步。

## 4. 数学基础

### 4.1 微分的极限定义

```
∂L/∂θ = lim (L(θ + Δ) - L(θ)) / Δ    as Δ → 0
```

直接实现：

```python
Δ = 1e-5
grad = (L(theta + Δ) - L(theta)) / Δ
```

### 4.2 中心差分（更精确）

```
∂L/∂θ ≈ (L(θ + ε) - L(θ - ε)) / (2ε)
```

用 θ 两侧的点做差分，比单侧差分精度高（误差是 O(ε²) 而非 O(ε)）。

**举例**：

设 `L(θ) = θ²`，真实梯度 `dL/dθ = 2θ`。

`θ = 3, ε = 0.01`：
- 单侧：`((3.01)² - 9) / 0.01 = (9.0601 - 9) / 0.01 = 6.01`，误差 0.01
- 中心：`((3.01)² - (2.99)²) / 0.02 = (9.0601 - 8.9401) / 0.02 = 6.000`，误差 0

**结论**：用中心差分。

## 5. 完整算法

```c
// 对每个参数 θ_i 算梯度
void compute_numerical_grads(GPT* model, const int* targets) {
    int num_params = count_params(model);
    float eps = 1e-3f;  // 适当小，但不能太小（浮点精度）

    for (int i = 0; i < num_params; i++) {
        float saved = model->params_memory[i];

        // 算 L(θ + ε)
        model->params_memory[i] = saved + eps;
        gpt_forward(model, model->inputs, targets);
        float L_plus = model->mean_loss;

        // 算 L(θ - ε)
        model->params_memory[i] = saved - eps;
        gpt_forward(model, model->inputs, targets);
        float L_minus = model->mean_loss;

        // 中心差分
        model->grads[i] = (L_plus - L_minus) / (2.0f * eps);

        // 还原参数
        model->params_memory[i] = saved;
    }
}
```

## 6. ε 怎么选？

太大 → 截断误差大（梯度不准）
太小 → 浮点精度误差大（`L_plus - L_minus` 几乎为 0）

**经验值**：
- `eps = 1e-3`：对教学模型够用
- `eps = 1e-5`：高精度但慢
- `eps = 1e-1`：太粗，仅用于 sanity check

**实践**：

```c
// 教学用 eps
float eps = 1e-3f;
```

## 7. 优化：mini-batch 求和再算梯度

每次 forward 都用整个 batch（B*T 个样本）算 loss：

```c
float L = 0;
for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
        L += -log(probs[b, t, targets[b, t]]);
    }
}
L /= (B * T);  // 平均 loss
```

数值微分时，每个参数扰动 ε 后算这个 loss，取中心差分。

## 8. 时间复杂度

设参数总数 `N`，每个 forward 耗时 `T_f`：

```
N 次循环 × 2 次 forward = 2N × T_f
```

我们的模型：
- N = 600K 参数
- T_f ≈ 10ms（CPU）

```
总时间 = 2 × 600,000 × 10ms = 12,000,000 ms = 200 分钟 ≈ 3.3 小时
```

**优化**：用更小的 batch（B=1）或只对部分参数算梯度（但都是"作弊"）。

## 9. 与真实梯度对比（验证）

为了验证数值微分算出来的梯度是对的，可以对比解析微分：

```c
// 1) 算数值微分
float eps = 1e-3f;
params[0] += eps;
float L_plus = gpt_forward(...);
params[0] -= 2*eps;
float L_minus = gpt_forward(...);
params[0] += eps;  // 还原
float numerical_grad = (L_plus - L_minus) / (2*eps);

// 2) 解析微分（如果实现的话）
float analytic_grad = compute_analytic_grad();

// 3) 对比
float diff = fabs(numerical_grad - analytic_grad);
printf("grad difference: %.6f\n", diff);
// 应该 < 1e-3
```

我们的项目不实现解析微分，所以**只信数值微分结果**（验证方法：检查 loss 是否下降）。

## 10. 验证梯度正确的另一种方法

**有限差分检查（Gradient check）**：

```
训练时观察：
1) loss 是否下降？
2) 下降速度是否合理？
3) 梯度值是否在合理范围（1e-5 ~ 1e2）？

如果 loss 不下降或爆炸 → 梯度算错了
```

## 11. 数值微分的陷阱

### 11.1 浮点精度问题

```c
// L(θ + ε) 和 L(θ - ε) 几乎相等时，差分会丢精度
float L_plus  = L(theta + 1e-7f);  // 浮点几乎不变
float L_minus = L(theta - 1e-7f);
float grad = (L_plus - L_minus) / (2e-7f);  // ≈ 0/0 = NaN!
```

**解决**：ε 不能太小。

### 11.2 参数溢出

如果参数初始化太大，加 ε 后可能 overflow：

```c
float x = 1e30f;
x += 1e-3f;  // 还是 1e30f
```

**解决**：参数初始化小（±0.02）。

### 11.3 死参数

如果参数对 loss 没影响（如被 mask 掉），梯度 ≈ 0：

```c
// 优化器会忽略 grad ≈ 0 的参数
```

## 12. 数值微分 vs 解析梯度对比

| 维度 | 数值微分 | 解析微分 |
|---|---|---|
| **代码量** | 5 行 | 600 行（每个算子）|
| **速度** | 慢（2N 次 forward）| 快（1 次 forward + 1 次 backward）|
| **精度** | 受 ε 限制 | 机器精度 |
| **教学性** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **工业使用** | ❌ | ✅ |
| **本项目** | ✅ | ❌ |

## 13. 完整 C 代码

```c
// c/autodiff.c
#include <math.h>

void compute_numerical_grads(GPT* model, const int* targets) {
    int num_params = count_params(model);
    float eps = 1e-3f;

    for (int i = 0; i < num_params; i++) {
        float saved = model->params_memory[i];

        // L(θ + ε)
        model->params_memory[i] = saved + eps;
        gpt_forward(model, model->inputs, targets);
        float L_plus = model->mean_loss;

        // L(θ - ε)
        model->params_memory[i] = saved - eps;
        gpt_forward(model, model->inputs, targets);
        float L_minus = model->mean_loss;

        // 中心差分
        model->grads[i] = (L_plus - L_minus) / (2.0f * eps);

        // 还原
        model->params_memory[i] = saved;

        // 进度
        if (i % 10000 == 0) {
            printf("grad %d / %d, current = %g\n", i, num_params, model->grads[i]);
        }
    }
}
```

## 14. 关键概念

| 概念 | 含义 |
|---|---|
| **梯度** | 调参数对 loss 的影响 |
| **中心差分** | `(L(θ+ε) - L(θ-ε)) / 2ε` |
| **ε** | 扰动大小，1e-3 是好默认 |
| **2N 次 forward** | 数值微分的代价 |
| **梯度检验** | 验证 loss 是否下降 |

## 15. 直觉总结

**数值微分就是把"求导"变成"做两次实验"**：

```
实验 1：参数加一点点 → loss 变成多少？
实验 2：参数减一点点 → loss 变成多少？

梯度 = (loss变化) / (参数变化) 的近似
```

物理类比：开车时看速度表，把油门稍微踩深一点，看速度变多少。

下一步：[07-OPTIMIZER.md](07-OPTIMIZER.md) — 看 AdamW 怎么用梯度更新参数。