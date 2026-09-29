# 04 MLP / FFN：每个 token 独立"思考"

> Attention 让 token 之间交流，MLP 让每个 token 独立加工信息。

## 1. 直觉

Attention 之后，每个 token 拥有了"上下文信息"（看到其他 token 后更新的表示）。MLP 的作用是**深度加工这些信息**。

类比：
- **Attention** = 开会讨论，每个人听其他人的意见
- **MLP** = 会后各自回家整理笔记

## 2. 结构

MLP 是**两层全连接网络 + 激活函数**：

```
input (C 维)
   ↓
Linear(C → 4C)        ← 升维：扩展"思维空间"
   ↓
GELU                   ← 非线性激活
   ↓
Linear(4C → C)        ← 降维：压回原维度
   ↓
output (C 维)
```

**公式**：

```
ffn(x) = Linear_2(GELU(Linear_1(x)))
       = W_2 · GELU(W_1 · x + b_1) + b_2
```

## 3. 为什么是 4 倍？

Transformer 标配升维到 **4C** 再降回 **C**。

直觉：
- 升维提供更大的"表达空间"（C→4C，每个 token 有更多维度思考）
- 降维把"加工完的信息"压回 C 维（与残差连接兼容）

论文 *"Attention Is All You Need"* 做了对比实验：

| 升维倍数 | 效果 | 备注 |
|---|---|---|
| 2C | 较差 | 表达空间不够 |
| **4C** | **最佳** | **默认选择** |
| 8C | 略好但慢 | 收益递减 |

## 4. GELU 激活函数

GELU = **Gaussian Error Linear Unit**

**公式**（精确版）：
```
GELU(x) = x · Φ(x)
```
其中 `Φ(x)` 是标准正态分布的累积分布函数（CDF）。

**近似版**（实践中常用）：
```
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))
```

**直觉**：
- `x > 0`：约等于 `x`（保留正值）
- `x < 0`：不完全截断，保留一些负值（与 ReLU 不同！）
- `x = 0`：约等于 0

### 4.1 GELU vs ReLU

```
        GELU                ReLU
   x · Φ(x)             max(0, x)
        │                    │
   ─────┼────────     ─────┼─────
        │  ╱                │   ╱
        │ ╱                 │  ╱
   ─────┼───────     ─────┼─────
       ╱│                │  ╱
      ╱ │                │ ╱
     ╱  │                │╱
```

**GELU 的优势**：
1. 平滑（连续可微）
2. 对负值保留少量信号（不是粗暴截断）
3. 实验效果好（Transformer 类任务）

**代码**：

```c
// c/activations.c:gelu_forward
float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x*x*x)));
}

void gelu_forward(float* out, const float* inp, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = gelu(inp[i]);
    }
}
```

## 5. 残差连接（Residual Connection）

每个子层都用 `y = x + sublayer(x)`：

```c
// 子层 1：Attention
x = x + attention(layer_norm_1(x))

// 子层 2：MLP
x = x + mlp(layer_norm_2(x))
```

**为什么需要残差？**

深层网络（6 层、12 层、几十层）的**梯度消失**问题：

```
没有残差:
∂L/∂x_in = ∂L/∂x_out · W₁ · W₂ · W₃ · ... · W_n
           ↑ 梯度经过很多次矩阵乘，趋近 0

有残差:
∂L/∂x_in = ∂L/∂x_out · (1 + ∂sublayer/∂x)
           ↑ 1 是直道，梯度至少不会消失
```

**类比**：
- 没残差 = 100 个人传话，源头听不清
- 有残差 = 100 个人传话 + 直接喊话源头

## 6. Pre-Norm vs Post-Norm

我们用 **Pre-Norm**（GPT-2 风格）：

```c
// Pre-Norm（我们的实现）
x = x + attention(layer_norm(x))    // 先 norm 再算子
x = x + mlp(layer_norm(x))

// Post-Norm（原论文）
x = layer_norm(x + attention(x))    // 算完再 norm
x = layer_norm(x + mlp(x))
```

**Pre-Norm 优势**：训练深层网络（24+ 层）更稳定。

## 7. 一个完整的 Transformer Block

```
输入 x (B, T, C)
   ↓
LayerNorm 1
   ↓
Attention(Q/K/V from x)        ← 见 03-ATTENTION.md
   ↓
+ 残差（加回原始 x）
   ↓
LayerNorm 2
   ↓
Linear(C → 4C) → GELU → Linear(4C → C)
   ↓
+ 残差
   ↓
输出 y (B, T, C)
```

## 8. 代码实现

### 8.1 MLP 前向

```c
// c/model.c:ffn_forward
void ffn_forward(
    float* out,        // (B, T, C) 输出
    const float* inp,  // (B, T, C) 输入
    const float* fcw,  // (4C, C) 第一层权重
    const float* fcb,  // (4C,)   第一层 bias
    const float* projw,// (C, 4C) 第二层权重
    const float* projb,// (C,)    第二层 bias
    int B, int T, int C
) {
    int C4 = 4 * C;

    // 第一层：C → 4C
    matmul_forward(fch_gelu, inp, fcw, fcb, B, T, C, C4);

    // GELU
    gelu_forward(fch_gelu, fch_gelu, B * T * C4);

    // 第二层：4C → C
    matmul_forward(out, fch_gelu, projw, projb, B, T, C4, C);
}
```

实际上分两步用 `matmul` 实现：

```c
// Step 1: matmul (C → 4C) + bias
matmul_forward(h, x, fcw, fcb, B, T, C, 4*C);

// Step 2: GELU
gelu_forward(h_gelu, h, B*T*4*C);

// Step 3: matmul (4C → C) + bias
matmul_forward(out, h_gelu, projw, projb, B, T, 4*C, C);
```

### 8.2 一个 Block 完整代码

```c
// c/model.c:block_forward
void block_forward(
    float* x,          // (B, T, C) 输入输出（in-place）
    float* att_out,    // (B, T, C) attention 中间值
    float* ln1,        // (B, T, C) layernorm 输出
    float* fch,        // (B, T, 4C) MLP 中间
    float* fch_gelu,   // (B, T, 4C) GELU 后
    const float* ln1w, const float* ln1b,
    const float* qkvw, const float* qkvb,
    const float* projw, const float* projb,
    const float* ln2w, const float* ln2b,
    const float* fcw, const float* fcb,
    const float* fcprojw, const float* fcprojb,
    int B, int T, int C, int NH
) {
    float* x_residual = x;  // 保存残差连接的源

    // ===== 子层 1: Attention =====
    layernorm_forward(ln1, x, ln1w, ln1b, B, T, C);    // Pre-LN
    attention_forward(att_out, ..., ln1, qkvw, qkvb, B, T, C, NH);
    residual_forward(x, att_out, B*T*C);                // x = residual + att_out

    // ===== 子层 2: MLP =====
    layernorm_forward(fch, x, ln2w, ln2b, B, T, C);    // Pre-LN
    matmul_forward(fch, ln1, fcw, fcb, B, T, C, 4*C);
    gelu_forward(fch_gelu, fch, B*T*4*C);
    matmul_forward(fch, fch_gelu, fcprojw, fcprojb, B, T, 4*C, C);
    residual_forward(x, fch, B*T*C);                    // x = x + fch
}
```

## 9. 数值示例

输入（一个 token，C=4 维）：
```
x = [0.5, 1.0, -0.3, 0.8]
```

**第一层 (C → 4C)**：
```
fcw = [[0.1, 0.2, 0.3, 0.4],     (4C=16 行 × C=4 列)
       ...
       ]
h = x @ fcw^T + fcb              # (16,)
h = [0.42, 0.86, -0.12, 1.34, 0.55, 0.91, 0.43, 1.02, 0.33, 0.77, 0.21, 0.95, 0.66, 0.88, 0.45, 1.11]
```

**GELU**：
```
h_gelu[i] = 0.5 * h[i] * (1 + tanh(0.7979 * (h[i] + 0.044715 * h[i]^3)))
h_gelu ≈ [0.21, 0.43, -0.06, 0.67, 0.275, 0.455, 0.215, 0.51, 0.165, 0.385, 0.105, 0.475, 0.33, 0.44, 0.225, 0.555]
```

**第二层 (4C → C)**：
```
out = h_gelu @ projw^T + projb    # (4,)
out = [0.42, -0.15, 0.78, 0.31]
```

**加残差**：
```
x_new = x + out = [0.5+0.42, 1.0-0.15, -0.3+0.78, 0.8+0.31]
      = [0.92, 0.85, 0.48, 1.11]
```

## 10. 关键概念

| 概念 | 公式 | 含义 |
|---|---|---|
| 升维 4C | `Linear(C→4C)` | 给模型更多表达空间 |
| GELU | `x · Φ(x)` | 平滑激活函数 |
| 降维 C | `Linear(4C→C)` | 与残差连接兼容 |
| 残差 | `y = x + sublayer(x)` | 防止梯度消失 |
| Pre-Norm | `sublayer(LN(x))` | 深层训练稳定 |

## 11. 直觉总结

**MLP 是"每个 token 的私人思考"**：
- Attention 让 token 之间传递信息（看别人说什么）
- MLP 让每个 token 自己加工信息（整理笔记）

整个 Transformer 就是在做：**反复的"看别人 → 自己思考"** 6 次（我们用 6 层 Block）。

下一步：[05-FORWARD.md](05-FORWARD.md) — 看完整 forward 流程怎么把所有零件拼起来。