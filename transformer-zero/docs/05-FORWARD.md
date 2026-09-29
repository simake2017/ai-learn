# 05 Forward：完整前向传播流程

> 把所有算子拼起来，看一次完整 forward 怎么从 token ids 走到 logits。

## 1. 完整流程图

```
输入: idx (B, T) 整数
   │
   ▼
embedding_forward()         # token + position embedding
   │
   ▼
x (B, T, C) 浮点
   │
   ▼
for l = 0 to n_layer-1:    # 6 个 Block 堆叠
   │
   ▼
   block_forward()         # 见 04-MLP.md
   │
   ▼
   x (B, T, C) 仍是
   │
   (回到循环)
   │
   ▼
final_layernorm()          # 最后一次 LayerNorm
   │
   ▼
lm_head()                  # C → V 维映射
   │
   ▼
logits (B, T, V)           # 每个位置预测下一个 token 的分数
```

## 2. 各步数据维度变化

假设 `B=1, T=3, C=128, n_layer=6, n_head=4, V=65`：

| 步骤 | 张量 | 形状 |
|---|---|---|
| 输入 | `idx` | `(1, 3)` int |
| Embedding | `tok_emb + pos_emb` | `(1, 3, 128)` float |
| Block 0 | `x` | `(1, 3, 128)` |
| Block 1 | `x` | `(1, 3, 128)` |
| ... | ... | ... |
| Block 5 | `x` | `(1, 3, 128)` |
| Final LN | `x` | `(1, 3, 128)` |
| LM Head | `logits` | `(1, 3, 65)` |

## 3. 参数张量清单

GPT-2 风格有 **16 类参数张量**（教学模型少一些但类似）：

```c
// c/transformer.h
typedef struct {
    float* wte;        // (V, C)      token embedding
    float* wpe;        // (maxT, C)   position embedding

    // 每层（共 n_layer 个，所以总参数数 × n_layer）
    float* ln1w, *ln1b;     // (n_layer, C)   attn 前的 LN
    float* qkvw, *qkvb;     // (n_layer, 3C, C)  QKV 投影
    float* projw, *projb;   // (n_layer, C, C)    attn 输出投影
    float* ln2w, *ln2b;     // (n_layer, C)   MLP 前的 LN
    float* fcw, *fcb;       // (n_layer, 4C, C)  MLP 升维
    float* fcprojw, *fcprojb; // (n_layer, C, 4C)  MLP 降维

    // 最终
    float* lnfw, *lnfb;     // (C,)   final LN
} Params;
```

为了方便访问每层参数，实际存储是把所有层展平到一维数组：

```
ln1w_total[0..C-1]           = layer 0 的 ln_1.w
ln1w_total[C..2C-1]         = layer 1 的 ln_1.w
...
```

访问第 l 层参数：
```c
float* l_ln1w = params->ln1w + l * C;
float* l_qkvw = params->qkvw + l * 3 * C * C;
```

## 4. 激活张量清单（中间存储，反向时用）

```c
typedef struct {
    float* x;             // (B, T, C) 当前层的输入/输出
    float* ln1;            // (B, T, C) attn 前 LN 输出
    float* qkv;            // (B, T, 3C) QKV 拼起来
    float* att;            // (B, n_head, T, T) attention 概率
    float* att_out;        // (B, T, C) attn 输出
    float* residual2;      // (B, T, C) 第一次残差后的
    float* ln2;            // (B, T, C) MLP 前 LN 输出
    float* fch;            // (B, T, 4C) MLP 第一层输出
    float* fch_gelu;       // (B, T, 4C) GELU 后
    float* fcproj;         // (B, T, C) MLP 第二层输出
    float* logits;         // (B, T, V) 最终 logits
    float* probs;          // (B, T, V) softmax 后概率
    float* losses;         // (B, T)   每个 token 的交叉熵
} Activations;
```

激活数量很多，但**只在前向时分配一次**，然后复用（in-place 操作）。

## 5. 完整 forward 代码

```c
// c/model.c:gpt_forward
void gpt_forward(
    GPT* model,
    const int* inputs,    // (B, T)
    const int* targets    // (B, T) or NULL
) {
    int B = model->batch_size;
    int T = model->seq_len;
    int C = model->config.n_embd;
    int V = model->config.vocab_size;
    int NH = model->config.n_head;
    int L = model->config.n_layer;

    // ========== Step 1: Embedding ==========
    embedding_forward(
        model->acts.x,
        inputs,
        model->params.wte,
        model->params.wpe,
        B, T, C
    );

    // ========== Step 2: n_layer 个 Transformer Block ==========
    for (int l = 0; l < L; l++) {
        // 取出第 l 层的参数
        float* l_ln1w    = model->params.ln1w    + l * C;
        float* l_ln1b    = model->params.ln1b    + l * C;
        float* l_qkvw    = model->params.qkvw    + l * 3*C * C;
        float* l_qkvb    = model->params.qkvb    + l * 3*C;
        float* l_projw   = model->params.projw   + l * C * C;
        float* l_projb   = model->params.projb   + l * C;
        float* l_ln2w    = model->params.ln2w    + l * C;
        float* l_ln2b    = model->params.ln2b    + l * C;
        float* l_fcw     = model->params.fcw     + l * 4*C * C;
        float* l_fcb     = model->params.fcb     + l * 4*C;
        float* l_fcprojw = model->params.fcprojw + l * C * 4*C;
        float* l_fcprojb = model->params.fcprojb + l * C;

        // 取出第 l 层的激活
        float* l_x        = model->acts.x;  // in-place
        float* l_ln1      = model->acts.ln1      + l * B*T*C;
        float* l_qkv      = model->acts.qkv      + l * B*T*3*C;
        float* l_preatt   = model->acts.preatt   + l * B*NH*T*T;
        float* l_att      = model->acts.att      + l * B*NH*T*T;
        float* l_att_out  = model->acts.att_out  + l * B*T*C;
        float* l_residual2= model->acts.residual2+ l * B*T*C;
        float* l_ln2      = model->acts.ln2      + l * B*T*C;
        float* l_fch      = model->acts.fch      + l * B*T*4*C;
        float* l_fch_gelu = model->acts.fch_gelu + l * B*T*4*C;
        float* l_fcproj   = model->acts.fcproj   + l * B*T*C;

        // ===== 子层 1: Attention =====
        layernorm_forward(l_ln1, l_x, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_att_out, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_att_out, l_att_out, l_projw, l_projb, B, T, C, C);
        residual_forward(l_residual2, l_x, l_att_out, B*T*C);

        // ===== 子层 2: MLP =====
        layernorm_forward(l_ln2, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_x, l_residual2, l_fcproj, B*T*C);
    }

    // ========== Step 3: Final LayerNorm ==========
    layernorm_forward(
        model->acts.x,
        model->acts.x,  // in-place
        model->params.lnfw, model->params.lnfb,
        B, T, C
    );

    // ========== Step 4: LM Head (C → V) ==========
    matmul_forward(
        model->acts.logits,
        model->acts.x,
        model->params.wte,  // weight tying: 共享 wte
        NULL,                // no bias
        B, T, C, V
    );

    // ========== Step 5: Softmax (可选, 用于 loss) ==========
    softmax_forward(model->acts.probs, model->acts.logits, B, T, V);

    // ========== Step 6: Loss ==========
    if (targets != NULL) {
        crossentropy_forward(
            model->acts.losses,
            model->acts.probs,
            targets,
            B, T, V
        );
        // 平均 loss
        float total = 0;
        for (int i = 0; i < B*T; i++) total += model->acts.losses[i];
        model->mean_loss = total / (B*T);
    } else {
        model->mean_loss = -1.0f;  // 推理模式
    }
}
```

## 6. 关键算子回顾

| 算子 | 作用 | 公式 |
|---|---|---|
| **Embedding** | token → 向量 | `wte[id] + wpe[pos]` |
| **LayerNorm** | 归一化 | `(x - μ) / σ · γ + β` |
| **MatMul** | 线性变换 | `y = x @ W^T + b` |
| **Attention** | token 交流 | `softmax(QK^T/√d) V` |
| **GELU** | 非线性 | `x · Φ(x)` |
| **Softmax** | 概率分布 | `exp(x)/Σexp(x)` |
| **CrossEntropy** | 损失 | `-log P(target)` |

## 7. 权重共享（Weight Tying）

LM Head 的权重 = Token Embedding 的权重：

```
logits = x @ wte^T     而不是    logits = x @ lm_head_w^T
```

**优点**：
1. 节省参数（V × C = 65 × 128 = 8K）
2. 实验证明几乎不影响效果
3. 训练更稳定

论文：PressUsing "Using the Output Embedding to Improve Language Models"。

## 8. 完整数据流示例

输入：`idx = [12, 45, 78]`（"I like apples" 的 token ids）

```
Step 0: Embedding
  x = [[0.11, 0.22, ..., 0.44],     # "I"
       [0.95, 1.06, ..., 1.28],     # "like"
       [0.59, 0.70, ..., 0.92]]     # "apples"
  shape: (1, 3, 128)

Step 1: Block 0
  # Attention 让 apples 看 I 和 like
  att[2] = [0.13, 0.27, 0.60]      # "apples" 主要看自己
  # MLP 加工
  x = [[0.92, 0.85, ..., 0.91], ...]

Step 2-5: Block 1-5（类似）

Step 6: Final LN
  x = [[0.50, 0.30, ..., 0.78], ...]

Step 7: LM Head
  logits[0] = [0.1, 0.3, 2.1, ..., 0.8]   # "I" 后面是 "like" 的分数
  logits[1] = [0.5, 3.4, 1.2, ..., 1.1]    # "like" 后面是 "apples"
  logits[2] = [0.7, 0.2, 0.9, ..., 2.5]    # "apples" 后面是 "."
  shape: (1, 3, 65)

Step 8: Softmax (可选)
  probs[1] = [0.01, 0.05, 0.20, ..., 0.10]  # 概率分布

Step 9: CrossEntropy (如果有 targets)
  loss = mean(-log P(target)) = 1.2
```

## 9. 内存使用

| 类型 | 大小 | 备注 |
|---|---|---|
| 参数 | ~600K × 4 = 2.4 MB | 静态 |
| 激活 | ~200K × 4 = 0.8 MB | 每次 forward |
| 梯度 | ~600K × 4 = 2.4 MB | 训练时 |
| AdamW 状态 (m + v) | ~1.2M × 4 = 4.8 MB | 训练时 |

**教学模型总内存 < 15 MB**，单核 CPU 足够。

## 10. 关键点总结

1. **Embedding + n_layer × (LN + Attn + Res + LN + MLP + Res) + Final LN + LM Head**
2. **权重共享**：wte 既用于输入嵌入，也用于 LM Head
3. **Pre-Norm**：每个子层前都做 LN
4. **残差连接**：每个子层都加回输入
5. **激活缓存**：所有中间值都存下来，反向时用

下一步：[06-AUTODIFF.md](06-AUTODIFF.md) — 看怎么用数值微分算梯度。