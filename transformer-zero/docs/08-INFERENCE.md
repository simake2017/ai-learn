# 08 Inference：自回归生成文本

> 训练完成后，模型怎么生成 "我喜欢苹果，它真好吃" → 预测 "苹果"？

## 1. 推理 vs 训练

| 维度 | 训练 | 推理 |
|---|---|---|
| 输入 | 整段 token | 已生成的 tokens |
| 输出 | loss + grad | 下一个 token 的概率 |
| 计算 | forward + grad + update | 仅 forward |
| 模式 | teacher forcing | 自回归 |
| 时间 | 慢 | 快 |

## 2. 自回归生成（Autoregressive Generation）

```
输入:  [I, like, apples, ,, they, are]
   ↓ 模型 forward
输出:  下个 token 的概率分布
   ↓ 采样
   ↓ [选 "delicious"]
新输入: [I, like, apples, ,, they, are, delicious]
   ↓ 模型 forward（重新跑一遍整段！）
输出:  下个 token 的概率分布
   ↓ 采样
   ↓ [选 "and"]
... 继续直到结束
```

**注意**：每次都要重新跑整段 forward（不像训练时 batch 并行）。这是推理慢的原因。

## 3. 完整推理代码

```c
// c/main.c:generate
void generate(
    GPT* model,
    const int* prompt,    // 输入 token 序列
    int prompt_len,       // 输入长度
    int max_new_tokens,    // 最多生成多少 token
    float temperature      // 采样温度（0 = greedy, 1 = 标准）
) {
    // 把 prompt 复制到工作区
    int tokens[1024];
    for (int i = 0; i < prompt_len; i++) tokens[i] = prompt[i];

    for (int step = 0; step < max_new_tokens; step++) {
        // 1) Forward：拿 logits
        // 把整段 tokens 输入，跑 forward
        int T = prompt_len + step;
        gpt_forward(model, tokens, NULL);  // NULL = 不算 loss

        // 2) 取最后一个位置的 logits
        // logits 形状 (B, T, V)，我们需要 logits[0, T-1, :]
        float* last_logits = model->acts.logits + (T - 1) * model->config.vocab_size;

        // 3) 应用温度
        for (int v = 0; v < model->config.vocab_size; v++) {
            last_logits[v] /= temperature;
        }

        // 4) Softmax → 概率
        softmax(last_logits, last_logits, model->config.vocab_size);

        // 5) 采样一个 token
        int next_token = sample_token(last_logits, model->config.vocab_size);

        // 6) 把新 token 加到序列
        tokens[prompt_len + step] = next_token;

        // 7) 打印
        printf("%c", decode_char(next_token));  // 或者用 tokenizer
    }
}
```

## 4. 采样策略

### 4.1 Greedy（贪心）

```c
int next_token = argmax(logits);
```

**永远选概率最大的 token**。

**问题**：容易重复（"the the the"）、缺乏多样性。

### 4.2 随机采样

```c
// 按概率分布采样
float probs[V];
softmax(probs, logits, V);
float coin = random_f32();  // [0, 1)
int cdf = 0;
for (int v = 0; v < V; v++) {
    cdf += probs[v];
    if (coin < cdf) return v;
}
```

**按概率采样**，多样性好。

### 4.3 温度（Temperature）

```c
// 高温（T > 1）：分布更平 → 更多样
// 低温（T → 0）：分布更尖 → 更确定
logits[v] = logits[v] / T;
softmax(probs, logits, V);
```

```
T = 0.1: [0.99, 0.005, 0.005]   → 几乎确定选第一个
T = 1.0: [0.5,  0.3,   0.2]     → 标准
T = 2.0: [0.33, 0.33,  0.34]    → 更随机
```

### 4.4 Top-K 采样

```c
// 只在概率最大的 K 个 token 中采样
int top_k_indices[K];
float top_k_probs[K];
// 取 top K 个
// 重新归一化
// 在这 K 个中采样
```

**效果**：避免选到奇怪的低概率 token。

### 4.5 Top-P (Nucleus) 采样

```c
// 按概率排序，取累计概率 >= p 的最小集合
// 在这个集合里重新归一化采样
```

**效果**：自适应地选多少个 token。

## 5. KV Cache 优化（生产用）

我们的简单实现每次重算整段 forward。生产中用 **KV Cache**：

```
不用 KV Cache:
  生成 token 1: 跑 [prompt] → token 1
  生成 token 2: 跑 [prompt, token 1] → token 2    ← 重复算！
  生成 token 3: 跑 [prompt, token 1, token 2] → token 3

用 KV Cache:
  第一次：跑 [prompt] → 保存 K, V
  第二次：只跑 [token 1]，用之前存的 K, V
  第三次：只跑 [token 2]，用之前存的 K, V
```

**加速 ~10x**。我们教学版不实现，但生产训练会用。

## 6. 完整例子："我喜欢苹果，它真好吃"

### 6.1 输入

```
prompt: "I like apples, they are"
tokens: [I, like, apples, ,, they, are] = [12, 45, 78, 5, 67, 89]
```

### 6.2 第 1 步：forward + 采样

```
输入: [12, 45, 78, 5, 67, 89]
logits[5] (位置 "are" 后): [0.5, 0.3, 1.2, 2.5, ..., 1.1]  (65 维)

softmax + 采样:
probs: [0.02, 0.01, 0.05, 0.50, ..., 0.04]  (65 维)
          ↓ 采样
选到 23（"delicious"）

新序列: [12, 45, 78, 5, 67, 89, 23]
```

### 6.3 第 2 步：再 forward

```
输入: [12, 45, 78, 5, 67, 89, 23]
logits[6] (位置 "delicious" 后): [...]

softmax + 采样:
选到 8 (".")

新序列: [12, 45, 78, 5, 67, 89, 23, 8]
```

### 6.4 继续直到 EOT

```
选到 <EOT> (假设是 0)
停止生成
```

最终输出："I like apples, they are delicious."

## 7. 解码 Token ID

我们的字符级 vocab（65 字符），可以直接映射回字符：

```c
char decode_char(int token_id) {
    // 反向 meta.pkl 里的 itos 映射
    return itos[token_id];  // int → char
}
```

如果是 BPE vocab（GPT-2 用），需要用 tiktoken 库：

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
text = enc.decode([15496, 995, 19684, 11, 532, 464])
# → "Hello world people, we are"
```

## 8. Python 胶水调用

```python
# python/inference.py
import ctypes

lib = ctypes.CDLL('./build/libtransformer.so')

# 加载模型
lib.load_model.argtypes = [ctypes.c_char_p]
lib.load_model.restype = ctypes.c_void_p

model = lib.load_model(b'gpt_tiny.bin')

# 设置输入
lib.gpt_forward.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int
]
lib.gpt_forward.restype = ctypes.c_float

prompt = [12, 45, 78, 5, 67, 89]  # "I like apples, they are"
prompt_arr = (ctypes.c_int * len(prompt))(*prompt)

# 调用 C 函数
loss = lib.gpt_forward(model, prompt_arr, len(prompt), 0)

# 取 logits，采样下一个 token
# ...
```

## 9. 完整 Python 推理脚本

```python
# python/inference_demo.py
import ctypes
import numpy as np

# 加载 C 库
lib = ctypes.CDLL('./build/libtransformer.so')

# 设置函数签名
lib.gpt_forward.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
    ctypes.c_int,
    ctypes.c_int
]
lib.gpt_forward.restype = ctypes.c_float

lib.get_logits.argtypes = [ctypes.c_void_p]
lib.get_logits.restype = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3)

lib.sample.argtypes = [ctypes.c_void_p, ctypes.c_float]
lib.sample.restype = ctypes.c_int

lib.load_model.argtypes = [ctypes.c_char_p]
lib.load_model.restype = ctypes.c_void_p

# 加载模型
model = lib.load_model(b'gpt_tiny.bin')

# 准备 prompt
prompt_text = "I like apples, they are"
prompt_ids = [stoi[c] for c in prompt_text]

# 反复生成
for _ in range(30):
    inputs = np.array([prompt_ids[-64:]], dtype=np.int32)  # 最多 64 token
    lib.gpt_forward(model, inputs, inputs.shape[0], inputs.shape[1])

    # 取最后一个位置的 logits
    logits = lib.get_logits(model)
    last_logits = logits[0, -1, :]

    # 应用温度
    temperature = 0.8
    probs = np.exp(last_logits / temperature)
    probs /= probs.sum()

    # 采样
    next_token = np.random.choice(len(probs), p=probs)
    prompt_ids.append(int(next_token))

    # 解码打印
    print(itos[next_token], end='', flush=True)

print()
```

## 10. 推理模式优化

训练时算所有位置的 loss：

```c
// 训练：算所有 T 个位置的 loss
logits = lm_head(x)  // (B, T, V)
loss = cross_entropy(logits, targets)  // 全部 T 个位置
```

推理时只需要最后一个位置的 logits：

```c
// 推理：只算最后一个位置
// 优化：把 LN 和 LM Head 应用到 x[:, -1, :] 即可
```

```c
void gpt_forward_inference(GPT* model, const int* inputs, int T) {
    // ... 跑 n_layer 个 Block ...

    // 取最后一个 token 的 x
    float* x_last = model->acts.x + (T-1) * C;

    // Final LN (只对最后一个位置)
    layernorm_one(x_last, ...);

    // LM Head (只对最后一个位置)
    matmul_forward(model->acts.logits, x_last, wte, NULL, 1, 1, C, V);
}
```

**加速 ~T 倍**（生成 token 时从 O(T²) 降到 O(T)）。

## 11. 关键概念

| 概念 | 说明 |
|---|---|
| **自回归** | 每次生成一个 token，加到末尾 |
| **温度采样** | 控制多样性 |
| **Greedy** | 永远选概率最大 |
| **Top-K / Top-P** | 限制采样范围 |
| **KV Cache** | 推理优化（生产用）|
| **EOT token** | 文本结束符 |

## 12. 直觉总结

**推理就是"自言自语 + 自己回答"**：

```
模型看到 "I like apples"
   ↓ 想了一下
"应该是 delicious 吧"
   ↓
"对了，那就 delicious. 然后呢？"
   ↓
"继续写... and crunchy"
   ↓
... 直到 "."
   ↓
"完了，停止生成"
```

每次只看"目前为止我说了什么"，决定"下一个该说什么"。

## 13. 完整工作流总结

```
1) 训练: train.py
   ↓
   训练数据 → forward → loss → 数值微分 → AdamW → 更新参数
   ↓
   1000 步后，loss 下降到 1.5 左右

2) 推理: inference_demo.py
   ↓
   加载训练好的参数
   ↓
   输入 prompt "I like apples, they are"
   ↓
   自回归生成 "delicious. and crunchy."
```

## 14. 推荐下一步

| 任务 | 命令 |
|---|---|
| 训练小模型 | `bash run.sh train` |
| 看 loss 曲线 | `python python/visualize_loss.py` |
| 推理 + 打印 | `python python/inference_demo.py` |
| 画 attention 热力图 | `python python/visualize_attention.py` |

完整项目就在 `/home/magene/runtime/ai/ai-learn/transformer-zero/` 下。祝学习愉快！