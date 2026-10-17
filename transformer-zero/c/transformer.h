/* =============================================================================
 * transformer.h - 从零实现 Transformer 的核心数据结构
 *
 * 教学项目：vocab=65, n_embd=128, n_layer=6, n_head=4
 *
 * 设计原则：
 *   1. 所有张量都用一维 float 数组存
 *   2. 每层参数展平到一维数组，按 l 索引切片
 *   3. 激活也展平到一维数组，按 l 索引切片
 *   4. 用结构体管理所有指针
 * =============================================================================
 */

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* === 配置 === */
#ifndef NUM_PER_TENSORS
#define NUM_PER_TENSORS 16
#endif

typedef struct {
    int vocab_size;      // 字符级 = 65
    int block_size;      // 最大序列长度
    int n_layer;         // Transformer Block 数
    int n_head;          // 多头注意力的头数
    int n_embd;          // 隐藏层维度
} Config;

/* === 模型参数（可学习的） === */
typedef struct {
    float* wte;       // (vocab_size, n_embd)        token embedding
    float* wpe;       // (block_size, n_embd)         position embedding

    // 每层：6 类共 12 个张量（n_layer 个，按 l 切片）
    float* ln1w;      // (n_layer, n_embd)           attn 前 LN gamma
    float* ln1b;      // (n_layer, n_embd)           attn 前 LN beta
    float* qkvw;      // (n_layer, 3*n_embd, n_embd) QKV 投影
    float* qkvb;      // (n_layer, 3*n_embd)         QKV bias
    float* projw;     // (n_layer, n_embd, n_embd)   attn 输出投影
    float* projb;     // (n_layer, n_embd)           attn bias
    float* ln2w;      // (n_layer, n_embd)           MLP 前 LN gamma
    float* ln2b;      // (n_layer, n_embd)           MLP 前 LN beta
    float* fcw;       // (n_layer, 4*n_embd, n_embd) MLP 第一层
    float* fcb;       // (n_layer, 4*n_embd)         MLP bias
    float* fcprojw;   // (n_layer, n_embd, 4*n_embd) MLP 第二层
    float* fcprojb;   // (n_layer, n_embd)           MLP bias

    // 最终
    float* lnfw;      // (n_embd)                     final LN gamma
    float* lnfb;      // (n_embd)                     final LN beta
} Params;

/* === 模型梯度（结构与 Params 一致） === */
typedef struct {
    float* wte; float* wpe;
    float* ln1w; float* ln1b;
    float* qkvw; float* qkvb;
    float* projw; float* projb;
    float* ln2w; float* ln2b;
    float* fcw; float* fcb;
    float* fcprojw; float* fcprojb;
    float* lnfw; float* lnfb;
} Grads;

/* === 激活（中间值，反向时用） === */
typedef struct {
    // 通用
    float* x;           // (B, T, C) 当前层输入/输出
    float* logits;      // (B, T, V) LM Head 输出
    float* probs;       // (B, T, V) softmax 后
    float* losses;      // (B, T)    每个 token 的 loss

    // 嵌入层
    float* tok_emb;     // (B, T, C) token embedding 输出
    float* pos_emb;     // (T, C)    position embedding（其实不必存）

    // 每层（n_layer 个切片）
    float* ln1;         // (n_layer, B, T, C)     attn 前 LN 输出
    float* qkv;         // (n_layer, B, T, 3C)    QKV 拼起来
    float* preatt;      // (n_layer, B, NH, T, T)  attention 分数
    float* att;         // (n_layer, B, NH, T, T)  attention 概率
    float* att_out;     // (n_layer, B, T, C)     attn 输出
    float* residual2;   // (n_layer, B, T, C)     第一次残差后
    float* ln2;         // (n_layer, B, T, C)     MLP 前 LN 输出
    float* fch;         // (n_layer, B, T, 4C)    MLP 第一层
    float* fch_gelu;    // (n_layer, B, T, 4C)    GELU 后
    float* fcproj;      // (n_layer, B, T, C)     MLP 第二层

    // 最终
    float* lnf;         // (B, T, C)               final LN 输出
} Activations;

/* === 完整 GPT 模型 === */
typedef struct {
    Config config;
    int batch_size;
    int seq_len;

    Params params;
    Grads grads;
    Activations acts;

    // 内存区（一大块连续内存，按 offset 切片）
    float* params_memory;
    float* grads_memory;
    float* acts_memory;

    // AdamW 状态
    float* m_memory;
    float* v_memory;

    int num_parameters;       // 参数总数
    int num_activations;      // 激活总数

    float mean_loss;          // 当前 forward 的平均 loss
} GPT;

/* === 初始化 / 内存管理 === */
void config_init(Config* c);
void config_small(Config* c);
void gpt_build_from_checkpoint(GPT* model, const char* checkpoint_path);
void gpt_save_checkpoint(GPT* model, const char* checkpoint_path);
void gpt_free(GPT* model);

/* === Forward === */
void gpt_forward(GPT* model, const int* inputs, const int* targets);

/* === Autodiff（数值微分） === */
void compute_numerical_grads(GPT* model, const int* inputs, const int* targets, float eps);
void zero_grads(Grads* grads, int n);

/* === Optimizer === */
typedef struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int t;
} AdamW;

void adamw_init(AdamW* opt);
void adamw_step(AdamW* opt, GPT* model);

/* === 算子 === */
/* === 内部辅助 === */
void fill_in_activation_sizes(int *act_sizes, Config config, int B, int T);
float* malloc_and_set_activations(Activations* acts, int *act_sizes);
void fill_in_parameter_sizes(int *param_sizes, Config config);
float* malloc_and_set_params(Params* params, int *param_sizes);
float* malloc_grads(Grads* grads, int *param_sizes);

void embedding_forward(float* out, const int* inputs,
                       const float* wte, const float* wpe,
                       int B, int T, int C);

void layernorm_forward(float* out, const float* inp,
                       const float* weight, const float* bias,
                       int B, int T, int C);

void matmul_forward(float* out, const float* inp,
                    const float* weight, const float* bias,
                    int B, int T, int C, int OC);

void attention_forward(float* out, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH);

/* 双向注意力（BERT 式，无因果掩码）：学习线演示「偷看未来」的对照算子 */
void attention_forward_bidirectional(float* out, float* preatt, float* att,
                                     const float* inp,
                                     int B, int T, int C, int NH);

void gelu_forward(float* out, const float* inp, int N);
void softmax_forward(float* probs, const float* logits, int B, int T, int V);
void crossentropy_forward(float* losses, const float* probs,
                         const int* targets, int B, int T, int V);
void residual_forward(float* out, const float* inp1, const float* inp2, int N);

/* === 数据加载 === */
typedef struct {
    int B;
    int T;
    int* inputs;
    int* targets;
    int num_batches;
    int current_idx;
    unsigned char* data;  // uint8（meta.pkl 里的 vocab 可能 < 256）
    int vocab_size;       // 数据集实际 vocab（应该是 65）
} DataLoader;

void dataloader_init(DataLoader* loader, const char* path, int B, int T, int vocab_size);
void dataloader_next_batch(DataLoader* loader);
void dataloader_free(DataLoader* loader);

/* === 推理 === */
int sample_token(const float* probs, int n, float temperature);

#endif // TRANSFORMER_H