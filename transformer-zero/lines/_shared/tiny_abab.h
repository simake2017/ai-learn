/* =============================================================================
 * tiny_abab.h - abab 最小模型公共夹具（供 tiny_train / trained_attn / generate 共用）
 *
 * 任务：vocab={a=0, b=1}，序列 a b a b ...，规则 a→b, b→a。
 * 规模：vocab=2, C=8, 1 层, 2 头（约 1032 参数），数值梯度 + AdamW。
 *
 * 这是「夹具数据 + 训练接线」，不是模型逻辑 ——  forward/梯度/优化器全部来自 c/。
 * 共用它的原因：三条线要解剖同一个模型，各自藏一份训练代码一定会脱钩。
 * =============================================================================
 */

#ifndef TINY_ABAB_H
#define TINY_ABAB_H

#include "../../c/transformer.h"

#define TINY_ABAB_T 8   /* 训练/检查用的序列长度 */

typedef struct {
    GPT model;
    AdamW opt;
} TinyAbab;

/* 训练样本：inputs = abababab, targets = babababa（每个位置的下一个字符） */
static inline const int* tiny_abab_inputs(void) {
    static const int inputs[TINY_ABAB_T] = {0, 1, 0, 1, 0, 1, 0, 1};
    return inputs;
}
static inline const int* tiny_abab_targets(void) {
    static const int targets[TINY_ABAB_T] = {1, 0, 1, 0, 1, 0, 1, 0};
    return targets;
}
static inline const char* tiny_abab_tok(int id) {
    return id ? "b" : "a";
}

/* 构建模型：激活按 alloc_T 分配（训练用 8，生成可放宽到 16），
 * 权重随机初始化（种子固定，三条线拿到同一个起点）。 */
static inline void tiny_abab_init(TinyAbab* t, int alloc_T) {
    GPT* m = &t->model;
    m->config.vocab_size = 2;
    m->config.block_size = 16;
    m->config.n_layer = 1;
    m->config.n_head = 2;
    m->config.n_embd = 8;
    m->batch_size = 1;
    m->seq_len = TINY_ABAB_T;   /* 默认训练长度 */

    int param_sizes[NUM_PER_TENSORS];
    fill_in_parameter_sizes(param_sizes, m->config);
    int total = 0;
    for (int i = 0; i < NUM_PER_TENSORS; i++) total += param_sizes[i];
    m->num_parameters = total;

    m->params_memory = malloc_and_set_params(&m->params, param_sizes);
    m->grads_memory = malloc_grads(&m->grads, param_sizes);
    m->m_memory = (float*)calloc(total, sizeof(float));
    m->v_memory = (float*)calloc(total, sizeof(float));

    int act_sizes[13] = {0};   /* 13 类激活张量（定义在 model.c，外部按 test_10 惯例用字面量） */
    fill_in_activation_sizes(act_sizes, m->config, 1, alloc_T);
    int total_acts = 0;
    for (int i = 0; i < 13; i++) total_acts += act_sizes[i];
    m->num_activations = total_acts;
    m->acts_memory = malloc_and_set_activations(&m->acts, act_sizes);

    srand(42);   /* 固定种子：三条线从同一个随机起点出发 */
    for (int i = 0; i < total; i++) {
        float r1 = (float)rand() / RAND_MAX + 1e-10f;
        float r2 = (float)rand() / RAND_MAX;
        m->params_memory[i] = (r1 + r2 - 1.5f) * 0.02f;
    }

    adamw_init(&t->opt);
}

/* 训练一步：forward(loss) → 数值梯度 → AdamW 更新。返回本步 loss。
 * 需要环境变量 NUMGRAD_QUIET=1 才能静默（调用方负责设置）。 */
static inline float tiny_abab_train_step(TinyAbab* t, const int* inputs, const int* targets) {
    GPT* m = &t->model;
    m->seq_len = TINY_ABAB_T;
    gpt_forward(m, inputs, targets);
    float loss = m->mean_loss;
    compute_numerical_grads(m, inputs, targets, 1e-3f);
    adamw_step(&t->opt, m);
    return loss;
}

static inline void tiny_abab_free(TinyAbab* t) {
    gpt_free(&t->model);
}

#endif /* TINY_ABAB_H */
