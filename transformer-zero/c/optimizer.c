/* =============================================================================
 * optimizer.c - AdamW 优化器
 *
 * m = β1·m + (1-β1)·g         (一阶矩)
 * v = β2·v + (1-β2)·g²        (二阶矩)
 * m̂ = m / (1 - β1^t)         (偏差修正)
 * v̂ = v / (1 - β2^t)
 * θ ← θ - lr·(m̂/(√v̂ + ε) + λ·θ)   (AdamW 更新)
 * =============================================================================
 */

#include "transformer.h"

void adamw_init(AdamW* opt) {
    opt->lr = 1e-3f;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->eps = 1e-8f;
    opt->weight_decay = 0.01f;
    opt->t = 0;
}

void adamw_step(AdamW* opt, GPT* model) {
    opt->t += 1;

    float bc1 = 1.0f - powf(opt->beta1, opt->t);
    float bc2 = 1.0f - powf(opt->beta2, opt->t);

    int n = model->num_parameters;
    for (int i = 0; i < n; i++) {
        float g = model->grads_memory[i];

        // 一阶矩
        model->m_memory[i] = opt->beta1 * model->m_memory[i] +
                              (1.0f - opt->beta1) * g;
        // 二阶矩
        model->v_memory[i] = opt->beta2 * model->v_memory[i] +
                              (1.0f - opt->beta2) * g * g;

        // 偏差修正
        float m_hat = model->m_memory[i] / bc1;
        float v_hat = model->v_memory[i] / bc2;

        // AdamW 更新
        float adam_update = m_hat / (sqrtf(v_hat) + opt->eps);
        float decay_update = opt->weight_decay * model->params_memory[i];

        model->params_memory[i] -= opt->lr * (adam_update + decay_update);
    }
}