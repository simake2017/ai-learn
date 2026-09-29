/* =============================================================================
 * test_12_adamw.c - AdamW 优化器单元测试
 *
 * 📚 学习目标：
 *   理解 AdamW 如何根据梯度更新模型参数
 *
 * 🧮 数学原理（AdamW 算法）：
 *
 *   t = t + 1                                    ← 步数计数
 *   m = β1·m + (1-β1)·g                          ← 一阶矩（梯度的指数移动平均）
 *   v = β2·v + (1-β2)·g²                         ← 二阶矩（梯度平方的指数移动平均）
 *   m̂ = m / (1 - β1^t)                          ← 偏差修正（初始阶段 m 偏小）
 *   v̂ = v / (1 - β2^t)                          ← 偏差修正
 *   θ = θ - lr × (m̂/(√v̂ + ε) + λ·θ)          ← AdamW 更新
 *                                            ↑ Adam 部分   ↑ weight decay
 *
 * 💡 各部分直觉：
 *   - m（一阶矩）：梯度的"方向"，平滑噪音
 *   - v（二阶矩）：梯度的"大小"，自适应学习率
 *   - m̂/(√v̂+ε)：方向 ÷ 大小 → 归一化的更新方向
 *   - β1^t 偏差修正：初始 m=0，前几步 m 偏小，除以 (1-β1^t) 修正
 *   - weight decay (λ·θ)：参数越大惩罚越大，防止过拟合
 *
 * 💡 AdamW vs Adam：
 *   Adam:  θ = θ - lr × m̂/(√v̂ + ε) - lr×λ×θ     ← decay 和 lr 耦合
 *   AdamW: θ = θ - lr × (m̂/(√v̂ + ε) + λ×θ)    ← decay 独立于自适应 lr
 *   AdamW 在实践中效果更好（GPT-2/3 都用 AdamW）
 *
 * 🔍 测试策略：
 *   手算 1 步 AdamW 更新，验证 m, v, θ 的值
 * =============================================================================
 */

#include "test_utils.h"
#include "../c/transformer.h"

int main() {
    TEST_BEGIN("AdamW 优化器");

    /* ──────────────────────────────────────────────
     * 手算 1 步
     *
     * 参数: lr=0.001, β1=0.9, β2=0.999, ε=1e-8, λ=0.01
     * 梯度 g = 1.0（假设只有一个参数）
     * 初始 θ = 0.5, m = 0, v = 0
     *
     * Step 1 (t=1):
     *   m = 0.9×0 + 0.1×1.0 = 0.1
     *   v = 0.999×0 + 0.001×1.0 = 0.001
     *   m̂ = 0.1 / (1 - 0.9^1) = 0.1 / 0.1 = 1.0
     *   v̂ = 0.001 / (1 - 0.999^1) = 0.001 / 0.001 = 1.0
     *   adam_update = 1.0 / (√1.0 + 1e-8) ≈ 1.0
     *   decay_update = 0.01 × 0.5 = 0.005
     *   θ = 0.5 - 0.001 × (1.0 + 0.005) = 0.5 - 0.001005 = 0.498995
     * ────────────────────────────────────────────── */

    printf("--- 手算 1 步 AdamW ---\n\n");

    /* 构建最小模型：只有 1 个参数（简化测试） */
    GPT model;
    model.config.vocab_size = 5;
    model.config.block_size = 4;
    model.config.n_layer = 1;
    model.config.n_head = 1;
    model.config.n_embd = 4;

    int param_sizes[NUM_PER_TENSORS];
    fill_in_parameter_sizes(param_sizes, model.config);
    int total = 0;
    for (int i = 0; i < NUM_PER_TENSORS; i++) total += param_sizes[i];
    model.num_parameters = total;

    model.params_memory = malloc_and_set_params(&model.params, param_sizes);
    model.grads_memory = malloc_grads(&model.grads, param_sizes);
    model.m_memory = (float*)calloc(total, sizeof(float));
    model.v_memory = (float*)calloc(total, sizeof(float));

    int B = 1, T = 2;
    model.batch_size = B;
    model.seq_len = T;

    int act_sizes[13] = {0};
    fill_in_activation_sizes(act_sizes, model.config, B, T);
    int num_acts = 0;
    for (int i = 0; i < 13; i++) num_acts += act_sizes[i];
    model.num_activations = num_acts;
    model.acts_memory = malloc_and_set_activations(&model.acts, act_sizes);

    /* 设置参数初始值 */
    model.params_memory[0] = 0.5f;

    /* 手动设置梯度 */
    model.grads_memory[0] = 1.0f;

    printf("初始: θ[0] = %.6f, g[0] = %.6f\n", model.params_memory[0], model.grads_memory[0]);
    printf("超参数: lr=0.001, β1=0.9, β2=0.999, ε=1e-8, λ=0.01\n\n");

    /* 执行 1 步 AdamW */
    AdamW opt;
    adamw_init(&opt);
    adamw_step(&opt, &model);

    /* 手算验证 */
    float g = 1.0f;
    float m_expected = 0.9f * 0.0f + 0.1f * g;       /* 0.1 */
    float v_expected = 0.999f * 0.0f + 0.001f * g * g; /* 0.001 */
    float bc1 = 1.0f - 0.9f;                          /* 0.1 */
    float bc2 = 1.0f - 0.999f;                        /* 0.001 */
    float m_hat = m_expected / bc1;                   /* 1.0 */
    float v_hat = v_expected / bc2;                   /* 1.0 */
    float adam_update = m_hat / (sqrtf(v_hat) + 1e-8f); /* ≈ 1.0 */
    float decay_update = 0.01f * 0.5f;                /* 0.005 */
    float theta_expected = 0.5f - 0.001f * (adam_update + decay_update);

    printf("--- 手算过程 ---\n");
    printf("  m = 0.9×0 + 0.1×1.0 = %.4f\n", m_expected);
    printf("  v = 0.999×0 + 0.001×1.0 = %.6f\n", v_expected);
    printf("  m̂ = %.4f / %.4f = %.4f\n", m_expected, bc1, m_hat);
    printf("  v̂ = %.6f / %.6f = %.4f\n", v_expected, bc2, v_hat);
    printf("  adam_update = %.4f / (√%.4f + 1e-8) = %.6f\n", m_hat, v_hat, adam_update);
    printf("  decay_update = 0.01 × 0.5 = %.6f\n", decay_update);
    printf("  θ = 0.5 - 0.001 × (%.6f + %.6f) = %.6f\n", adam_update, decay_update, theta_expected);
    print_sep();

    printf("--- 验证 ---\n");
    ASSERT_CLOSE(model.params_memory[0], theta_expected, 1e-5, "θ[0] after 1 step");
    ASSERT_CLOSE(model.m_memory[0], m_expected, 1e-5, "m[0] (一阶矩)");
    ASSERT_CLOSE(model.v_memory[0], v_expected, 1e-5, "v[0] (二阶矩)");

    /* ──────────────────────────────────────────────
     * 测试 2: 多步更新，loss 应该下降
     * ────────────────────────────────────────────── */

    print_sep();
    printf("--- 测试 2: 多步更新，参数朝最优方向移动 ---\n\n");

    /* 继续执行几步（用同一梯度） */
    float theta_prev = model.params_memory[0];
    for (int step = 0; step < 5; step++) {
        /* 每步重新设置梯度 = 1.0 */
        model.grads_memory[0] = 1.0f;
        adamw_step(&opt, &model);
        printf("step %d: θ[0] = %.6f (变化: %.6f)\n",
               step + 2, model.params_memory[0],
               model.params_memory[0] - theta_prev);
        theta_prev = model.params_memory[0];
    }

    /* 正梯度 → 参数应该减小 */
    ASSERT_TRUE(model.params_memory[0] < 0.5f,
                "正梯度 → 参数减小");

    /* ──────────────────────────────────────────────
     * 测试 3: 零梯度 → 只有 weight decay 起作用
     * ────────────────────────────────────────────── */

    print_sep();
    printf("--- 测试 3: 零梯度 + weight decay ---\n\n");

    float theta_before = model.params_memory[0];
    model.grads_memory[0] = 0.0f;  /* 零梯度 */

    /* 重置 m, v */
    model.m_memory[0] = 0.0f;
    model.v_memory[0] = 0.0f;
    opt.t = 0;  /* 重置步数 */

    adamw_step(&opt, &model);

    printf("零梯度: θ 应该因 weight decay 而减小\n");
    printf("  θ before = %.6f, θ after = %.6f\n", theta_before, model.params_memory[0]);
    ASSERT_TRUE(model.params_memory[0] < theta_before,
                "零梯度 + weight decay → 参数减小");

    gpt_free(&model);

    TEST_END();
}
