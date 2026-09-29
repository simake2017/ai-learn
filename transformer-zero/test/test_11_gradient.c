/* =============================================================================
 * test_11_gradient.c - 数值梯度单元测试
 *
 * 📚 学习目标：
 *   理解数值微分（有限差分）如何近似计算梯度
 *
 * 🧮 数学原理：
 *   中心差分法：
 *   ∂L/∂θ_i ≈ [L(θ_i + ε) - L(θ_i - ε)] / (2ε)
 *
 *   为什么用中心差分而不是前向差分？
 *   - 前向差分: [L(θ+ε) - L(θ)] / ε     → 误差 O(ε)
 *   - 中心差分: [L(θ+ε) - L(θ-ε)] / (2ε) → 误差 O(ε²)  ← 更精确！
 *
 * 💡 数值梯度的用途：
 *   1. 验证解析梯度（反向传播）是否正确（gradient checking）
 *   2. 本项目中作为教学用途，直接用来训练（慢但直观）
 *   3. 实际生产中用反向传播（快 1000 倍以上）
 *
 * ⚠️ 缺点：
 *   每个参数需要 2 次完整前向传播
 *   100K 参数 → 200K 次 forward → 非常慢！
 *
 * 🔍 测试策略：
 *   对 1 个参数做数值梯度，验证：
 *   1. 梯度值不为零（参数确实影响 loss）
 *   2. 梯度方向正确（朝梯度方向扰动 → loss 增大）
 * =============================================================================
 */

#include "test_utils.h"
#include "../c/transformer.h"

int main() {
    TEST_BEGIN("数值梯度 (中心差分)");

    /* 构建最小模型 */
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

    srand(42);
    for (int i = 0; i < total; i++) {
        float r1 = (float)rand() / RAND_MAX + 1e-10f;
        float r2 = (float)rand() / RAND_MAX;
        float g = sqrtf(-2.0f * logf(r1)) * cosf(2.0f * 3.14159f * r2);
        model.params_memory[i] = g * 0.02f;
    }

    int B = 1, T = 3;
    model.batch_size = B;
    model.seq_len = T;

    int act_sizes[13] = {0};
    fill_in_activation_sizes(act_sizes, model.config, B, T);
    int num_acts = 0;
    for (int i = 0; i < 13; i++) num_acts += act_sizes[i];
    model.num_activations = num_acts;
    model.acts_memory = malloc_and_set_activations(&model.acts, act_sizes);

    int inputs[]  = {0, 1, 2};
    int targets[] = {1, 2, 3};

    printf("模型: V=%d, C=%d, L=%d, 参数数=%d\n",
           model.config.vocab_size, model.config.n_embd,
           model.config.n_layer, total);
    print_sep();

    /* --- 先算一次基准 loss --- */
    gpt_forward(&model, inputs, targets);
    float base_loss = model.mean_loss;
    printf("基准 loss = %.6f\n\n", base_loss);

    /* --- 对前 5 个参数手动做中心差分 --- */
    printf("--- 手动中心差分 (前 5 个参数) ---\n\n");

    float eps = 1e-3f;
    for (int i = 0; i < 5; i++) {
        float saved = model.params_memory[i];

        /* L(θ + ε) */
        model.params_memory[i] = saved + eps;
        gpt_forward(&model, inputs, targets);
        float L_plus = model.mean_loss;

        /* L(θ - ε) */
        model.params_memory[i] = saved - eps;
        gpt_forward(&model, inputs, targets);
        float L_minus = model.mean_loss;

        /* 中心差分 */
        float grad_manual = (L_plus - L_minus) / (2.0f * eps);

        /* 还原 */
        model.params_memory[i] = saved;

        printf("param[%d] = %.6f\n", i, saved);
        printf("  L(+ε) = %.6f, L(-ε) = %.6f\n", L_plus, L_minus);
        printf("  数值梯度 = %.6f\n", grad_manual);
        print_sep();
    }

    /* --- 用 compute_numerical_grads 批量计算 --- */
    printf("--- 调用 compute_numerical_grads ---\n\n");

    gpt_forward(&model, inputs, targets);  /* 恢复基准 loss */
    compute_numerical_grads(&model, inputs, targets, eps);

    /* 验证：至少有一些梯度不为零 */
    int nonzero = 0;
    for (int i = 0; i < total; i++) {
        if (fabsf(model.grads_memory[i]) > 1e-8f) nonzero++;
    }
    printf("非零梯度: %d / %d\n", nonzero, total);
    ASSERT_TRUE(nonzero > 0, "存在非零梯度");

    /* --- 验证梯度方向 --- */
    print_sep();
    printf("--- 验证梯度方向 ---\n\n");

    /* 找到第一个非零梯度的参数 */
    int test_idx = -1;
    for (int i = 0; i < total; i++) {
        if (fabsf(model.grads_memory[i]) > 1e-6f) {
            test_idx = i;
            break;
        }
    }

    if (test_idx >= 0) {
        float grad = model.grads_memory[test_idx];
        float saved = model.params_memory[test_idx];

        /* 朝梯度正方向扰动 → loss 应该增大 */
        model.params_memory[test_idx] = saved + eps;
        gpt_forward(&model, inputs, targets);
        float L_pos = model.mean_loss;

        /* 朝梯度负方向扰动 → loss 应该减小 */
        model.params_memory[test_idx] = saved - eps;
        gpt_forward(&model, inputs, targets);
        float L_neg = model.mean_loss;

        model.params_memory[test_idx] = saved;

        printf("param[%d]: grad = %.6f\n", test_idx, grad);
        printf("  L(+ε) = %.6f (应该 %s)\n", L_pos, grad > 0 ? "增大" : "减小");
        printf("  L(-ε) = %.6f (应该 %s)\n", L_neg, grad > 0 ? "减小" : "增大");

        if (grad > 0) {
            ASSERT_TRUE(L_pos > base_loss - 1e-4f, "正方向 → loss 增大");
        } else {
            ASSERT_TRUE(L_neg > base_loss - 1e-4f, "负方向 → loss 增大");
        }
    }

    gpt_free(&model);

    TEST_END();
}
