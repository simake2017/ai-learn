/* =============================================================================
 * test_10_forward.c - 完整前向传播集成测试
 *
 * 📚 学习目标：
 *   验证整个 GPT 前向传播流程能正确运行：
 *   embedding → [LayerNorm → Attention → Residual → LayerNorm → MLP → Residual]×N
 *   → Final LayerNorm → LM Head → Softmax → CrossEntropy
 *
 * 🧮 数据流：
 *   inputs (B, T)         ← token id 序列
 *   → embedding (B, T, C) ← 词向量 + 位置编码
 *   → block×N             ← N 个 Transformer Block
 *   → logits (B, T, V)    ← LM Head 输出（每个位置对 V 个 token 的分数）
 *   → probs (B, T, V)     ← softmax 转概率
 *   → loss (B, T)         ← 交叉熵
 *
 * 💡 这个测试验证：
 *   1. 前向传播不崩溃
 *   2. loss 是合理的正数（不是 NaN / Inf）
 *   3. 随机初始化时 loss ≈ log(V)（均匀分布的交叉熵）
 *   4. probs 每行和 = 1
 * =============================================================================
 */

#include "test_utils.h"
#include "../c/transformer.h"

int main() {
    TEST_BEGIN("完整前向传播 (集成测试)");

    /* ──────────────────────────────────────────────
     * 用小模型测试（快 + 容易观察）
     * vocab=10, n_embd=8, n_layer=1, n_head=2
     * ────────────────────────────────────────────── */

    printf("--- 构建小模型 ---\n\n");

    GPT model;
    model.config.vocab_size = 10;
    model.config.block_size = 8;
    model.config.n_layer = 1;
    model.config.n_head = 2;
    model.config.n_embd = 8;

    int param_sizes[NUM_PER_TENSORS];
    fill_in_parameter_sizes(param_sizes, model.config);
    int total = 0;
    for (int i = 0; i < NUM_PER_TENSORS; i++) total += param_sizes[i];
    model.num_parameters = total;

    model.params_memory = malloc_and_set_params(&model.params, param_sizes);
    model.grads_memory = malloc_grads(&model.grads, param_sizes);
    model.m_memory = (float*)calloc(total, sizeof(float));
    model.v_memory = (float*)calloc(total, sizeof(float));

    /* 随机初始化 */
    srand(42);
    for (int i = 0; i < total; i++) {
        float r1 = (float)rand() / RAND_MAX + 1e-10f;
        float r2 = (float)rand() / RAND_MAX;
        float g = sqrtf(-2.0f * logf(r1)) * cosf(2.0f * 3.14159f * r2);
        model.params_memory[i] = g * 0.02f;
    }

    int B = 2, T = 4;
    model.batch_size = B;
    model.seq_len = T;

    int act_sizes[13] = {0};
    fill_in_activation_sizes(act_sizes, model.config, B, T);
    int num_acts = 0;
    for (int i = 0; i < 13; i++) num_acts += act_sizes[i];
    model.num_activations = num_acts;
    model.acts_memory = malloc_and_set_activations(&model.acts, act_sizes);

    printf("模型配置: V=%d, C=%d, L=%d, NH=%d\n",
           model.config.vocab_size, model.config.n_embd,
           model.config.n_layer, model.config.n_head);
    printf("参数数: %d\n", model.num_parameters);
    printf("激活数: %d\n", model.num_activations);
    print_sep();

    /* --- 运行前向传播 --- */
    printf("--- 运行前向传播 ---\n\n");

    int inputs[]  = {1, 3, 5, 7,  2, 4, 6, 8};    /* B=2, T=4 */
    int targets[] = {3, 5, 7, 1,  4, 6, 8, 2};    /* 下一个 token */

    print_ivec("inputs", inputs, B * T);
    print_ivec("targets", targets, B * T);
    print_sep();

    gpt_forward(&model, inputs, targets);

    /* --- 验证 loss --- */
    float loss = model.mean_loss;
    float expected_random_loss = logf((float)model.config.vocab_size);  /* ≈ 2.303 for V=10 */

    printf("Loss = %.4f\n", loss);
    printf("随机初始化的期望 loss ≈ log(V) = log(%d) = %.4f\n",
           model.config.vocab_size, expected_random_loss);
    print_sep();

    ASSERT_TRUE(loss > 0.0f, "loss > 0 (正数)");
    ASSERT_TRUE(!isnan(loss), "loss 不是 NaN");
    ASSERT_TRUE(!isinf(loss), "loss 不是 Inf");
    ASSERT_TRUE(loss < 100.0f, "loss < 100 (合理范围)");

    /* 随机初始化时 loss 应该接近 log(V)，允许一些偏差 */
    ASSERT_TRUE(fabsf(loss - expected_random_loss) < 2.0f,
                "loss 接近 log(V) (随机初始化预期)");

    print_sep();

    /* --- 验证 probs 每行和 = 1 --- */
    printf("--- 验证 softmax 输出 ---\n\n");

    int V = model.config.vocab_size;
    int all_sum_one = 1;
    for (int bt = 0; bt < B * T; bt++) {
        float sum = 0.0f;
        for (int v = 0; v < V; v++) {
            sum += model.acts.probs[bt * V + v];
        }
        if (fabsf(sum - 1.0f) > 1e-4f) {
            all_sum_one = 0;
        }
    }
    ASSERT_TRUE(all_sum_one, "所有 probs 行和 ≈ 1.0");

    /* 打印第一个位置的 prob 分布 */
    print_vec("probs[0] (第 1 个 token 的预测)", model.acts.probs, V);

    /* 清理 */
    gpt_free(&model);

    TEST_END();
}
