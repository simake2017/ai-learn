/* =============================================================================
 * test_07_softmax.c - Softmax 单元测试
 *
 * 📚 学习目标：
 *   理解 softmax 如何将 logits（原始分数）转换为概率分布
 *
 * 🧮 数学原理：
 *   softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
 *
 *   数值稳定版（代码中的实现）：
 *   1. max = max(x)                    ← 找最大值
 *   2. exp_i = exp(x_i - max)          ← 减去最大值再 exp
 *   3. softmax(x_i) = exp_i / Σ exp_j  ← 归一化
 *
 * 💡 为什么要减最大值？
 *   exp(1000) = ∞ (溢出)
 *   exp(1000 - 1000) = exp(0) = 1 (安全)
 *   减去常数不影响归一化结果（分子分母同除 exp(max)）
 *
 * 🔍 测试策略：
 *   1. 小值手算
 *   2. 大值验证数值稳定性
 *   3. 验证概率和 = 1
 * =============================================================================
 */

#include "test_utils.h"
#include "../c/transformer.h"

int main() {
    TEST_BEGIN("Softmax 概率转换");

    /* ──────────────────────────────────────────────
     * 测试 1: 3 维向量手算
     *
     * logits = [2.0, 1.0, 0.1]
     *
     * 朴素计算:
     *   exp(2.0) = 7.389
     *   exp(1.0) = 2.718
     *   exp(0.1) = 1.105
     *   sum = 11.212
     *   probs = [0.659, 0.242, 0.099]
     *
     * 数值稳定版 (max=2.0):
     *   exp(0.0)  = 1.000
     *   exp(-1.0) = 0.368
     *   exp(-1.9) = 0.150
     *   sum = 1.518
     *   probs = [0.659, 0.242, 0.098]
     * ────────────────────────────────────────────── */

    printf("--- 测试 1: 手算 3 维 ---\n\n");

    float logits1[] = {2.0f, 1.0f, 0.1f};
    float probs1[3];

    /* B=1, T=1, V=3 */
    softmax_forward(probs1, logits1, 1, 1, 3);

    float e0 = expf(0.0f);
    float e1 = expf(-1.0f);
    float e2 = expf(-1.9f);
    float sum_e = e0 + e1 + e2;

    printf("logits = [2.0, 1.0, 0.1], max = 2.0\n");
    printf("exp(x-max) = [%.3f, %.3f, %.3f], sum = %.3f\n", e0, e1, e2, sum_e);
    print_sep();

    ASSERT_CLOSE(probs1[0], e0 / sum_e, 1e-4, "prob[0] ≈ 0.659");
    ASSERT_CLOSE(probs1[1], e1 / sum_e, 1e-4, "prob[1] ≈ 0.242");
    ASSERT_CLOSE(probs1[2], e2 / sum_e, 1e-4, "prob[2] ≈ 0.098");

    /* 验证和 = 1 */
    float total = probs1[0] + probs1[1] + probs1[2];
    ASSERT_CLOSE(total, 1.0f, 1e-5, "概率和 = 1.0");

    print_sep();

    /* ──────────────────────────────────────────────
     * 测试 2: 数值稳定性（大 logits）
     *
     * logits = [1000, 1001, 1002]
     * 朴素 exp 会溢出，但数值稳定版应该正常工作
     * ────────────────────────────────────────────── */

    printf("--- 测试 2: 数值稳定性 ---\n\n");

    float logits2[] = {1000.0f, 1001.0f, 1002.0f};
    float probs2[3];

    softmax_forward(probs2, logits2, 1, 1, 3);

    print_vec("大 logits", logits2, 3);
    print_vec("probs (应该不溢出)", probs2, 3);

    float total2 = probs2[0] + probs2[1] + probs2[2];
    ASSERT_CLOSE(total2, 1.0f, 1e-5, "大值: 概率和 = 1.0");

    /* 比例关系: exp(-2) : exp(-1) : exp(0) */
    float r0 = expf(-2.0f) / (expf(-2.0f) + expf(-1.0f) + expf(0.0f));
    float r1 = expf(-1.0f) / (expf(-2.0f) + expf(-1.0f) + expf(0.0f));
    float r2 = expf(0.0f)  / (expf(-2.0f) + expf(-1.0f) + expf(0.0f));
    ASSERT_CLOSE(probs2[0], r0, 1e-5, "大值: prob[0]");
    ASSERT_CLOSE(probs2[1], r1, 1e-5, "大值: prob[1]");
    ASSERT_CLOSE(probs2[2], r2, 1e-5, "大值: prob[2]");

    /* ──────────────────────────────────────────────
     * 测试 3: 等值 logits → 均匀分布
     * ────────────────────────────────────────────── */

    print_sep();
    printf("--- 测试 3: 等值 → 均匀分布 ---\n\n");

    float logits3[] = {5.0f, 5.0f, 5.0f, 5.0f};
    float probs3[4];

    softmax_forward(probs3, logits3, 1, 1, 4);

    ASSERT_CLOSE(probs3[0], 0.25f, 1e-5, "均匀: prob[0] = 0.25");
    ASSERT_CLOSE(probs3[1], 0.25f, 1e-5, "均匀: prob[1] = 0.25");
    ASSERT_CLOSE(probs3[2], 0.25f, 1e-5, "均匀: prob[2] = 0.25");
    ASSERT_CLOSE(probs3[3], 0.25f, 1e-5, "均匀: prob[3] = 0.25");

    TEST_END();
}
