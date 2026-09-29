/* =============================================================================
 * test_06_gelu.c - GELU 激活函数单元测试
 *
 * 📚 学习目标：
 *   理解 GELU (Gaussian Error Linear Unit) 的工作原理
 *
 * 🧮 数学原理：
 *   GELU(x) = x × Φ(x)
 *
 *   其中 Φ(x) 是标准正态分布的累积分布函数（CDF）
 *   近似实现: GELU(x) ≈ 0.5x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
 *
 * 💡 GELU vs ReLU：
 *   - ReLU(x) = max(0, x) → 简单粗暴，负数直接砍掉
 *   - GELU(x) → 负数附近"软着陆"，不是直接归零
 *   - GELU 在 BERT、GPT 等模型中广泛使用
 *
 * 📊 关键值：
 *   GELU(0)  ≈ 0
 *   GELU(1)  ≈ 0.8413
 *   GELU(-1) ≈ -0.1587
 *   GELU(2)  ≈ 1.9545
 *   GELU(-2) ≈ -0.0455
 * =============================================================================
 */

#include "test_utils.h"
#include "../c/transformer.h"

int main() {
    TEST_BEGIN("GELU 激活函数");

    /* ──────────────────────────────────────────────
     * 测试 1: 已知值验证
     * ────────────────────────────────────────────── */

    printf("--- 测试 1: 已知值 ---\n\n");

    float input[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f};
    float expected[] = {0.0f, 0.8413f, -0.1587f, 1.9545f, -0.0455f, 0.3457f, -0.1543f};
    int n = 7;
    float out[7];

    gelu_forward(out, input, n);

    for (int i = 0; i < n; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "GELU(%.1f) = %.4f", input[i], expected[i]);
        ASSERT_CLOSE(out[i], expected[i], 0.01f, msg);
    }

    print_sep();
    print_vec("input", input, n);
    print_vec("GELU(input)", out, n);

    /* ──────────────────────────────────────────────
     * 测试 2: GELU 的性质
     *
     * 1. GELU(0) ≈ 0（过原点）
     * 2. x > 0 时 GELU(x) ≈ x（大正数接近恒等）
     * 3. x < 0 时 GELU(x) ≈ 0（大负数接近零）
     * 4. 不是严格的偶函数或奇函数
     * ────────────────────────────────────────────── */

    print_sep();
    printf("--- 测试 2: GELU 性质验证 ---\n\n");

    float big[] = {5.0f, -5.0f, 10.0f, -10.0f};
    float big_out[4];
    gelu_forward(big_out, big, 4);

    printf("大正数: GELU(5) ≈ 5, GELU(10) ≈ 10\n");
    ASSERT_CLOSE(big_out[0], 5.0f, 0.01f, "GELU(5) ≈ 5 (接近恒等)");
    ASSERT_CLOSE(big_out[2], 10.0f, 0.01f, "GELU(10) ≈ 10 (接近恒等)");

    printf("大负数: GELU(-5) ≈ 0, GELU(-10) ≈ 0\n");
    ASSERT_CLOSE(big_out[1], 0.0f, 0.01f, "GELU(-5) ≈ 0 (接近零)");
    ASSERT_CLOSE(big_out[3], 0.0f, 0.01f, "GELU(-10) ≈ 0 (接近零)");

    /* GELU(0) = 0 */
    float zero[] = {0.0f};
    float zero_out[1];
    gelu_forward(zero_out, zero, 1);
    ASSERT_CLOSE(zero_out[0], 0.0f, 1e-5, "GELU(0) = 0");

    TEST_END();
}
