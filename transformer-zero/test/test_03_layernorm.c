/* =============================================================================
 * test_03_layernorm.c - Layer Normalization 单元测试
 *
 * 📚 学习目标：
 *   理解 LayerNorm 如何将每个 token 的向量归一化到均值 0、方差 1，
 *   然后再用 gamma/beta 做缩放和平移
 *
 * 🧮 数学原理：
 *   对每个 (b, t) 位置的 C 维向量：
 *
 *   1. μ = (1/C) Σ x_i               ← 均值
 *   2. σ² = (1/C) Σ (x_i - μ)²       ← 方差
 *   3. rstd = 1 / √(σ² + ε)           ← 逆标准差 (ε 防除零)
 *   4. y_i = (x_i - μ) × rstd × γ_i + β_i   ← 归一化 + 缩放 + 平移
 *
 * 💡 为什么需要 LayerNorm？
 *   - 深度网络中，每一层的输出幅度会越来越大（梯度爆炸）
 *   - LN 把输出拉回到"均值 0、方差 1"的标准分布
 *   - γ 和 β 是可学习参数，让模型自己决定"要不要恢复原来的分布"
 *
 * 🔍 测试策略：
 *   手算一个 4 维向量的 LN 过程
 * =============================================================================
 */

#include "test_utils.h"
#include "../c/transformer.h"

int main() {
    TEST_BEGIN("Layer Normalization");

    /* ──────────────────────────────────────────────
     * 手算示例
     *
     * x = [1.0, 2.0, 3.0, 4.0]    (C = 4)
     *
     * Step 1: μ = (1+2+3+4)/4 = 2.5
     * Step 2: σ² = [(1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²] / 4
     *            = [2.25 + 0.25 + 0.25 + 2.25] / 4
     *            = 5.0 / 4 = 1.25
     * Step 3: rstd = 1 / √(1.25 + 1e-5) ≈ 0.89442
     * Step 4: γ = [1, 1, 1, 1], β = [0, 0, 0, 0]（默认值）
     *         y_0 = (1.0 - 2.5) × 0.89442 × 1 + 0 = -1.34164
     *         y_1 = (2.0 - 2.5) × 0.89442 × 1 + 0 = -0.44721
     *         y_2 = (3.0 - 2.5) × 0.89442 × 1 + 0 =  0.44721
     *         y_3 = (4.0 - 2.5) × 0.89442 × 1 + 0 =  1.34164
     * ────────────────────────────────────────────── */

    int B = 1, T = 1, C = 4;

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};  /* 不缩放 */
    float beta[]  = {0.0f, 0.0f, 0.0f, 0.0f};  /* 不平移 */
    float out[4];

    print_vec("x (输入)", x, C);
    print_vec("gamma (γ)", gamma, C);
    print_vec("beta (β)", beta, C);
    print_sep();

    printf("--- 手算过程 ---\n");
    float mean = (1.0f + 2.0f + 3.0f + 4.0f) / 4.0f;
    float var = 1.25f;
    float rstd = 1.0f / sqrtf(var + 1e-5f);
    printf("  μ = %.5f\n", mean);
    printf("  σ² = %.5f\n", var);
    printf("  rstd = 1/√(%.5f + 1e-5) = %.5f\n", var, rstd);
    print_sep();

    layernorm_forward(out, x, gamma, beta, B, T, C);

    printf("--- 验证 ---\n");
    ASSERT_CLOSE(out[0], -1.34164f, 1e-4, "y[0] = (1-2.5)×rstd");
    ASSERT_CLOSE(out[1], -0.44721f, 1e-4, "y[1] = (2-2.5)×rstd");
    ASSERT_CLOSE(out[2],  0.44721f, 1e-4, "y[2] = (3-2.5)×rstd");
    ASSERT_CLOSE(out[3],  1.34164f, 1e-4, "y[3] = (4-2.5)×rstd");

    /* 验证归一化后均值为 0 */
    float out_mean = (out[0] + out[1] + out[2] + out[3]) / 4.0f;
    ASSERT_CLOSE(out_mean, 0.0f, 1e-5, "归一化后均值 ≈ 0");
    print_sep();

    /* ──────────────────────────────────────────────
     * 测试 2: γ 和 β 的作用
     *
     * γ = [2, 2, 2, 2] → 放大 2 倍
     * β = [1, 1, 1, 1] → 平移 +1
     * ────────────────────────────────────────────── */

    printf("--- 测试 2: γ=2, β=1 (缩放+平移) ---\n\n");

    float gamma2[] = {2.0f, 2.0f, 2.0f, 2.0f};
    float beta2[]  = {1.0f, 1.0f, 1.0f, 1.0f};
    float out2[4];

    layernorm_forward(out2, x, gamma2, beta2, B, T, C);

    print_vec("y (γ=2, β=1)", out2, C);

    /* y[i] = normalized[i] * 2 + 1 */
    ASSERT_CLOSE(out2[0], -1.34164f * 2.0f + 1.0f, 1e-4, "y2[0] = norm×2+1");
    ASSERT_CLOSE(out2[3],  1.34164f * 2.0f + 1.0f, 1e-4, "y2[3] = norm×2+1");

    /* 归一化后均值应该 = β = 1 */
    float out2_mean = (out2[0] + out2[1] + out2[2] + out2[3]) / 4.0f;
    ASSERT_CLOSE(out2_mean, 1.0f, 1e-4, "缩放平移后均值 ≈ β = 1");

    TEST_END();
}
