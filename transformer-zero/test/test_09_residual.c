/* =============================================================================
 * test_09_residual.c - 残差连接 (Residual Connection) 单元测试
 *
 * 📚 学习目标：
 *   理解残差连接为什么是 Transformer（和所有深度网络）的关键
 *
 * 🧮 数学原理：
 *   output = input + sublayer_output
 *
 *   在 Transformer Block 中：
 *   x_new = x + Attention(LayerNorm(x))    ← 子层 1
 *   x_new = x_new + MLP(LayerNorm(x_new))  ← 子层 2
 *
 * 💡 为什么需要残差连接？
 *
 *   没有残差：x → F(x)
 *     信息只能通过 F 传递。如果 F 学得不好，原始信息就丢了
 *
 *   有残差：x → x + F(x)
 *     即使 F(x) = 0（子层什么都没学到），输出仍然是 x（原始信息保留）
 *     子层只需要学习"补充什么信息"，而不是"重新编码所有信息"
 *
 *   梯度角度：∂L/∂x = ∂L/∂(x+F(x)) × (1 + ∂F/∂x)
 *     即使 ∂F/∂x 很小，梯度仍然可以通过 "1" 这个直连通路回传
 *     解决了深度网络的梯度消失问题
 * =============================================================================
 */

#include "test_utils.h"
#include "../c/transformer.h"

int main() {
    TEST_BEGIN("残差连接 (Residual Connection)");

    /* ──────────────────────────────────────────────
     * 测试 1: 基本加法
     *
     * input = [1, 2, 3, 4]
     * sublayer_output = [10, 20, 30, 40]
     * residual = input + sublayer = [11, 22, 33, 44]
     * ────────────────────────────────────────────── */

    printf("--- 测试 1: 基本残差加法 ---\n\n");

    float inp1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float sub1[] = {10.0f, 20.0f, 30.0f, 40.0f};
    float out1[4];

    print_vec("x (原始输入)", inp1, 4);
    print_vec("F(x) (子层输出)", sub1, 4);

    residual_forward(out1, inp1, sub1, 4);

    print_vec("x + F(x) (残差后)", out1, 4);
    print_sep();

    ASSERT_CLOSE(out1[0], 11.0f, 1e-5, "residual[0] = 1+10");
    ASSERT_CLOSE(out1[1], 22.0f, 1e-5, "residual[1] = 2+20");
    ASSERT_CLOSE(out1[2], 33.0f, 1e-5, "residual[2] = 3+30");
    ASSERT_CLOSE(out1[3], 44.0f, 1e-5, "residual[3] = 4+40");

    /* ──────────────────────────────────────────────
     * 测试 2: F(x) = 0 → 输出 = x（子层没学到东西时）
     *
     * 这展示了残差连接的核心优势：
     * 即使子层完全没用，信息也不会丢失
     * ────────────────────────────────────────────── */

    print_sep();
    printf("--- 测试 2: F(x) = 0 → 信息保留 ---\n\n");

    float inp2[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float zero_sub[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out2[4];

    residual_forward(out2, inp2, zero_sub, 4);

    printf("F(x) = [0, 0, 0, 0] → output = x\n");
    ASSERT_CLOSE(out2[0], 1.0f, 1e-5, "zero sublayer: out[0] = x[0]");
    ASSERT_CLOSE(out2[3], 4.0f, 1e-5, "zero sublayer: out[3] = x[3]");

    /* ──────────────────────────────────────────────
     * 测试 3: 模拟多层残差
     *
     * 模拟 3 个 Transformer Block 的残差累积
     * ────────────────────────────────────────────── */

    print_sep();
    printf("--- 测试 3: 多层残差累积 ---\n\n");

    float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out3[4];

    printf("初始 x: "); print_vec("", x, 4);

    /* Block 1: 子层输出 = [0.1, 0.2, 0.3, 0.4] */
    float f1[] = {0.1f, 0.2f, 0.3f, 0.4f};
    residual_forward(out3, x, f1, 4);
    printf("Block 1: x + F1(x): "); print_vec("", out3, 4);
    ASSERT_CLOSE(out3[0], 1.1f, 1e-5, "block1[0]");

    /* Block 2: 子层输出 = [0.5, 0.5, 0.5, 0.5] */
    float f2[] = {0.5f, 0.5f, 0.5f, 0.5f};
    residual_forward(out3, out3, f2, 4);
    printf("Block 2: x + F2(x): "); print_vec("", out3, 4);
    ASSERT_CLOSE(out3[0], 1.6f, 1e-5, "block2[0]");

    /* Block 3: 子层输出 = [1.0, 1.0, 1.0, 1.0] */
    float f3[] = {1.0f, 1.0f, 1.0f, 1.0f};
    residual_forward(out3, out3, f3, 4);
    printf("Block 3: x + F3(x): "); print_vec("", out3, 4);
    ASSERT_CLOSE(out3[0], 2.6f, 1e-5, "block3[0]");

    printf("\n最终 = 原始x + F1 + F2 + F3（信息逐层叠加）\n");

    TEST_END();
}
