/* =============================================================================
 * test_04_qkv.c - QKV 投影单元测试
 *
 * 📚 学习目标：
 *   理解 Transformer 如何通过一次 matmul 同时生成 Q、K、V 三个矩阵
 *
 * 🧮 数学原理：
 *   QKV 投影本质上是 3 个独立的线性变换，但为了效率合并为 1 个：
 *
 *   [Q | K | V] = x @ W_qkv^T + b_qkv
 *
 *   其中 W_qkv 的形状是 (3C, C)，前 C 行是 W_Q，中间 C 行是 W_K，后 C 行是 W_V
 *   输出的形状是 (B, T, 3C)，前 C 列是 Q，中间 C 列是 K，后 C 列是 V
 *
 * 💡 为什么不分开算 3 次 matmul？
 *   - 一次大矩阵乘法比三次小矩阵乘法快（更好的 GPU 利用率 / cache 命中）
 *   - 这也是为什么参数叫 qkvw 而不是 wq, wk, wv 分开的
 *
 * 🔍 测试策略：
 *   构造已知的 W_Q, W_K, W_V，拼成 W_QKV，验证拆分后的 Q, K, V 正确
 * =============================================================================
 */

#include "test_utils.h"
#include "../c/transformer.h"

int main() {
    TEST_BEGIN("QKV 投影 (一次 matmul 生成 Q/K/V)");

    /* ──────────────────────────────────────────────
     * 简化参数: B=1, T=1, C=2 (2 维词向量)
     *
     * 输入 x = [1.0, 2.0]  (1 个 token, 2 维)
     *
     * W_Q = | 1  0 |    → Q = x @ W_Q^T = [1, 2]
     *       | 0  1 |
     *
     * W_K = | 1  1 |    → K = x @ W_K^T = [3, 3]
     *       | 1  1 |
     *
     * W_V = | 0  1 |    → V = x @ W_V^T = [2, 2]
     *       | 0  1 |
     *
     * bias 全部为 0
     * ────────────────────────────────────────────── */

    int B = 1, T = 1, C = 2;

    float x[] = {1.0f, 2.0f};

    /* 拼接 W_QKV: (3C × C) = (6 × 2)
     * 前 2 行 = W_Q, 中间 2 行 = W_K, 后 2 行 = W_V */
    float qkvw[] = {
        1, 0,    /* W_Q 第 1 行 */
        0, 1,    /* W_Q 第 2 行 */
        1, 1,    /* W_K 第 1 行 */
        1, 1,    /* W_K 第 2 行 */
        0, 1,    /* W_V 第 1 行 */
        0, 1,    /* W_V 第 2 行 */
    };

    float qkvb[] = {0, 0, 0, 0, 0, 0};  /* 6 个 bias 全 0 */
    float out[6];  /* (1, 1, 3C) = 6 个元素 */

    print_vec("x (输入)", x, C);
    print_mat("W_QKV (6×2)", qkvw, 6, 2);
    print_sep();

    /* 调用 matmul_forward: out = x @ W_qkv^T + b */
    matmul_forward(out, x, qkvw, qkvb, B, T, C, 3 * C);

    print_vec("QKV 输出 (拼接)", out, 3 * C);
    print_sep();

    /* 拆分 Q, K, V 并验证 */
    float* Q = out;           /* 前 C 个元素 */
    float* K = out + C;       /* 中间 C 个元素 */
    float* V = out + 2 * C;   /* 后 C 个元素 */

    printf("--- 拆分 Q, K, V ---\n\n");
    print_vec("Q", Q, C);
    print_vec("K", K, C);
    print_vec("V", V, C);
    print_sep();

    printf("--- 手算验证 ---\n");
    printf("Q = [1,2] @ W_Q^T = [1×1+2×0, 1×0+2×1] = [1, 2]\n");
    ASSERT_CLOSE(Q[0], 1.0f, 1e-5, "Q[0]");
    ASSERT_CLOSE(Q[1], 2.0f, 1e-5, "Q[1]");

    printf("K = [1,2] @ W_K^T = [1×1+2×1, 1×1+2×1] = [3, 3]\n");
    ASSERT_CLOSE(K[0], 3.0f, 1e-5, "K[0]");
    ASSERT_CLOSE(K[1], 3.0f, 1e-5, "K[1]");

    printf("V = [1,2] @ W_V^T = [1×0+2×1, 1×0+2×1] = [2, 2]\n");
    ASSERT_CLOSE(V[0], 2.0f, 1e-5, "V[0]");
    ASSERT_CLOSE(V[1], 2.0f, 1e-5, "V[1]");

    print_sep();

    /* ──────────────────────────────────────────────
     * 测试 2: 带 bias
     * ────────────────────────────────────────────── */

    printf("--- 测试 2: 带 bias ---\n\n");

    float bias2[] = {0.5f, 0.5f,    /* Q 的 bias */
                     1.0f, 1.0f,    /* K 的 bias */
                    -1.0f, -1.0f};  /* V 的 bias */
    float out2[6];

    matmul_forward(out2, x, qkvw, bias2, B, T, C, 3 * C);

    printf("Q_bias = [1+0.5, 2+0.5] = [1.5, 2.5]\n");
    ASSERT_CLOSE(out2[0], 1.5f, 1e-5, "Q+bias[0]");
    ASSERT_CLOSE(out2[1], 2.5f, 1e-5, "Q+bias[1]");

    printf("K_bias = [3+1, 3+1] = [4, 4]\n");
    ASSERT_CLOSE(out2[2], 4.0f, 1e-5, "K+bias[0]");
    ASSERT_CLOSE(out2[3], 4.0f, 1e-5, "K+bias[1]");

    printf("V_bias = [2-1, 2-1] = [1, 1]\n");
    ASSERT_CLOSE(out2[4], 1.0f, 1e-5, "V+bias[0]");
    ASSERT_CLOSE(out2[5], 1.0f, 1e-5, "V+bias[1]");

    TEST_END();
}
