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
     * 参数配置: B=1, T=2 (2 个 token), C=3 (3 维特征)
     *
     * 输入 x (2 × 3):
     *   token 0: [1.0, 2.0, 3.0]
     *   token 1: [4.0, 5.0, 6.0]
     *
     * W_Q (3×3，恒等特征抽取):
     *   | 1 0 0 |  → Q0 = [1, 2, 3],  Q1 = [4, 5, 6]
     *   | 0 1 0 |
     *   | 0 0 1 |
     *
     * W_K (3×3，逆序特征抽取):
     *   | 0 0 1 |  → K0 = [3, 2, 1],  K1 = [6, 5, 4]
     *   | 0 1 0 |
     *   | 1 0 0 |
     *
     * W_V (3×3，组合特征抽取):
     *   | 1 1 0 |  → V0 = [1+2, 2+3, 1+3] = [3, 5, 4]
     *   | 0 1 1 |  → V1 = [4+5, 5+6, 4+6] = [9, 11, 10]
     *   | 1 0 1 |
     *
     * 输出形状: (B, T, 3C) = (1, 2, 9) 共 18 个数
     * 内存排布: 每个 token 独占一段 3C (9) 维连续内存 [Q | K | V]
     * ────────────────────────────────────────────── */

    int B = 1, T = 2, C = 3;

    float x[] = {
        1.0f, 2.0f, 3.0f,   /* token 0 */ // mrd 某个token 的各个特征维度 集合,里面是特征1 特征2 特征3
        4.0f, 5.0f, 6.0f    /* token 1 */
    };

    /* 拼接 W_QKV: (3C × C) = (9 × 3)
     * 前 3 行 = W_Q, 中间 3 行 = W_K, 后 3 行 = W_V */
    float qkvw[] = {
        /* W_Q (3×3) */
        1, 0, 0, // mrd** 根据特征维度 查询某个 key1
        0, 1, 0, // mrd** 根据特征维度 查询某个 key2
        0, 0, 1,
        /* W_K (3×3) */
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
        /* W_V (3×3) */
        1, 1, 0,
        0, 1, 1,
        1, 0, 1
    };

    float qkvb[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  /* 9 个 bias 全 0 */
    float out[18];  /* (B=1, T=2, 3C=9) = 18 个元素 */

    print_mat("x (2个 Token, 每个3维)", x, T, C);
    print_mat("W_QKV (9×3, 包含WQ, WK, WV)", qkvw, 3 * C, C);
    print_sep();

    /* 调用 matmul_forward: out = x @ W_qkv^T + b */
    matmul_forward(out, x, qkvw, qkvb, B, T, C, 3 * C);

    print_mat("QKV 输出 (2行, 每行9维 [Q|K|V])", out, T, 3 * C);
    print_sep();

    /* ──────────────────────────────────────────────
     * 拆分验证: 验证内存排布与手算一致性
     * 注意：对每个 token t，其在 out 中的偏移是 t * 3C
     * ────────────────────────────────────────────── */
    printf("--- 拆分 Token 0 的 Q, K, V ---\n\n");
    float* Q0 = out + 0 * (3 * C);
    float* K0 = Q0 + C;
    float* V0 = K0 + C;
    print_vec("Q0", Q0, C);
    print_vec("K0", K0, C);
    print_vec("V0", V0, C);

    printf("\nToken 0 验证:\n");
    ASSERT_CLOSE(Q0[0], 1.0f, 1e-5, "Q0[0] = 1");
    ASSERT_CLOSE(Q0[1], 2.0f, 1e-5, "Q0[1] = 2");
    ASSERT_CLOSE(Q0[2], 3.0f, 1e-5, "Q0[2] = 3");

    ASSERT_CLOSE(K0[0], 3.0f, 1e-5, "K0[0] = 3");
    ASSERT_CLOSE(K0[1], 2.0f, 1e-5, "K0[1] = 2");
    ASSERT_CLOSE(K0[2], 1.0f, 1e-5, "K0[2] = 1");

    ASSERT_CLOSE(V0[0], 3.0f, 1e-5, "V0[0] = 1+2 = 3");
    ASSERT_CLOSE(V0[1], 5.0f, 1e-5, "V0[1] = 2+3 = 5");
    ASSERT_CLOSE(V0[2], 4.0f, 1e-5, "V0[2] = 1+3 = 4");
    print_sep();

    printf("--- 拆分 Token 1 的 Q, K, V ---\n\n");
    float* Q1 = out + 1 * (3 * C);
    float* K1 = Q1 + C;
    float* V1 = K1 + C;
    print_vec("Q1", Q1, C);
    print_vec("K1", K1, C);
    print_vec("V1", V1, C);

    printf("\nToken 1 验证:\n");
    ASSERT_CLOSE(Q1[0], 4.0f, 1e-5, "Q1[0] = 4");
    ASSERT_CLOSE(Q1[1], 5.0f, 1e-5, "Q1[1] = 5");
    ASSERT_CLOSE(Q1[2], 6.0f, 1e-5, "Q1[2] = 6");

    ASSERT_CLOSE(K1[0], 6.0f, 1e-5, "K1[0] = 6");
    ASSERT_CLOSE(K1[1], 5.0f, 1e-5, "K1[1] = 5");
    ASSERT_CLOSE(K1[2], 4.0f, 1e-5, "K1[2] = 4");

    ASSERT_CLOSE(V1[0], 9.0f,  1e-5, "V1[0] = 4+5 = 9");
    ASSERT_CLOSE(V1[1], 11.0f, 1e-5, "V1[1] = 5+6 = 11");
    ASSERT_CLOSE(V1[2], 10.0f, 1e-5, "V1[2] = 4+6 = 10");
    print_sep();

    /* ──────────────────────────────────────────────
     * 测试 2: 带 bias
     * ────────────────────────────────────────────── */
    printf("--- 测试 2: 带 bias ---\n\n");

    float bias2[] = {
        0.5f,  0.5f,  0.5f,   /* Q 的 bias */
        1.0f,  1.0f,  1.0f,   /* K 的 bias */
       -1.0f, -1.0f, -1.0f    /* V 的 bias */
    };
    float out2[18];

    matmul_forward(out2, x, qkvw, bias2, B, T, C, 3 * C);

    float* Q0_b = out2 + 0 * (3 * C);
    float* K0_b = Q0_b + C;
    float* V0_b = K0_b + C;

    printf("Token 0 叠加 bias 验证:\n");
    ASSERT_CLOSE(Q0_b[0], 1.5f, 1e-5, "Q0_b[0] = 1 + 0.5");
    ASSERT_CLOSE(Q0_b[1], 2.5f, 1e-5, "Q0_b[1] = 2 + 0.5");
    ASSERT_CLOSE(Q0_b[2], 3.5f, 1e-5, "Q0_b[2] = 3 + 0.5");

    ASSERT_CLOSE(K0_b[0], 4.0f, 1e-5, "K0_b[0] = 3 + 1.0");
    ASSERT_CLOSE(K0_b[1], 3.0f, 1e-5, "K0_b[1] = 2 + 1.0");
    ASSERT_CLOSE(K0_b[2], 2.0f, 1e-5, "K0_b[2] = 1 + 1.0");

    ASSERT_CLOSE(V0_b[0], 2.0f, 1e-5, "V0_b[0] = 3 - 1.0");
    ASSERT_CLOSE(V0_b[1], 4.0f, 1e-5, "V0_b[1] = 5 - 1.0");
    ASSERT_CLOSE(V0_b[2], 3.0f, 1e-5, "V0_b[2] = 4 - 1.0");

    TEST_END();
}
