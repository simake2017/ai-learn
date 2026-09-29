/* =============================================================================
 * test_05_attention.c - 多头因果自注意力单元测试
 *
 * 📚 学习目标：
 *   理解 attention_forward 的完整流程：
 *   Q·K^T/√d → causal mask → softmax → 加权求和 V
 *
 * 🧮 数学原理（每个 head, 每个 token 位置）：
 *
 *   Step 1: score[t1][t2] = Q[t1] · K[t2] / √d_k
 *   Step 2: 因果掩码 → t2 > t1 的位置设为 -∞（看不到未来）
 *   Step 3: att[t1] = softmax(score[t1])
 *   Step 4: out[t1] = Σ_t2 att[t1][t2] × V[t2]
 *
 * 💡 因果掩码 (Causal Mask)：
 *   位置 t 只能"看到"位置 0~t 的 token，不能看未来
 *   这让模型训练时可以一次处理整段文本（teacher forcing）
 *
 * 🔍 测试策略：
 *   B=1, T=2, C=4, NH=2 (2 个 head, 每个 head 2 维)
 *   手算完整的注意力过程
 * =============================================================================
 */

#include "test_utils.h"
#include "../c/transformer.h"

int main() {
    TEST_BEGIN("因果自注意力 (Causal Self-Attention)");

    /* ──────────────────────────────────────────────
     * 参数设置
     * B=1 (1 个句子), T=2 (2 个 token),
     * C=4 (总维度), NH=2 (2 个 head)
     * hs = C/NH = 2 (每个 head 2 维)
     *
     * attention_forward 的输入是已经拼好的 QKV: (B, T, 3C)
     * 3C = 12, 所以每个 token 有 12 维: [Q(4) | K(4) | V(4)]
     * ────────────────────────────────────────────── */

    int B = 1, T = 2, C = 4, NH = 2;
    int C3 = 3 * C;  /* = 12 */
    int hs = C / NH;  /* = 2 */

    /* 构造 QKV 输入 (1 × 2 × 12)
     *
     * Token 0:
     *   Q[0] = [1, 0,  1, 0]    ← head0: [1,0], head1: [1,0]
     *   K[0] = [1, 0,  1, 0]
     *   V[0] = [10, 20,  30, 40]
     *
     * Token 1:
     *   Q[1] = [0, 1,  0, 1]    ← head0: [0,1], head1: [0,1]
     *   K[1] = [0, 1,  0, 1]
     *   V[1] = [50, 60,  70, 80]
     *
     * 拼接: [Q0 | K0 | V0 | Q1 | K1 | V1]
     */
    float qkv[2 * 12] = {
        /* Token 0: Q(4) K(4) V(4) */
        1, 0, 1, 0,     /* Q[0] */
        1, 0, 1, 0,     /* K[0] */
        10, 20, 30, 40, /* V[0] */
        /* Token 1: Q(4) K(4) V(4) */
        0, 1, 0, 1,     /* Q[1] */
        0, 1, 0, 1,     /* K[1] */
        50, 60, 70, 80  /* V[1] */
    };

    float out[B * T * C];       /* (1, 2, 4) */
    float preatt[B * NH * T * T]; /* (1, 2, 2, 2) */
    float att[B * NH * T * T];    /* (1, 2, 2, 2) */

    memset(out, 0, sizeof(out));
    memset(preatt, 0, sizeof(preatt));
    memset(att, 0, sizeof(att));

    printf("--- 输入 QKV ---\n\n");
    printf("Token 0: Q=[1,0,1,0] K=[1,0,1,0] V=[10,20,30,40]\n");
    printf("Token 1: Q=[0,1,0,1] K=[0,1,0,1] V=[50,60,70,80]\n");
    printf("NH=2, hs=2 (每个 head 看 2 维)\n\n");

    attention_forward(out, preatt, att, qkv, B, T, C, NH);

    /* ──────────────────────────────────────────────
     * 手算 Head 0 (维度 0, 1)
     *
     * Q[0]_h0 = [1, 0],  K[0]_h0 = [1, 0],  V[0]_h0 = [10, 20]
     * Q[1]_h0 = [0, 1],  K[1]_h0 = [0, 1],  V[1]_h0 = [50, 60]
     *
     * scale = 1/√2 ≈ 0.7071
     *
     * --- Position 0 (只看自己) ---
     * score[0][0] = [1,0]·[1,0] × 0.7071 = 1 × 0.7071 = 0.7071
     * score[0][1] = -∞ (因果掩码，看不到未来)
     * att[0] = softmax([0.7071, -∞]) = [1.0, 0.0]
     * out[0]_h0 = 1.0 × V[0]_h0 + 0.0 × V[1]_h0 = [10, 20]
     *
     * --- Position 1 (看自己 + token 0) ---
     * score[1][0] = [0,1]·[1,0] × 0.7071 = 0 × 0.7071 = 0
     * score[1][1] = [0,1]·[0,1] × 0.7071 = 1 × 0.7071 = 0.7071
     * att[1] = softmax([0, 0.7071])
     *        = [e^0/(e^0+e^0.7071), e^0.7071/(e^0+e^0.7071)]
     *        = [1/(1+2.028), 2.028/(1+2.028)]
     *        = [0.3303, 0.6697]
     * out[1]_h0 = 0.3303 × [10,20] + 0.6697 × [50,60]
     *           = [3.303+33.485, 6.606+40.182]
     *           = [36.788, 46.788]
     * ────────────────────────────────────────────── */

    printf("--- 手算验证 Head 0 ---\n\n");

    /* Position 0: 只看自己 → 输出 = V[0] */
    printf("Position 0: 只看自己, att=[1.0, 0.0]\n");
    printf("  out[0]_h0 = 1.0 × [10,20] = [10, 20]\n");
    ASSERT_CLOSE(out[0], 10.0f, 1e-2, "out[0][h0,d0] = V[0][0]");
    ASSERT_CLOSE(out[1], 20.0f, 1e-2, "out[0][h0,d1] = V[0][1]");
    print_sep();

    /* Position 1: 看两个 token
     *
     * 输出布局: out[b * T * C + t * C + h * hs + i]
     *   (b=0, t=0, h=0): out[0], out[1]
     *   (b=0, t=0, h=1): out[2], out[3]
     *   (b=0, t=1, h=0): out[4], out[5]   ← 我们要验证的
     *   (b=0, t=1, h=1): out[6], out[7]
     */
    float scale = 1.0f / sqrtf((float)hs);
    printf("Position 1: score = [0, %.4f], scale = %.4f\n", 0.7071f, scale);
    float e0 = expf(0.0f);
    float e1 = expf(1.0f * scale);  /* [0,1]·[0,1] = 1, × scale */
    float sum_e = e0 + e1;
    float a0 = e0 / sum_e;
    float a1 = e1 / sum_e;
    printf("  att = [%.4f, %.4f]\n", a0, a1);

    float expected_d0 = a0 * 10.0f + a1 * 50.0f;
    float expected_d1 = a0 * 20.0f + a1 * 60.0f;
    printf("  out[1]_h0 = [%.3f, %.3f]\n", expected_d0, expected_d1);

    /* out[4] = out[b=0, t=1, h=0, d=0], out[5] = out[b=0, t=1, h=0, d=1] */
    ASSERT_CLOSE(out[4], expected_d0, 1e-2, "out[1][h0,d0]");
    ASSERT_CLOSE(out[5], expected_d1, 1e-2, "out[1][h0,d1]");

    print_sep();

    /* ──────────────────────────────────────────────
     * 验证因果掩码：att[0][1] 应该 = 0
     * ────────────────────────────────────────────── */

    printf("--- 验证因果掩码 ---\n\n");
    /* att 的布局: att[b * NH * T * T + h * T * T + t1 * T + t2] */
    float att_00 = att[0 * NH * T * T + 0 * T * T + 0 * T + 0]; /* h0, t=0, 看 t=0 */
    float att_01 = att[0 * NH * T * T + 0 * T * T + 0 * T + 1]; /* h0, t=0, 看 t=1 */

    printf("att[h0, t=0]: [看t=0: %.4f, 看t=1: %.4f]\n", att_00, att_01);
    ASSERT_CLOSE(att_00, 1.0f, 1e-5, "att[0→0] = 1.0 (只看自己)");
    ASSERT_CLOSE(att_01, 0.0f, 1e-5, "att[0→1] = 0.0 (因果掩码)");

    TEST_END();
}
