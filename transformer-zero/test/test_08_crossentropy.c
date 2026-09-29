/* =============================================================================
 * test_08_crossentropy.c - 交叉熵损失函数单元测试
 *
 * 📚 学习目标：
 *   理解交叉熵如何衡量"预测概率"和"真实答案"之间的差距
 *
 * 🧮 数学原理：
 *   CrossEntropy = -log(P(target))
 *
 *   对每个 (b, t) 位置：
 *     target = 正确答案的 token id
 *     p = probs[b, t, target]  ← 模型给正确答案分配的概率
 *     loss = -log(p)
 *
 * 💡 直觉：
 *   - 模型预测正确（p 接近 1）→ -log(1) = 0 → loss 小 ✅
 *   - 模型预测错误（p 接近 0）→ -log(0) = ∞ → loss 大 ❌
 *   - loss 就是"模型对正确答案的惊讶程度"
 *
 * 📊 关键值：
 *   p = 1.0  → loss = 0       （完美预测）
 *   p = 0.5  → loss = 0.693   （50-50 猜测）
 *   p = 0.1  → loss = 2.303   （很不确定）
 *   p = 0.01 → loss = 4.605   （几乎不知道）
 * =============================================================================
 */

#include "test_utils.h"
#include "../c/transformer.h"

int main() {
    TEST_BEGIN("交叉熵损失函数");

    /* ──────────────────────────────────────────────
     * 测试 1: 手算
     *
     * B=1, T=2, V=3 (词表 3 个 token)
     *
     * probs = | 0.7  0.2  0.1 |   target = [0, 2]
     *         | 0.1  0.1  0.8 |
     *
     * loss[0] = -log(0.7) = 0.3567   ← 第 1 个位置，答案是 token 0
     * loss[1] = -log(0.8) = 0.2231   ← 第 2 个位置，答案是 token 2
     * ────────────────────────────────────────────── */

    printf("--- 测试 1: 手算 ---\n\n");

    int B = 1, T = 2, V = 3;
    float probs[] = {0.7f, 0.2f, 0.1f,    /* position 0 */
                     0.1f, 0.1f, 0.8f};   /* position 1 */
    int targets[] = {0, 2};                /* 正确答案 */
    float losses[2];

    print_mat("probs", probs, T, V);
    print_ivec("targets", targets, T);
    print_sep();

    crossentropy_forward(losses, probs, targets, B, T, V);

    float expected0 = -logf(0.7f);
    float expected1 = -logf(0.8f);

    printf("loss[0] = -log(0.7) = %.4f\n", expected0);
    printf("loss[1] = -log(0.8) = %.4f\n", expected1);
    print_sep();

    ASSERT_CLOSE(losses[0], expected0, 1e-4, "loss[0] = -log(0.7)");
    ASSERT_CLOSE(losses[1], expected1, 1e-4, "loss[1] = -log(0.8)");

    /* ──────────────────────────────────────────────
     * 测试 2: 概率越高 → loss 越低
     * ────────────────────────────────────────────── */

    print_sep();
    printf("--- 测试 2: 概率 vs loss 关系 ---\n\n");

    float probs2[] = {0.99f, 0.005f, 0.005f,    /* 很确定 → token 0 */
                      0.01f, 0.98f,  0.01f,     /* 很确定 → token 1 */
                      0.33f, 0.34f,  0.33f};    /* 很不确定 → token 2 */
    int targets2[] = {0, 1, 2};
    float losses2[3];

    crossentropy_forward(losses2, probs2, targets2, 1, 3, 3);

    printf("高概率 (p=0.99): loss = %.4f\n", losses2[0]);
    printf("高概率 (p=0.98): loss = %.4f\n", losses2[1]);
    printf("低概率 (p=0.33): loss = %.4f\n", losses2[2]);

    ASSERT_TRUE(losses2[0] < losses2[2], "高概率 loss < 低概率 loss");
    ASSERT_TRUE(losses2[1] < losses2[2], "高概率 loss < 低概率 loss");

    /* ──────────────────────────────────────────────
     * 测试 3: 极低概率的保护
     *
     * probs[target] 接近 0 → -log(0) = -inf
     * 代码中用 p < 1e-30 时 clamp 到 1e-30
     * ────────────────────────────────────────────── */

    print_sep();
    printf("--- 测试 3: 极低概率保护 ---\n\n");

    float probs3[] = {1.0f, 0.0f, 0.0f};  /* token 1 概率 = 0 */
    int targets3[] = {1};                   /* 但答案是 token 1 */
    float losses3[1];

    crossentropy_forward(losses3, probs3, targets3, 1, 1, 3);

    printf("p=0.0, target=1: loss = -log(1e-30) = %.2f (被 clamp)\n", losses3[0]);
    ASSERT_CLOSE(losses3[0], -logf(1e-30f), 1.0f, "极低概率被 clamp");
    ASSERT_TRUE(losses3[0] < 1e10f, "不是 inf (被保护了)");

    TEST_END();
}
