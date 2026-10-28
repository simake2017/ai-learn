/* =============================================================================
 * test_01_embedding.c - Token Embedding + Position Embedding 单元测试
 *
 * 📚 学习目标：
 *   理解 embedding_forward 如何将 token id 转换为稠密向量
 *
 * 🧮 数学原理：
 *   out[b, t, :] = wte[token_id, :] + wpe[t, :]
 *
 *   - wte (word token embedding): 每个词一个 C 维向量，通过查表获取
 *   - wpe (word position embedding): 每个位置一个 C 维向量，表示"第几个词"
 *   - 两者相加 = 既有"这个词是什么意思"，又有"它在句子的什么位置"
 *
 * 🔍 测试策略：
 *   1. 手动设置 wte 和 wpe 的值（全 1、全 2 等，方便手算）
 *   2. 调用 embedding_forward
 *   3. 验证输出 = wte[token] + wpe[pos]
 * =============================================================================
 */

#include "test_utils.h"
#include "../c/transformer.h"

int main() {
    TEST_BEGIN("Embedding 查表 + 位置编码");

    /* ──────────────────────────────────────────────
     * 测试参数
     * B = 1 (1 个句子)
     * T = 3 (3 个 token)
     * C = 4 (每个 token 用 4 维向量表示)
     * V = 5 (词表大小 5 个 token: 0~4)
     * ────────────────────────────────────────────── */
    int B = 1, T = 3, C = 4, V = 5;

    /* --- 构造 wte: 词表矩阵 (V × C) ---
     *
     * token 0 → [0.1, 0.1, 0.1, 0.1]
     * token 1 → [0.2, 0.2, 0.2, 0.2]
     * token 2 → [0.3, 0.3, 0.3, 0.3]
     * token 3 → [0.4, 0.4, 0.4, 0.4]
     * token 4 → [0.5, 0.5, 0.5, 0.5]
     *
     * 💡 每个 token 有一个固定向量，就像字典里每个词有一个"含义坐标"
     */
    float wte[V * C];
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < C; j++) {
            wte[i * C + j] = (i + 1) * 0.1f;
        }
    }
    print_mat("wte (词表)", wte, V, C);

    /* --- 构造 wpe: 位置矩阵 (T × C) ---
     *
     * pos 0 → [1.0, 1.0, 1.0, 1.0]
     * pos 1 → [2.0, 2.0, 2.0, 2.0]
     * pos 2 → [3.0, 3.0, 3.0, 3.0]
     *
     * 💡 位置编码让模型知道"这个词在句子的第几个位置"
     *    真实 Transformer 用 sin/cos 或可学习的位置编码
     */
    float wpe[T * C];
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < C; j++) {
            wpe[i * C + j] = (float)(i + 1);
        }
    }
    print_mat("wpe (位置)", wpe, T, C);

    /* --- 输入 token 序列: [2, 0, 4] ---
     *
     * 位置 0 → token 2 (词向量 [0.3, 0.3, 0.3, 0.3])
     * 位置 1 → token 0 (词向量 [0.1, 0.1, 0.1, 0.1])
     * 位置 2 → token 4 (词向量 [0.5, 0.5, 0.5, 0.5])
     */
    int inputs[3] = {2, 0, 4};
    print_ivec("inputs (token id)", inputs, T);
    print_sep();

    /* --- 运行 embedding_forward --- */
    float out[B * T * C];
    embedding_forward(out, inputs, wte, wpe, B, T, C);

    /* --- 验证结果 --- */

    /* 位置 0: token 2 → wte[2] + wpe[0] = [0.3+1.0, 0.3+1.0, 0.3+1.0, 0.3+1.0] */
    printf("位置 0 (token=2): wte[2] + wpe[0] = [0.3+1.0, ...] = [1.3, ...]\n");
    ASSERT_CLOSE(out[0], 1.3f, 1e-5, "out[0][0]");
    ASSERT_CLOSE(out[1], 1.3f, 1e-5, "out[0][1]");
    ASSERT_CLOSE(out[2], 1.3f, 1e-5, "out[0][2]");
    ASSERT_CLOSE(out[3], 1.3f, 1e-5, "out[0][3]");
    print_sep();

    /* 位置 1: token 0 → wte[0] + wpe[1] = [0.1+2.0, ...] = [2.1, ...] */
    printf("位置 1 (token=0): wte[0] + wpe[1] = [0.1+2.0, ...] = [2.1, ...]\n");
    ASSERT_CLOSE(out[4], 2.1f, 1e-5, "out[1][0]");
    ASSERT_CLOSE(out[5], 2.1f, 1e-5, "out[1][1]");
    ASSERT_CLOSE(out[6], 2.1f, 1e-5, "out[1][2]");
    ASSERT_CLOSE(out[7], 2.1f, 1e-5, "out[1][3]");
    print_sep();

    /* 位置 2: token 4 → wte[4] + wpe[2] = [0.5+3.0, ...] = [3.5, ...] */
    printf("位置 2 (token=4): wte[4] + wpe[2] = [0.5+3.0, ...] = [3.5, ...]\n");
    ASSERT_CLOSE(out[8],  3.5f, 1e-5, "out[2][0]");
    ASSERT_CLOSE(out[9],  3.5f, 1e-5, "out[2][1]");
    ASSERT_CLOSE(out[10], 3.5f, 1e-5, "out[2][2]");
    ASSERT_CLOSE(out[11], 3.5f, 1e-5, "out[2][3]");

    print_sep();
    print_mat("output (B×T×C)", out, B * T, C);

    TEST_END();
}

