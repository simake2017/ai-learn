/* =============================================================================
 * attention.c - 多头因果自注意力（Multi-Head Causal Self-Attention）
 *
 * 输入：inp 是 (B, T, 3C) 拼起来的 Q/K/V（由 matmul 一次性算出）
 * 输出：out 是 (B, T, C)
 *
 * 4 步流程（每个 (b, t, h)）：
 *   1. 算 Q·K^T / sqrt(d)，并记录最大值（数值稳定）
 *   2. exp(x - max)，累加和
 *   3. 归一化（顺便实现 causal mask：未来位置 = 0）
 *   4. 加权求和 V
 *
 * 因果掩码：只算 t2 <= t（位置 t 看不到未来）
 * =============================================================================
 */

#include "transformer.h"

void attention_forward(float* out, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH) {
    int C3 = C * 3;          // Q/K/V 拼起来
    int hs = C / NH;         // 每个 head 维度
    float scale = 1.0f / sqrtf((float)hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                // 取出位置 (b, t, h) 的 Q
                // Q 在 inp 的前 C 维
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // preatt[b, h, t, :] 和 att[b, h, t, :]
                float* preatt_bth = preatt + ((b * NH + h) * T + t) * T;
                float* att_bth = att + ((b * NH + h) * T + t) * T;

                // === Step 1: Q · K^T / sqrt(d) ===
                // 顺便找出最大值（softmax 数值稳定）
                float maxval = -1e30f;
                for (int t2 = 0; t2 <= t; t2++) {  // ← 因果 mask：只看 ≤ t
                    // K 在 inp 的中间 C 维
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;

                    // 点积
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;

                    preatt_bth[t2] = val;
                    if (val > maxval) maxval = val;
                }

                // === Step 2: exp(x - max) + 累加 ===
                //
                // expf(x) 就是 e^x（e ≈ 2.71828），这里借它的三个性质：
                //   1. 输出恒为正        → 注意力比例不能是负数
                //   2. 保序（单调递增）   → 打分高的词，比例也高，排序不颠倒
                //   3. 把"分差"变成"倍数" → 分数差 0.7，exp 后变成 ~2 倍关系，
                //                          这就是 softmax 偏爱高分者的原因
                //
                // 为什么要减 maxval？
                //   数学上 softmax 减不减最大值结果一样（分子分母同缩放），
                //   但数值上：若分数是 100，exp(100) 直接溢出成 inf。
                //   先减去本行最高分，最大值变成 exp(0) = 1，其余落在 0~1，绝对安全。
                //
                // 顺便完成因果 mask：未来位置既不参与 exp 也不参与累加，
                // 并显式写 0（等价于教科书"分数设 -∞"的写法，但省一次 exp）。
                float sum = 0.0f;
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        // 分子：exp(分数 - 本行最高分)，暂存到 att_bth（还没归一化）
                        float e = expf(preatt_bth[t2] - maxval);
                        att_bth[t2] = e;
                        sum += e;  // 累加分母：所有分子之和
                    } else {
                        // 因果 mask：未来位置 = 0（0 / sum = 0，所以不必真算 exp）
                        att_bth[t2] = 0.0f;
                    }
                }

                // === Step 3: 归一化（除以分母）===
                //
                // att[t2] = e[t2] / sum，即"该词的分量占全部分量的比例"。
                // 用乘法代替除法：除法比乘法慢，缓存倒数 1/sum 后逐个乘。
                // 归一化后所有 att 加起来恒等于 1，这一性质保证 Step 4 的
                // 加权求和是合法的"按比例混合"（凸组合），而不是无界累加。
                float inv_sum = 1.0f / sum;
                for (int t2 = 0; t2 <= t; t2++) {
                    att_bth[t2] *= inv_sum;  // e / sum → 最终注意力比例
                }

                // === Step 4: 加权求和 V ===
                // V 在 inp 的最后 C 维
                for (int i = 0; i < hs; i++) {
                    float o = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + 2 * C;
                        o += att_bth[t2] * value_t2[i];
                    }
                    // out[b, t, h, i]
                    out[b * T * C + t * C + h * hs + i] = o;
                }
            }
        }
    }
}
/* === attention_forward_bidirectional =========================================
 * 双向注意力（BERT 式，无因果掩码）：每个位置可以看到全部位置（包括未来）。
 * 计算流程与 attention_forward 完全相同，只是把 t2 的循环上限从 t 放开到 T。
 * 用于编码器类任务；也是学习线演示「偷看未来」效果的对照算子。
 * ============================================================================= */
void attention_forward_bidirectional(float* out, float* preatt, float* att,
                                     const float* inp,
                                     int B, int T, int C, int NH) {
    int C3 = C * 3;
    int hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + ((b * NH + h) * T + t) * T;
                float* att_bth = att + ((b * NH + h) * T + t) * T;

                // Step 1: Q · K^T / sqrt(d)，全部 t2 都算（无掩码）
                float maxval = -1e30f;
                for (int t2 = 0; t2 < T; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    preatt_bth[t2] = val;
                    if (val > maxval) maxval = val;
                }

                // Step 2: exp(x - max) + 累加
                float sum = 0.0f;
                for (int t2 = 0; t2 < T; t2++) {
                    float e = expf(preatt_bth[t2] - maxval);
                    att_bth[t2] = e;
                    sum += e;
                }

                // Step 3: 归一化
                float inv_sum = 1.0f / sum;
                for (int t2 = 0; t2 < T; t2++) {
                    att_bth[t2] *= inv_sum;
                }

                // Step 4: 加权求和 V
                for (int i = 0; i < hs; i++) {
                    float o = 0.0f;
                    for (int t2 = 0; t2 < T; t2++) {
                        const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + 2 * C;
                        o += att_bth[t2] * value_t2[i];
                    }
                    out[b * T * C + t * C + h * hs + i] = o;
                }
            }
        }
    }
}
