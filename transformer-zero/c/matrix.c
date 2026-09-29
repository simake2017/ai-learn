/* =============================================================================
 * matrix.c - 矩阵运算（matmul）和 residual
 *
 * matmul_forward: y = x @ W^T + b
 *   x: (B, T, C)
 *   W: (OC, C)
 *   b: (OC,) 可选
 *   y: (B, T, OC)
 *
 * residual_forward: out = a + b  (in-place 可能)
 * =============================================================================
 */

#include "transformer.h"

/* matmul_forward
 *
 * 计算 y = x @ W^T + b
 *   x: (B, T, C) 展平 = (B*T, C)
 *   W: (OC, C)
 *   y: (B, T, OC)
 *
 * 数学公式：
 *   y[b, t, o] = sum_i x[b, t, i] * W[o, i] + b[o]
 *
 * 复杂度：O(B * T * OC * C)
 */
void matmul_forward(float* out, const float* inp,
                    const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    for (int b = 0; b < B; b++) { // 句子数量
        for (int t = 0; t < T; t++) { // token 数量
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) { // 特征数量
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) { // 维度数
                    val += inp[bt * C + i] /* 向量值,向量矩阵 */  * weight[o * C + i] /* 权重值, 权重矩阵 */;
                }
                out[bt * OC + o] = val; // 行列的方式存储，每行是token ，每列是特征分
            }
        }
    }
}

/* residual_forward
 *
 * out[i] = inp1[i] + inp2[i]
 * in-place 安全（inp1 == out 也可以）
 */
void residual_forward(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}