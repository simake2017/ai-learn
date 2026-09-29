/* =============================================================================
 * layernorm.c - Layer Normalization
 *
 * 对每个 (b, t) 位置，沿最后一维（n_embd）归一化：
 *   mean = (1/C) Σ x_i
 *   var  = (1/C) Σ (x_i - mean)²
 *   rstd = 1 / sqrt(var + eps)
 *   y_i  = (x_i - mean) * rstd * gamma_i + beta_i
 * =============================================================================
 */

#include "transformer.h"

void layernorm_forward(float* out, const float* inp,
                       const float* weight, const float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* x = inp + b * T * C + t * C;
            float* o = out + b * T * C + t * C;

            // 计算均值
            float mean = 0.0f;
            for (int i = 0; i < C; i++) {
                mean += x[i];
            }
            mean /= C;

            // 计算方差
            float var = 0.0f;
            for (int i = 0; i < C; i++) {
                float d = x[i] - mean;
                var += d * d;
            }
            var /= C;

            // rstd
            float rstd = 1.0f / sqrtf(var + eps);

            // 归一化 + 缩放 + 平移
            for (int i = 0; i < C; i++) {
                float n = (x[i] - mean) * rstd;
                o[i] = n * weight[i] + bias[i];
            }
        }
    }
}