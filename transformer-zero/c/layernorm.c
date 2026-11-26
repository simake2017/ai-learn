/* =============================================================================
 * layernorm.c - Layer Normalization
 *
 * 作用：对每个 token 的特征向量做 z-score 标准化，把每个 (b, t) 位置的
 * C 个特征缩放到"均值 0、标准差 1"的尺度上，再通过可学习的 γ、β 恢复尺度。
 *
 * 对每个 (b, t) 位置，沿最后一维（n_embd）归一化：
 *   mean = (1/C) Σ x_i
 *   var  = (1/C) Σ (x_i - mean)²
 *   rstd = 1 / sqrt(var + eps)
 *   y_i  = (x_i - mean) * rstd * gamma_i + beta_i
 *
 * 注意：
 *   - 归一化的"组"是单个 token 内部的 C 个特征，与 batch、其他 token 无关；
 *   - 标准化只保证均值/方差，不保证变成正态分布；
 *   - z 化后的值表示"偏离均值几个标准差"，绝大多数落在 [-3, 3]，但无硬边界；
 *   - eps 防止方差为 0 时除零。
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

            // 计算均值（对这个 token 的 C 个特征求平均）
            float mean = 0.0f;
            for (int i = 0; i < C; i++) {
                mean += x[i];
            }
            mean /= C;

            // 计算方差（衡量这 C 个特征围绕均值的分散程度）
            float var = 0.0f;
            for (int i = 0; i < C; i++) {
                float d = x[i] - mean;
                var += d * d;
            }
            var /= C;

            // rstd = 1/标准差，后面乘它相当于除以标准差；eps 防除零
            float rstd = 1.0f / sqrtf(var + eps);

            // 归一化 + 缩放 + 平移
            for (int i = 0; i < C; i++) {
                // z 化：n 表示 x[i] 偏离均值几个标准差（绝大多数落在 [-3, 3]，无硬边界）
                float n = (x[i] - mean) * rstd;
                // γ、β 是可学习参数：让模型自己决定每个维度恢复多少原始尺度
                o[i] = n * weight[i] + bias[i]; // mrd 这里是计算出新的权重矩阵值
            }
        }
    }
}