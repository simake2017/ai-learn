/* =============================================================================
 * activations.c - GELU, Softmax, CrossEntropy
 *
 * GELU(x) = x * Φ(x) ≈ 0.5x(1 + tanh(√(2/π)·(x + 0.044715x³)))
 *
 * Softmax: 数值稳定版（减最大值）
 *
 * CrossEntropy: -log(probs[target])
 * =============================================================================
 */

#include "transformer.h"

/* GELU：近似版（tanh 形式） */
static inline float gelu(float x) {
    float c = 0.7978845608f;  // sqrt(2/π)
    float b = 0.044715f;
    float inner = c * (x + b * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

void gelu_forward(float* out, const float* inp, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = gelu(inp[i]);
    }
}

/* Softmax：逐行（每个 (b,t) 位置）
 *
 * 数值稳定版：先减最大值
 *   exp(x_i - max) / Σ exp(x_j - max)
 */
void softmax_forward(float* probs, const float* logits, int B, int T, int V) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* logits_bt = logits + b * T * V + t * V;
            float* probs_bt = probs + b * T * V + t * V;

            // Step 1: 找最大值
            float maxval = -1e30f;
            for (int v = 0; v < V; v++) {
                if (logits_bt[v] > maxval) maxval = logits_bt[v];
            }

            // Step 2: exp(x - max)
            float sum = 0.0f;
            for (int v = 0; v < V; v++) {
                float e = expf(logits_bt[v] - maxval);
                probs_bt[v] = e;
                sum += e;
            }

            // Step 3: 归一化
            float inv_sum = 1.0f / sum;
            for (int v = 0; v < V; v++) {
                probs_bt[v] *= inv_sum;
            }
        }
    }
}

/* CrossEntropy: losses[b, t] = -log(probs[b, t, targets[b, t]]) */
void crossentropy_forward(float* losses, const float* probs,
                          const int* targets, int B, int T, int V) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* probs_bt = probs + b * T * V + t * V;
            int target = targets[b * T + t];
            float p = probs_bt[target];
            // 防止 log(0) = -inf，加一个很小的值
            if (p < 1e-30f) p = 1e-30f;
            losses[b * T + t] = -logf(p);
        }
    }
}