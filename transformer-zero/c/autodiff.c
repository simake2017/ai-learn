#include "transformer.h"

void compute_numerical_grads(GPT* model, const int* inputs, const int* targets, float eps) {
    int n = model->num_parameters;
    if (n <= 0) {
        fprintf(stderr, "Model not initialized!\n");
        return;
    }

    printf("Computing numerical grads: %d params, eps=%g\n", n, eps);

    float saved_loss = model->mean_loss;

    for (int i = 0; i < n; i++) {
        float saved = model->params_memory[i];

        // L(θ + ε)
        model->params_memory[i] = saved + eps;
        gpt_forward(model, inputs, targets);
        float L_plus = model->mean_loss;

        // L(θ - ε)
        model->params_memory[i] = saved - eps;
        gpt_forward(model, inputs, targets);
        float L_minus = model->mean_loss;

        // 中心差分
        model->grads_memory[i] = (L_plus - L_minus) / (2.0f * eps);

        // 还原
        model->params_memory[i] = saved;

        if (i % 5000 == 0 || i == n - 1) {
            printf("  grad %d / %d (%.1f%%), sample = %g\n",
                   i, n, 100.0f * i / n, model->grads_memory[i]);
        }
    }

    model->mean_loss = saved_loss;
}
