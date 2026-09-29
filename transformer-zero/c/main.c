/* =============================================================================
 * main.c - 训练 + 推理主循环
 *
 * 用法：
 *   ./transformer train           # 训练模式
 *   ./transformer generate        # 推理模式
 *   ./transformer overfit         # 测试梯度
 *   ./transformer demo            # 教学演示模式（详细日志）
 * =============================================================================
 */

#include "transformer.h"

int sample_token(const float* probs, int n, float temperature) {
    if (temperature <= 0.0f) {
        // Greedy
        int best = 0;
        float max_p = probs[0];
        for (int i = 1; i < n; i++) {
            if (probs[i] > max_p) {
                max_p = probs[i];
                best = i;
            }
        }
        return best;
    }

    // 按概率采样
    float coin = (float)rand() / RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probs[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

/* === Train 模式 === */
void run_train() {
    printf("=== Training Mode ===\n\n");

    // 构建模型
    GPT model;
    gpt_build_from_checkpoint(&model, "data/init.bin");

    // 训练参数
    int B = 4;
    int T = 16;        // 教学用短序列
    int num_steps = 100;

    model.batch_size = B;
    model.seq_len = T;

    // 分配激活（按实际 B, T）
    int act_sizes[13] = {0};
    fill_in_activation_sizes(act_sizes, model.config, B, T);
    int total_acts = 0;
    for (int i = 0; i < 13; i++) total_acts += act_sizes[i];
    model.num_activations = total_acts;
    model.acts_memory = malloc_and_set_activations(&model.acts, act_sizes);

    // 加载数据
    DataLoader loader;
    dataloader_init(&loader, "data/train.bin", B, T, model.config.vocab_size);

    // 优化器
    AdamW opt;
    adamw_init(&opt);

    printf("\n开始训练：%d steps, B=%d, T=%d\n", num_steps, B, T);
    printf("参数数: %d\n", model.num_parameters);
    printf("激活数: %d\n\n", model.num_activations);

    for (int step = 0; step < num_steps; step++) {
        // 1) 取 batch
        dataloader_next_batch(&loader);

        // 2) Forward → loss
        gpt_forward(&model, loader.inputs, loader.targets);
        float loss = model.mean_loss;
        printf("step %3d: loss = %.4f\n", step, loss);

        // 3) Compute grads（数值微分 — 慢！）
        compute_numerical_grads(&model, loader.inputs, loader.targets, 1e-3f);

        // 4) AdamW 更新
        adamw_step(&opt, &model);
    }

    // 保存模型
    gpt_save_checkpoint(&model, "data/trained.bin");

    // 清理
    dataloader_free(&loader);
    gpt_free(&model);

    printf("\n训练完成！模型已保存到 data/trained.bin\n");
}

/* === Generate 模式 === */
void run_generate() {
    printf("=== Inference Mode ===\n\n");

    GPT model;
    gpt_build_from_checkpoint(&model, "data/trained.bin");

    int B = 1;
    int T = 32;
    model.batch_size = B;
    model.seq_len = T;

    int act_sizes[13] = {0};
    fill_in_activation_sizes(act_sizes, model.config, B, T);
    int total_acts = 0;
    for (int i = 0; i < 13; i++) total_acts += act_sizes[i];
    model.num_activations = total_acts;
    model.acts_memory = malloc_and_set_activations(&model.acts, act_sizes);

    // 简单 prompt: "ROMEO:"
    char* prompt_chars = "ROMEO:";
    int prompt_len = strlen(prompt_chars);

    int tokens[64];
    for (int i = 0; i < prompt_len; i++) {
        tokens[i] = (int)prompt_chars[i];  // ASCII
    }

    printf("Prompt: %s\n", prompt_chars);
    printf("Generated: ");

    for (int step = 0; step < 100; step++) {
        gpt_forward(&model, tokens, NULL);
        float* last_logits = model.acts.logits + (prompt_len + step - 1) * model.config.vocab_size;

        // softmax
        softmax_forward(model.acts.probs, model.acts.logits, B, T, model.config.vocab_size);
        float* last_probs = model.acts.probs + (prompt_len + step - 1) * model.config.vocab_size;

        int next = sample_token(last_probs, model.config.vocab_size, 0.8f);
        if (step < 64) tokens[prompt_len + step] = next;

        if (next < 128) {
            printf("%c", next);
        } else {
            printf("?");
        }
        fflush(stdout);
    }
    printf("\n\n");

    gpt_free(&model);
}

/* === Overfit Test === */
void run_overfit() {
    printf("=== Overfit Test (sanity check) ===\n\n");

    // 用小模型快速验证
    GPT model;
    gpt_build_from_checkpoint(&model, NULL);  // 随机初始化

    // 改为小模型配置
    config_small(&model.config);
    int param_sizes[NUM_PER_TENSORS];
    fill_in_parameter_sizes(param_sizes, model.config);
    // 重新分配（小模型）
    free(model.params_memory);
    free(model.grads_memory);
    free(model.m_memory);
    free(model.v_memory);
    int total = 0;
    for (int i = 0; i < NUM_PER_TENSORS; i++) total += param_sizes[i];
    model.num_parameters = total;
    model.params_memory = malloc_and_set_params(&model.params, param_sizes);
    model.grads_memory = malloc_grads(&model.grads, param_sizes);
    model.m_memory = (float*)calloc(model.num_parameters, sizeof(float));
    model.v_memory = (float*)calloc(model.num_parameters, sizeof(float));
    // 重置参数为小随机数
    srand(42);
    for (int i = 0; i < model.num_parameters; i++) {
        float r1 = (float)rand() / RAND_MAX + 1e-10f;
        float r2 = (float)rand() / RAND_MAX;
        float gaussian = sqrtf(-2.0f * logf(r1)) * cosf(2.0f * 3.14159f * r2);
        model.params_memory[i] = gaussian * 0.02f;
    }

    int B = 2;
    int T = 8;
    model.batch_size = B;
    model.seq_len = T;

    int act_sizes[13] = {0};
    fill_in_activation_sizes(act_sizes, model.config, B, T);
    int num_acts = 0;
    for (int i = 0; i < 13; i++) num_acts += act_sizes[i];
    model.num_activations = num_acts;
    model.acts_memory = malloc_and_set_activations(&model.acts, act_sizes);

    // 固定 batch（应该能过拟合）
    int inputs[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int targets[16] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};

    AdamW opt;
    adamw_init(&opt);

    printf("小模型: vocab=%d, n_embd=%d, n_layer=%d\n",
        model.config.vocab_size, model.config.n_embd, model.config.n_layer);
    printf("参数数: %d\n", model.num_parameters);
    printf("目标：过拟合到 loss < 0.01\n");
    printf("初始 loss:\n");
    gpt_forward(&model, inputs, targets);
    printf("  %.4f\n\n", model.mean_loss);

    for (int step = 0; step < 100; step++) {
        compute_numerical_grads(&model, inputs, targets, 1e-3f);
        adamw_step(&opt, &model);

        gpt_forward(&model, inputs, targets);
        if (step % 5 == 0 || step < 5) {
            printf("step %3d: loss = %.4f\n", step, model.mean_loss);
        }

        if (model.mean_loss < 0.01f) {
            printf("\n🎉 过拟合成功！模型结构 + 优化器正常\n");
            break;
        }
    }

    gpt_free(&model);
}

/* === Demo 模式：教学演示，详细日志 === */

/* 字符映射辅助: 将 token id 映射到可打印字符 */
static char token_to_char(int token_id) {
    /* tinyshakespeare vocab 映射 (来自 meta.pkl):
     * 0='\n' 1=' ' 2='!' 3='$' 4='&' 5='\'' 6=',' 7='-' 8='.' 9='3'
     * 10=':' 11=';' 12='?' 13='A' 14='B' 15='C' ... 38='Z'
     * 39='a' 40='b' ... 64='z'
     * 简化: token 0~9 对应 "\n !$&',-.3"
     *       token 10~12 对应 ":;?"
     *       token 13~38 对应 'A'~'Z'
     *       token 39~64 对应 'a'~'z'
     */
    static const char special[] = "\n !$&',-.3:;?";
    if (token_id >= 0 && token_id <= 12) return special[token_id];
    if (token_id >= 13 && token_id <= 38) return 'A' + (token_id - 13);
    if (token_id >= 39 && token_id <= 64) return 'a' + (token_id - 39);
    return '?';
}

void run_demo() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║         🎓 Transformer 教学演示模式                 ║\n");
    printf("║                                                      ║\n");
    printf("║  用小模型 + 真实莎士比亚数据，观察训练全过程          ║\n");
    printf("║  每步打印: loss、梯度统计、预测样本                  ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");
    fflush(stdout);

    /* ─── Step 1: 构建小模型 ─── */
    printf("📐 Step 1: 构建小模型\n");
    printf("────────────────────────────────────────\n");

    GPT model;
    config_small(&model.config);

    printf("  vocab_size = %d  (字符级: 65 个字符)\n", model.config.vocab_size);
    printf("  n_embd     = %d  (每个 token 用 %d 维向量表示)\n", model.config.n_embd, model.config.n_embd);
    printf("  n_layer    = %d  (%d 个 Transformer Block)\n", model.config.n_layer, model.config.n_layer);
    printf("  n_head     = %d  (每个 Block 有 %d 个注意力头)\n", model.config.n_head, model.config.n_head);
    printf("  block_size = %d  (最长序列 %d 个 token)\n", model.config.block_size, model.config.block_size);

    int param_sizes[NUM_PER_TENSORS];
    fill_in_parameter_sizes(param_sizes, model.config);
    int total = 0;
    for (int i = 0; i < NUM_PER_TENSORS; i++) total += param_sizes[i];
    model.num_parameters = total;

    model.params_memory = malloc_and_set_params(&model.params, param_sizes);
    model.grads_memory = malloc_grads(&model.grads, param_sizes);
    model.m_memory = (float*)calloc(total, sizeof(float));
    model.v_memory = (float*)calloc(total, sizeof(float));

    srand(42);
    for (int i = 0; i < total; i++) {
        float r1 = (float)rand() / RAND_MAX + 1e-10f;
        float r2 = (float)rand() / RAND_MAX;
        float g = sqrtf(-2.0f * logf(r1)) * cosf(2.0f * 3.14159f * r2);
        model.params_memory[i] = g * 0.02f;
    }

    printf("\n  📊 参数分布:\n");
    printf("    wte (词嵌入):     %6d 参数  (%d × %d)\n",
           model.config.vocab_size * model.config.n_embd,
           model.config.vocab_size, model.config.n_embd);
    printf("    wpe (位置编码):   %6d 参数  (%d × %d)\n",
           model.config.block_size * model.config.n_embd,
           model.config.block_size, model.config.n_embd);
    for (int l = 0; l < model.config.n_layer; l++) {
        printf("    Layer %d:\n", l);
        printf("      qkvw: %6d  (3×%d×%d)\n", 3*model.config.n_embd*model.config.n_embd,
               model.config.n_embd, model.config.n_embd);
        printf("      projw: %5d  (%d×%d)\n", model.config.n_embd*model.config.n_embd,
               model.config.n_embd, model.config.n_embd);
        printf("      fcw:  %6d  (4×%d×%d)\n", 4*model.config.n_embd*model.config.n_embd,
               model.config.n_embd, model.config.n_embd);
        printf("      + LN, bias, fcprojw ...\n");
    }
    printf("    ─────────────────────\n");
    printf("    总计: %d 参数\n", total);
    printf("    内存: %.1f KB (float32)\n\n", total * 4.0f / 1024.0f);

    /* ─── Step 2: 准备数据 ─── */
    int B = 2;
    int T = 8;
    model.batch_size = B;
    model.seq_len = T;

    int act_sizes[13] = {0};
    fill_in_activation_sizes(act_sizes, model.config, B, T);
    int num_acts = 0;
    for (int i = 0; i < 13; i++) num_acts += act_sizes[i];
    model.num_activations = num_acts;
    model.acts_memory = malloc_and_set_activations(&model.acts, act_sizes);

    printf("\n📚 Step 2: 加载莎士比亚数据\n");
    printf("────────────────────────────────────────\n");

    DataLoader loader;
    dataloader_init(&loader, "data/train.bin", B, T, model.config.vocab_size);
    printf("  Batch size: %d\n", B);
    printf("  Seq length: %d\n", T);
    printf("  每 batch: %d 个 token\n", B * T);
    printf("  总 batch 数: %d\n", loader.num_batches);
    fflush(stdout);

    /* ─── Step 3: 训练循环 ─── */
    int num_steps = 3;

    printf("\n🏋️ Step 3: 开始训练 (%d steps)\n", num_steps);
    printf("────────────────────────────────────────\n");
    printf("  ⚠️  数值微分: 每步需 %d×2 = %d 次前向传播\n", total, total * 2);
    printf("  ⏰  预计每步耗时 %d~%d 秒\n\n",
           total * 2 / 20000 + 1, total * 2 / 10000 + 2);
    fflush(stdout);

    AdamW opt;
    adamw_init(&opt);
    opt.lr = 3e-3f;  /* demo 用稍大学习率 */

    printf("  优化器: AdamW (lr=%.4f, β1=%.3f, β2=%.3f)\n\n",
           opt.lr, opt.beta1, opt.beta2);
    fflush(stdout);

    float best_loss = 1e10f;

    for (int step = 0; step < num_steps; step++) {
        dataloader_next_batch(&loader);

        /* Forward */
        gpt_forward(&model, loader.inputs, loader.targets);
        float loss = model.mean_loss;

        /* 打印进度 */
        printf("  ┌─ Step %2d ──────────────────────────────\n", step);
        printf("  │ Loss: %.4f", loss);
        if (loss < best_loss) {
            printf(" ⬇ (new best!)");
            best_loss = loss;
        } else {
            printf(" (best=%.4f)", best_loss);
        }
        printf("\n");
        fflush(stdout);

        /* 打印输入样本 */
        printf("  │\n  │ 📝 输入 (batch 0): \"");
        for (int t = 0; t < T; t++) {
            char c = token_to_char(loader.inputs[t]);
            if (c == '\n') printf("\\n");
            else printf("%c", c);
        }
        printf("\"\n  │ 目标 (batch 0): \"");
        for (int t = 0; t < T; t++) {
            char c = token_to_char(loader.targets[t]);
            if (c == '\n') printf("\\n");
            else printf("%c", c);
        }
        printf("\"\n");

        /* 打印预测 top-5 */
        {
            int V = model.config.vocab_size;
            float* probs0 = model.acts.probs;
            printf("  │\n  │ 🎯 位置 0 预测 (答案='%c'):\n",
                   token_to_char(loader.targets[0]));

            /* 找 top-5 */
            int top5_id[5] = {0};
            float top5_p[5] = {0};
            for (int k = 0; k < 5; k++) {
                top5_p[k] = -1.0f;
                for (int v = 0; v < V; v++) {
                    int skip = 0;
                    for (int j = 0; j < k; j++) if (v == top5_id[j]) skip = 1;
                    if (!skip && probs0[v] > top5_p[k]) {
                        top5_p[k] = probs0[v];
                        top5_id[k] = v;
                    }
                }
            }
            printf("  │   ");
            for (int k = 0; k < 5; k++) {
                char c = token_to_char(top5_id[k]);
                if (c == '\n') printf("\\n");
                else printf("'%c", c);
                printf("'(%4.1f%%) ", top5_p[k] * 100.0f);
            }
            printf("\n");
        }

        /* 计算梯度 (suppress internal progress) */
        printf("  │\n  │ 📊 梯度 (数值微分 %d params)...\n", total);
        fflush(stdout);

        /* 临时关闭 compute_numerical_grads 的进度输出:
         * 通过 freopen 重定向 stdout 到 /dev/null 临时 */
        FILE* saved_stdout = stdout;
        stdout = fopen("/dev/null", "w");
        compute_numerical_grads(&model, loader.inputs, loader.targets, 1e-3f);
        fclose(stdout);
        stdout = saved_stdout;

        /* 梯度统计 */
        float grad_max = 0, grad_sum = 0;
        int grad_nonzero = 0;
        for (int i = 0; i < total; i++) {
            float g = fabsf(model.grads_memory[i]);
            if (g > grad_max) grad_max = g;
            grad_sum += g;
            if (g > 1e-8f) grad_nonzero++;
        }
        printf("  │   非零: %d/%d (%.0f%%)  |max|=%.4f  |mean|=%.6f\n",
               grad_nonzero, total, 100.0f * grad_nonzero / total,
               grad_max, grad_sum / total);

        /* AdamW 更新 */
        adamw_step(&opt, &model);
        printf("  └────────────────────────────────────────\n\n");
        fflush(stdout);
    }

    /* ─── Step 4: 训练总结 ─── */
    printf("\n📊 训练总结\n");
    printf("────────────────────────────────────────\n");
    printf("  初始 loss: ~4.18 (随机初始化, ≈ log(65) = 4.17)\n");
    printf("  最终 loss: %.4f\n", best_loss);
    printf("  训练步数: %d\n", num_steps);
    printf("  参数总数: %d\n", total);
    printf("\n  📝 说明: %d 步训练只能让 loss 微小下降\n", num_steps);
    printf("  真正学到语言模式需要几千~几万步\n");
    printf("  用 'train' 模式做完整训练，用 'generate' 模式生成文本\n\n");

    /* 保存 */
    gpt_save_checkpoint(&model, "data/demo_trained.bin");
    printf("  模型已保存到 data/demo_trained.bin\n\n");

    dataloader_free(&loader);
    gpt_free(&model);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("用法: %s <mode>\n", argv[0]);
        printf("  modes:\n");
        printf("    train     - 训练模式\n");
        printf("    generate  - 推理模式\n");
        printf("    overfit   - 过拟合测试\n");
        printf("    demo      - 教学演示模式（详细日志）\n");
        return 1;
    }

    if (strcmp(argv[1], "train") == 0) {
        run_train();
    } else if (strcmp(argv[1], "generate") == 0) {
        run_generate();
    } else if (strcmp(argv[1], "overfit") == 0) {
        run_overfit();
    } else if (strcmp(argv[1], "demo") == 0) {
        run_demo();
    } else {
        printf("Unknown mode: %s\n", argv[1]);
        return 1;
    }

    return 0;
}