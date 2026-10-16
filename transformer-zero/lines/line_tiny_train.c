/* =============================================================================
 * line_tiny_train.c - 学习线：从零训练一个能背出 abab 的最小 GPT
 *
 * 📚 这条线贯穿的竖线（训练闭环，对应 c/main.c run_train 的真实流程）：
 *
 *   数据(abab) ──▶ forward ──▶ crossentropy loss ──▶ 数值梯度 ──▶ AdamW ──▶ 更新参数 ──┐
 *       ▲                                                                            │
 *       └──────────────────────────────────── 循环 N 步 ◀────────────────────────────┘
 *
 * 🎭 任务：abab 续写
 *   vocab 只有 2 个字符 {a=0, b=1}，序列就是 a b a b a b ...
 *   规则一句话：看到 a 出 b，看到 b 出 a。
 *   人手算得出的正确答案 = 能全程监督模型的每一个预测。
 *
 * 🧮 为什么这个任务能训练出来？
 *   正确答案强制要求「当前 token 去关注上一个 token」—— 正好是注意力机制的看家本领。
 *   训练好后（trained_attn 线）你会看到注意力矩阵自己长出 0/1 交替结构。
 *
 * ⚙️ 规模：vocab=2, C=8, 1 层, 2 头，约 1 千参数。
 *    梯度用中心差分数值微分（2 次 forward / 参数），AdamW 更新 —— 全部是真算子。
 *
 * 🔍 预期结果：
 *   初始 loss ≈ log(2) ≈ 0.693（随机模型对二选一完全没头绪）
 *   训练后 loss → 0.00x，每个位置的 top-1 预测 = 正确答案
 * =============================================================================
 */

#include <stdio.h>
#include "../c/transformer.h"
#include "_shared/line.h"

int line_tiny_train_run(LineCtx* ctx) {
    /* ──────────────────────────────────────────────
     * 最小模型配置（参照 test_10_forward 的手工搭法）
     * ────────────────────────────────────────────── */
    GPT model;
    model.config.vocab_size = 2;    // 只有 a, b
    model.config.block_size = 16;
    model.config.n_layer = 1;
    model.config.n_head = 2;
    model.config.n_embd = 8;

    int B = 1, T = 8;               // 一句话 8 个 token
    model.batch_size = B;
    model.seq_len = T;

    int param_sizes[NUM_PER_TENSORS];
    fill_in_parameter_sizes(param_sizes, model.config);
    int total = 0;
    for (int i = 0; i < NUM_PER_TENSORS; i++) total += param_sizes[i];
    model.num_parameters = total;

    model.params_memory = malloc_and_set_params(&model.params, param_sizes);
    model.grads_memory = malloc_grads(&model.grads, param_sizes);
    model.m_memory = (float*)calloc(total, sizeof(float));
    model.v_memory = (float*)calloc(total, sizeof(float));

    int act_sizes[13] = {0};   /* 13 类激活张量（定义在 model.c，外部按 test_10 的惯例用字面量） */
    fill_in_activation_sizes(act_sizes, model.config, B, T);
    int total_acts = 0;
    for (int i = 0; i < 13; i++) total_acts += act_sizes[i];
    model.num_activations = total_acts;
    model.acts_memory = malloc_and_set_activations(&model.acts, act_sizes);

    /* 随机初始化（和 test_10 同款种子） */
    srand(42);
    for (int i = 0; i < total; i++) {
        float r1 = (float)rand() / RAND_MAX + 1e-10f;
        float r2 = (float)rand() / RAND_MAX;
        model.params_memory[i] = (r1 + r2 - 1.5f) * 0.02f;  // 小随机数， centered
    }

    /* ──────────────────────────────────────────────
     * 训练数据：a b a b a b a b，目标 = 每个位置的下一个字符
     * inputs[i] 的答案是 targets[i]（已错位：input a → target b）
     * ────────────────────────────────────────────── */
    int inputs[8]  = {0, 1, 0, 1, 0, 1, 0, 1};   // a b a b a b a b
    int targets[8] = {1, 0, 1, 0, 1, 0, 1, 0};   // b a b a b a b a
    const char* tok[2] = {"a", "b"};
    const int NUM_STEPS = 500;
    const float LOG2_2 = 0.6931472f;  // log(2)：二选一完全瞎猜的 loss 下限

    /* ═══ 跳点 ① 任务与数据 ═══ */
    line_hop(ctx, "任务：abab 续写（vocab=2，规则：a→b, b→a）");
    line_note(ctx, "模型参数: %d（wte/wpe/ln/qkv/proj/mlp/lnf 全部算上）", total);
    {
        char seq[64] = {0};
        for (int i = 0; i < T; i++) strcat(seq, tok[inputs[i]]);
        line_note(ctx, "输入序列 : %s  (%d 个 token)", seq, T);
        seq[0] = 0;
        for (int i = 0; i < T; i++) strcat(seq, tok[targets[i]]);
        line_note(ctx, "正确答案 : %s  （每个位置的下一个字符）", seq);
    }
    line_note(ctx, "模型唯一需要学会的事：注意力里「看我左边那个词」，就能 100%% 答对。");

    /* ═══ 跳点 ② 随机初始化：先答一题看看 ═══ */
    line_hop(ctx, "随机初始化的模型：先考它一次");
    gpt_forward(&model, inputs, targets);
    float loss0 = model.mean_loss;
    line_note(ctx, "forward 一遍（embedding → 1 个 block → final LN → logits → softmax → loss）");
    line_note(ctx, "随机模型给出的 loss = %.4f，而瞎猜二选一的理论值是 log(2) = %.4f",
              loss0, LOG2_2);
    line_check_close(ctx, loss0, LOG2_2, 0.05f, "初始 loss ≈ log(2) ≈ 0.693（随机模型对 a/b 完全没头绪）");

    /* 看第一个位置的预测分布：应该接近 50/50 */
    line_note(ctx, "第一个位置（看到 a）的预测分布: P(a)=%.3f, P(b)=%.3f —— 接近抛硬币",
              model.acts.probs[0], model.acts.probs[1]);
    line_check_true(ctx, model.acts.probs[0] > 0.3f && model.acts.probs[0] < 0.7f,
                    "随机模型的首位置预测接近 50/50");

    /* ═══ 跳点 ③ 训练循环：forward → 数值梯度 → AdamW ═══ */
    line_hop(ctx, "训练 %d 步：每步 = forward(loss) + 数值梯度(中心差分) + AdamW 更新",
             NUM_STEPS);
    line_note(ctx, "数值梯度：每个参数分别 +eps/-eps 各跑一次 forward，用 loss 差估斜率。");
    line_note(ctx, "本模型 %d 参数 × 2 次 forward = 每步 %d 次前向 —— 教学规模，秒级跑完。",
              total, total * 2);
    line_note(ctx, "");

    /* 数值梯度的进度条默认很吵，训练期间静默，只留 loss 曲线 */
    setenv("NUMGRAD_QUIET", "1", 1);

    AdamW opt;
    adamw_init(&opt);
    float loss_final = loss0;
    for (int step = 0; step < NUM_STEPS; step++) {
        gpt_forward(&model, inputs, targets);
        loss_final = model.mean_loss;
        compute_numerical_grads(&model, inputs, targets, 1e-3f);
        adamw_step(&opt, &model);
        if (step % 100 == 0 || step == NUM_STEPS - 1) {
            line_note(ctx, "  step %3d: loss = %.4f", step, loss_final);
        }
    }
    line_note(ctx, "");
    line_check_true(ctx, loss_final < 0.05f,
                    "训练后 loss < 0.05（从 0.693 降下来，降幅一个数量级以上）");
    line_check_true(ctx, loss_final < loss0 / 10.0f, "训练后 loss 比初始小 10 倍以上");

    /* ═══ 跳点 ④ 考卷复查：逐位置核对预测 ═══ */
    line_hop(ctx, "复查：训练后的模型逐位置预测 vs 正确答案");
    gpt_forward(&model, inputs, targets);   // 用最终参数再跑一遍
    {
        int correct = 0;
        for (int t = 0; t < T; t++) {
            const float* p = model.acts.probs + t * 2;   // 位置 t 的 (P(a), P(b))
            int pred = (p[1] > p[0]) ? 1 : 0;
            int ok = (pred == targets[t]);
            correct += ok;
            if (!ctx->quiet) {
                printf("  位置 %d: 上文=%s → 预测 %s (P=%.4f) | 答案 %s %s\n",
                       t, tok[inputs[t]], tok[pred], p[pred], tok[targets[t]],
                       ok ? LN_GREEN "✓" LN_RESET : LN_RED "✗" LN_RESET);
            }
        }
        line_check_true(ctx, correct == T, "全部 8 个位置的 top-1 预测都正确");
        line_check_close(ctx, model.acts.probs[1 * 2 + 0], 1.0f, 0.05f,
                         "位置1（看到 b 应出 a）：P(a) ≈ 1.0");
        line_check_close(ctx, model.acts.probs[0 * 2 + 1], 1.0f, 0.05f,
                         "位置0（看到 a 应出 b）：P(b) ≈ 1.0");
    }
    line_note(ctx, "");
    line_note(ctx, "这 %d 个 float 里现在装着一个完整的「abab 世界模型」——", total);
    line_note(ctx, "它没有背答案，而是把规则压进了 W_QKV/W_proj 的权重里（见线 trained_attn 的解剖）。");

    gpt_free(&model);
    return line_summary(ctx);
}
