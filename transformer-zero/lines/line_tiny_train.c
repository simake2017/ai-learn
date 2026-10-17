/* =============================================================================
 * line_tiny_train.c - 学习线：从零训练一个能背出 abab 的最小 GPT
 *
 * 📚 这条线贯穿的竖线（训练闭环，对应 c/main.c run_train 的真实流程）：
 *
 *   数据(abab) ──▶ forward ──▶ crossentropy loss ──▶ 数值梯度 ──▶ AdamW ──▶ 更新参数 ──┐
 *       ▲                                                                            │
 *       └──────────────────────────────────── 循环 N 步 ◀────────────────────────────┘
 *
 * 🎭 任务：abab 续写（夹具见 _shared/tiny_abab.h，trained_attn/generate 线共用）
 *   vocab 只有 2 个字符 {a=0, b=1}，序列就是 a b a b a b ...
 *   规则一句话：看到 a 出 b，看到 b 出 a。
 *   人手算得出的正确答案 = 能全程监督模型的每一个预测。
 *
 * ⚙️ 规模：vocab=2, C=8, 1 层, 2 头，1032 参数。
 *    梯度用中心差分数值微分（2 次 forward / 参数），AdamW 更新 —— 全部是真算子。
 *
 * 🔍 预期结果：
 *   初始 loss ≈ log(2) ≈ 0.693（随机模型对二选一完全没头绪）
 *   训练 500 步后 loss → 0.03，每个位置的 top-1 预测 = 正确答案
 * =============================================================================
 */

#include <stdio.h>
#include "../c/transformer.h"
#include "_shared/line.h"
#include "_shared/tiny_abab.h"

int line_tiny_train_run(LineCtx* ctx) {
    /* ═══ 跳点 ① 任务与数据 ═══ */
    line_hop(ctx, "任务：abab 续写（vocab=2，规则：a→b, b→a）");
    TinyAbab t;
    tiny_abab_init(&t, TINY_ABAB_T);
    {
        char seq[64] = {0};
        for (int i = 0; i < TINY_ABAB_T; i++) strcat(seq, tiny_abab_tok(tiny_abab_inputs()[i]));
        line_note(ctx, "模型参数: %d（wte/wpe/ln/qkv/proj/mlp/lnf 全部算上）", t.model.num_parameters);
        line_note(ctx, "输入序列 : %s  (%d 个 token)", seq, TINY_ABAB_T);
        seq[0] = 0;
        for (int i = 0; i < TINY_ABAB_T; i++) strcat(seq, tiny_abab_tok(tiny_abab_targets()[i]));
        line_note(ctx, "正确答案 : %s  （每个位置的下一个字符）", seq);
    }
    line_note(ctx, "模型唯一需要学会的事：知道「自己在第几位」，就能 100%% 答对。");
    line_note(ctx, "（到底靠注意力还是靠位置编码？见 trained_attn 线的解剖。）");

    /* ═══ 跳点 ② 随机初始化：先考它一次 ═══ */
    line_hop(ctx, "随机初始化的模型：先考它一次");
    const float LOG2_2 = 0.6931472f;  // log(2)：二选一完全瞎猜的 loss
    gpt_forward(&t.model, tiny_abab_inputs(), tiny_abab_targets());
    float loss0 = t.model.mean_loss;
    line_note(ctx, "forward 一遍（embedding → 1 个 block → final LN → logits → softmax → loss）");
    line_note(ctx, "随机模型给出的 loss = %.4f，而瞎猜二选一的理论值是 log(2) = %.4f", loss0, LOG2_2);
    line_check_close(ctx, loss0, LOG2_2, 0.05f, "初始 loss ≈ log(2) ≈ 0.693（随机模型对 a/b 完全没头绪）");

    line_note(ctx, "第一个位置（看到 a）的预测分布: P(a)=%.3f, P(b)=%.3f —— 接近抛硬币",
              t.model.acts.probs[0], t.model.acts.probs[1]);
    line_check_true(ctx, t.model.acts.probs[0] > 0.3f && t.model.acts.probs[0] < 0.7f,
                    "随机模型的首位置预测接近 50/50");

    /* ═══ 跳点 ③ 训练循环：forward → 数值梯度 → AdamW ═══ */
    const int NUM_STEPS = 500;
    line_hop(ctx, "训练 %d 步：每步 = forward(loss) + 数值梯度(中心差分) + AdamW 更新", NUM_STEPS);
    line_note(ctx, "数值梯度：每个参数分别 +eps/-eps 各跑一次 forward，用 loss 差估斜率。");
    line_note(ctx, "%d 参数 × 2 次 forward = 每步 %d 次前向 —— 教学规模，几十秒跑完。",
              t.model.num_parameters, t.model.num_parameters * 2);

    /* 数值梯度的进度条默认很吵，训练期间静默，只留 loss 曲线 */
    setenv("NUMGRAD_QUIET", "1", 1);

    float loss_final = loss0;
    for (int step = 0; step < NUM_STEPS; step++) {
        loss_final = tiny_abab_train_step(&t, tiny_abab_inputs(), tiny_abab_targets());
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
    gpt_forward(&t.model, tiny_abab_inputs(), tiny_abab_targets());   // 用最终参数再跑一遍
    {
        int correct = 0;
        for (int i = 0; i < TINY_ABAB_T; i++) {
            const float* p = t.model.acts.probs + i * 2;   // 位置 i 的 (P(a), P(b))
            int pred = (p[1] > p[0]) ? 1 : 0;
            int ok = (pred == tiny_abab_targets()[i]);
            correct += ok;
            if (!ctx->quiet) {
                printf("  位置 %d: 上文=%s → 预测 %s (P=%.4f) | 答案 %s %s\n",
                       i, tiny_abab_tok(tiny_abab_inputs()[i]),
                       tiny_abab_tok(pred), p[pred], tiny_abab_tok(tiny_abab_targets()[i]),
                       ok ? LN_GREEN "✓" LN_RESET : LN_RED "✗" LN_RESET);
            }
        }
        line_check_true(ctx, correct == TINY_ABAB_T, "全部 8 个位置的 top-1 预测都正确");
        line_check_close(ctx, t.model.acts.probs[1 * 2 + 0], 1.0f, 0.05f,
                         "位置1（看到 b 应出 a）：P(a) ≈ 1.0");
        line_check_close(ctx, t.model.acts.probs[0 * 2 + 1], 1.0f, 0.05f,
                         "位置0（看到 a 应出 b）：P(b) ≈ 1.0");
    }
    line_note(ctx, "");
    line_note(ctx, "这 %d 个 float 里现在装着一个完整的「abab 世界模型」。", t.model.num_parameters);
    line_note(ctx, "它把规则压进了哪里的权重？attention 真的学会「看上一个词」了吗？");
    line_note(ctx, "→ 继续跑 trained_attn 线，当场解剖这个模型。");

    tiny_abab_free(&t);
    return line_summary(ctx);
}
