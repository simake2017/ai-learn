/* =============================================================================
 * line_generate.c - 学习线：自回归生成（训练 → 推理 的完整闭环）
 *
 * 📚 这条线贯穿的竖线（gpt_forward 的「推理形态」，c/main.c run_generate）：
 *
 *   seed ──▶ forward ──▶ 取最后一个位置的 softmax ──▶ 采样/argmax 出下一个 token
 *     ▲                                                        │
 *     └──────────────── 把它拼回序列，再来一遍 ◀────────────────┘
 *
 * 🎭 案例：abab 模型（tiny_abab 夹具）。从种子 a 出发，让它自己接龙。
 *   这条线是 tiny_train 线的「下集」：训练时每个位置的答案是人给的，
 *   生成时答案要靠模型自己一个一个字吐出来 —— 这就是「自回归」。
 *
 * 🔍 关键机制（和训练时的三个不同）：
 *   1) targets = NULL：只算 probs，不算 loss（没人批卷了）
 *   2) 每步只取【最后一个位置】的分布：前面的位置算过了也不会再用
 *   3) 序列越来越长：T 从 1 涨到 N，激活缓存在 tiny_abab_init 时已按 16 分配好
 *
 * 采样策略：本线用 argmax（贪心）保证输出确定、可校验。
 *   项目里的 sample_token(probs, n, temperature) 是带温度的随机采样，
 *   temperature→0 时退化为 argmax，temperature 越大越发散。
 * =============================================================================
 */

#include <stdio.h>
#include "../c/transformer.h"
#include "_shared/line.h"
#include "_shared/tiny_abab.h"

int line_generate_run(LineCtx* ctx) {
    /* ═══ 跳点 ① 复用训练：同一个模型，切换成推理形态 ═══ */
    line_hop(ctx, "先训练 abab 模型（tiny_train 线同款），再让它自己写作");
    TinyAbab t;
    tiny_abab_init(&t, 16);   /* 激活按 T=16 分配：生成时序列会逐步变长 */
    setenv("NUMGRAD_QUIET", "1", 1);
    const int STEPS = 500;
    float loss = 0.693f;
    for (int s = 0; s < STEPS; s++) {
        loss = tiny_abab_train_step(&t, tiny_abab_inputs(), tiny_abab_targets());
    }
    line_note(ctx, "训练 %d 步，loss = %.4f —— 考卷时代结束，现在进入创作时代。", STEPS, loss);
    line_check_true(ctx, loss < 0.1f, "训练 loss < 0.1（生成质量的前提）");

    /* ═══ 跳点 ② 自回归主循环 ═══ */
    line_hop(ctx, "自回归生成：seed=a，每次把新 token 拼回去再预测");
    line_note(ctx, "循环三件套：gpt_forward(seq, NULL) → 末位置 argmax → seq[len++] = next");
    int seq[16];
    int len = 0;
    seq[len++] = 0;   /* 种子：a */
    line_note(ctx, "  第 1 步: T=1，模型只看到 [a]，预测下一个");
    while (len < 13) {
        t.model.seq_len = len;                      /* 只前向当前长度 */
        gpt_forward(&t.model, seq, NULL);           /* targets=NULL：只出 probs */
        const float* p = t.model.acts.probs + (len - 1) * 2;  /* 末位置分布 */
        int next = (p[1] > p[0]) ? 1 : 0;           /* argmax 采样 */
        if (!ctx->quiet && len <= 4) {
            printf("  T=%2d 上文=", len);
            for (int i = 0; i < len; i++) printf("%s", tiny_abab_tok(seq[i]));
            printf(" → P(a)=%.3f P(b)=%.3f → 选出 %s\n", p[0], p[1], tiny_abab_tok(next));
        }
        seq[len++] = next;
    }
    line_note(ctx, "  ...后续各步同理，序列一路长到 T=13");
    {
        char gen[32] = {0};
        for (int i = 0; i < len; i++) strcat(gen, tiny_abab_tok(seq[i]));
        line_note(ctx, "生成结果: %s", gen);
    }
    line_check_true(ctx, seq[1] == 1, "seed a 之后第 1 个生成 token = b");
    {
        int ok = 1;
        for (int i = 1; i < len; i++) if (seq[i] != (i % 2)) { ok = 0; break; }
        line_check_true(ctx, ok, "从 a 出发生成的 13 个 token 严格交替 ababab...");
    }

    /* ═══ 跳点 ③ 换种子再考一次 ═══ */
    line_hop(ctx, "换种子：从 b 出发，考的是同一套权重");
    int seq2[16];
    int len2 = 0;
    seq2[len2++] = 1;   /* 种子：b */
    while (len2 < 13) {
        t.model.seq_len = len2;
        gpt_forward(&t.model, seq2, NULL);
        const float* p = t.model.acts.probs + (len2 - 1) * 2;
        seq2[len2++] = (p[1] > p[0]) ? 1 : 0;
    }
    {
        char gen[32] = {0};
        for (int i = 0; i < len2; i++) strcat(gen, tiny_abab_tok(seq2[i]));
        line_note(ctx, "生成结果: %s", gen);
    }
    {
        int ok = 1;
        for (int i = 1; i < len2; i++) if (seq2[i] != ((i + 1) % 2)) { ok = 0; break; }
        line_check_true(ctx, ok, "从 b 出发生成 bababa...（规则 b→a 同样生效）");
    }
    line_note(ctx, "");
    line_note(ctx, "同一组权重，两种种子，两种展开 —— 模型学到的是「规则」不是「答案」。");

    /* ═══ 跳点 ④ 收束：训练闭环全貌 ═══ */
    line_hop(ctx, "收束：至此最小模型的全生命周期已经闭环");
    line_note(ctx, "  embed      : id ──▶ 向量（身份 + 座次）");
    line_note(ctx, "  ln_attn    : 向量 ──▶ 互相搬运语义（本例退化为历史平均池化）");
    line_note(ctx, "  multi_head : 同一句话多问几个问题");
    line_note(ctx, "  causal_mask: 每个位置只看过去，防泄题");
    line_note(ctx, "  tiny_train : 数值梯度 + AdamW 把规则压进 1032 个 float");
    line_note(ctx, "  trained_attn: 解剖发现：池化 + wpe 奇偶座次就是最优分工");
    line_note(ctx, "  generate   : 自回归展开 —— 你刚看完的这一跳");
    line_note(ctx, "");
    line_note(ctx, "把 vocab=2 换成 65、层数换成 6、数据换成莎士比亚，就是 c/main.c 的主程序。");

    tiny_abab_free(&t);
    return line_summary(ctx);
}
