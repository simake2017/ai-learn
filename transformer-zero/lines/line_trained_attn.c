/* =============================================================================
 * line_trained_attn.c - 学习线：解剖 tiny_train 训练出来的模型
 *
 * 📚 这条线贯穿的竖线（与前两条线的交汇：机制 × 训练）：
 *
 *   tiny_train 线训好的 1032 个 float ──▶ 逐张量解剖
 *     acts.preatt / acts.att   注意力到底在看谁？
 *     params.wpe               位置编码学成了什么形状？
 *     params.qkvw              Q/K/V 三个矩阵谁长大了、谁没长？
 *
 * 🎭 案例：abab 模型（tiny_abab 夹具，和 tiny_train 线同一个随机起点）。
 *
 * 🔍 校准实验发现（ surprises，都是真跑出来的 ）：
 *   1) 注意力矩阵是【均匀的】！每个位置对各历史位置几乎等权（1/(t+1)），
 *      preatt 全场 |score| < 0.01 —— Q·K 点积被训成了约等于 0，softmax 输出均匀。
 *      模型没有学「盯着上一个词」，它选择了「把全部历史平均一下」。
 *   2) 奇偶规则的真相在 wpe：相邻位置的向量几乎反向
 *      （wpe[0]·wpe[1] / (|wpe0||wpe1|) ≈ -0.96），
 *      位置本身编码了奇偶 —— 既然看座次就知道该出 a 还是 b，
 *      「看上一个词」这个特技根本没必要练。
 *   3) |W_V|² 长得比 |W_QK|² 大 —— 历史平均值（V 的内容）被用上，
 *      而打分通道（Q/K）被弱化。
 *
 * 教训：别把所有功劳记到注意力头上。它是一个池化器 + 一个位置奇偶判读器，
 * 一起解出了这道题。
 * =============================================================================
 */

#include <stdio.h>
#include "../c/transformer.h"
#include "_shared/line.h"
#include "_shared/tiny_abab.h"

int line_trained_attn_run(LineCtx* ctx) {
    line_hop(ctx, "训练 abab 模型（和 tiny_train 线同一个起点、同一个流程）");
    TinyAbab t;
    tiny_abab_init(&t, TINY_ABAB_T);
    setenv("NUMGRAD_QUIET", "1", 1);
    const int STEPS = 400;
    float loss = 0.693f;
    for (int s = 0; s < STEPS; s++) {
        loss = tiny_abab_train_step(&t, tiny_abab_inputs(), tiny_abab_targets());
    }
    line_note(ctx, "复训 %d 步（数值梯度 + AdamW，和 tiny_train 完全同一套算子）", STEPS);
    line_note(ctx, "loss = %.4f —— 又一台「abab 世界模型」出厂，下面开始解剖。", loss);
    line_check_true(ctx, loss < 0.1f, "复训 loss < 0.1（模型确实学会了任务）");

    /* ──────────────────────────────────────────────
     * 解剖 1：注意力矩阵
     * ────────────────────────────────────────────── */
    line_hop(ctx, "解剖 ① 注意力矩阵 acts.att：每个位置在看谁？");
    line_note(ctx, "预期中的「标准答案」是 0/1 交替：每个位置盯着上一个词。");
    gpt_forward(&t.model, tiny_abab_inputs(), tiny_abab_targets());
    float* att = t.model.acts.att;      /* (1, NH=2, T=8, T=8) */
    float* pa = t.model.acts.preatt;
    for (int h = 0; h < 2; h++) {
        char title[64];
        snprintf(title, sizeof(title), "att 头 %d（训练后）", h);
        line_attn_mat(ctx, title, att + h * TINY_ABAB_T * TINY_ABAB_T,
                      TINY_ABAB_T, NULL, NULL);
    }
    /* 均匀性定量：att[h][t][j] 应 ≈ 1/(t+1)（j <= t） */
    {
        float maxdev = 0.0f;
        for (int h = 0; h < 2; h++)
            for (int i = 0; i < TINY_ABAB_T; i++)
                for (int j = 0; j <= i; j++) {
                    float expect = 1.0f / (i + 1);
                    float dev = fabsf(att[(h * TINY_ABAB_T + i) * TINY_ABAB_T + j] - expect);
                    if (dev > maxdev) maxdev = dev;
                }
        line_note(ctx, "uniform 偏差 max|att - 1/(t+1)| = %.4f", maxdev);
        line_check_true(ctx, maxdev < 0.02f,
                        "两个头的注意力都退化为「历史均匀池化」（与 1/(t+1) 偏差 < 0.02）");

        float pamax = 0.0f;
        for (int h = 0; h < 2; h++)
            for (int i = 0; i < TINY_ABAB_T; i++)
                for (int j = 0; j <= i; j++) {
                    float v = fabsf(pa[(h * TINY_ABAB_T + i) * TINY_ABAB_T + j]);
                    if (v > pamax) pamax = v;
                }
        line_note(ctx, "preatt 全场最大 |Q·K/√d| = %.4f —— 打分趋零，softmax 自然均匀。", pamax);
        line_check_true(ctx, pamax < 0.01f, "preatt 全场 |score| < 0.01（打分通道被训没了）");
    }

    /* ──────────────────────────────────────────────
     * 解剖 2：位置编码
     * ────────────────────────────────────────────── */
    line_hop(ctx, "解剖 ② 位置编码 params.wpe：奇偶规则的藏身处");
    float* wpe = t.model.params.wpe;    /* (16, 8)，打印相邻两行的余弦 */
    line_note(ctx, "相邻位置的 wpe 向量两两算余弦相似度：");
    float min_cos = 1.0f;
    for (int p = 0; p < 7; p++) {
        float dot = 0, n0 = 0, n1 = 0;
        for (int i = 0; i < 8; i++) {
            dot += wpe[p * 8 + i] * wpe[(p + 1) * 8 + i];
            n0 += wpe[p * 8 + i] * wpe[p * 8 + i];
            n1 += wpe[(p + 1) * 8 + i] * wpe[(p + 1) * 8 + i];
        }
        float cos = dot / (sqrtf(n0) * sqrtf(n1));
        if (!ctx->quiet) printf("  cos(wpe[%d], wpe[%d]) = %+.3f\n", p, p + 1, cos);
        if (cos < min_cos) min_cos = cos;
    }
    line_note(ctx, "相邻位置几乎反向（余弦 ≈ -1）—— 偶数位和奇数位各聚一端，");
    line_note(ctx, "  座次本身就在喊「我是奇数位，该出 a 还是 b 看我就行」。");
    line_check_true(ctx, min_cos < -0.8f,
                    "wpe 相邻位置余弦 < -0.8（近反向，位置编码写入了奇偶交替结构）");

    /* ──────────────────────────────────────────────
     * 解剖 3：QKV 权重谁长大了
     * ────────────────────────────────────────────── */
    line_hop(ctx, "解剖 ③ QKV 权重矩阵：三个角色谁被训练改变了？");
    float qk2 = 0, v2 = 0;
    for (int i = 0; i < 16 * 8; i++) qk2 += t.model.params.qkvw[i] * t.model.params.qkvw[i];
    for (int i = 16 * 8; i < 24 * 8; i++) v2 += t.model.params.qkvw[i] * t.model.params.qkvw[i];
    line_note(ctx, "|W_QK|² = %.4f   |W_V|² = %.4f", qk2, v2);
    line_note(ctx, "V（搬运内容）长大了，Q/K（打分通道）几乎停在初始化水平 ——");
    line_note(ctx, "  与 preatt≈0、att 均匀互相印证：模型弱化了「挑着看」，");
    line_note(ctx, "  保留了「平均拿」，再配合 wpe 的奇偶座次，正是解 abab 的最短路。");
    line_check_true(ctx, v2 > qk2, "|W_V|² > |W_QK|²（内容搬运通道比打分通道长得更大）");

    /* ──────────────────────────────────────────────
     * 收尾：机制 × 训练 的交汇结论
     * ────────────────────────────────────────────── */
    line_hop(ctx, "结论：同一个 forward，两种读法");
    line_note(ctx, "机制线（ln_attn/multi_head）告诉你注意力【能】做什么：按相关性搬运语义。");
    line_note(ctx, "训练线（tiny_train）告诉你梯度【实际】把它用成了什么：历史平均池化。");
    line_note(ctx, "对 abab 这道题，最短路根本不用「看上一个词」——");
    line_note(ctx, "  但「看上一个词」的解法在大模型里是真实存在的（指代、句法都靠它），");
    line_note(ctx, "  tiny_train 线里那 0.9 的置信度就是这三件零件凑出来的最优分工。");

    tiny_abab_free(&t);
    return line_summary(ctx);
}
