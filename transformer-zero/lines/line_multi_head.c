/* =============================================================================
 * line_multi_head.c - 学习线：多头注意力（一个句子，多种看法）
 *
 * 📚 这条线贯穿的竖线（attention_forward 内部，attention.c）：
 *
 *   qkv (T, 3C) ──切 C 成 NH 段──▶ 头0: Q0·K0/√d → softmax → ×V0 ─┐
 *                                   头1: Q1·K1/√d → softmax → ×V1 ─┤ concat
 *                                                                  ▼
 *                                                    att_out (T, C) ──×W_proj──▶ 混合
 *
 * 🎭 案例：「小猫 追 它」，C=8 拆成 2 个头（每个头在自己的 4 维子空间里打分）
 *   - 头 0「自扫门前雪」：Q/K 取角色维 → 每个词的注意力指向自己
 *   - 头 1「朝拜句首」：Q 取 +2t、K 取 -2t（t = 位置）→ 分数 = -2t·s，
 *     离句首越远衰减越狠 → 所有词一致看向「小猫」
 *   - 两个头各看各的、互不知道对方存在；最后的 W_proj 负责把两份「看法」重新混合
 *
 * 🔍 手工构造可精确手算：头0 的 softmax 由角色点积驱动，头1 由位置点积驱动。
 * =============================================================================
 */

#include <stdio.h>
#include "../c/transformer.h"
#include "_shared/line.h"

int line_multi_head_run(LineCtx* ctx) {
    int B = 1, T = 3, C = 8, NH = 2;
    int hs = C / NH;              /* 每个头的子空间维度 = 4 */
    float scale = 1.0f / sqrtf((float)hs);
    const char* word[3] = {"小猫", "追", "它"};

    /* x：dim0..2 = 角色 one-hot；dim4 = 2t（位置放大），dim5 = -2t。
     * 注意：真实模型里没有手工的 2t，位置特征是 wpe 学出来的（见线 embed），
     *       这里只是为了把头 1 的行为写到能手算。 */
    float x[3 * 8] = {
        /* 小猫 t=0 */ 1, 0, 0, 0,  0,  0, 0, 0,
        /* 追   t=1 */ 0, 1, 0, 0,  2, -2, 0, 0,
        /* 它   t=2 */ 0, 0, 1, 0,  4, -4, 0, 0,
    };

    /* W_QKV (24×8)。注意 Q 的行号即 Q 的维度：行 0..3 是头0 的 Q，行 4..7 是头1 的 Q。
     *  K 同理：行 8..11 头0 的 K，行 12..15 头1 的 K。V：行 16..19 头0，行 20..23 头1。 */
    float qkvw[24 * 8] = {0};
    /* 头0 的 Q：取角色维 → Q_h0(词) = 角色 one-hot */
    qkvw[0 * 8 + 0] = 1.0f;
    qkvw[1 * 8 + 1] = 1.0f;
    qkvw[2 * 8 + 2] = 1.0f;
    /* 头1 的 Q：取 dim4 (+2t) */
    qkvw[4 * 8 + 4] = 1.0f;
    /* 头0 的 K：取角色维 */
    qkvw[8 * 8 + 0] = 1.0f;
    qkvw[9 * 8 + 1] = 1.0f;
    qkvw[10 * 8 + 2] = 1.0f;
    /* 头1 的 K：取 dim5 (-2t)。分数 = (+2t)·(-2s)/2 = -2t·s：s 越小分越高 → 看向句首 */
    qkvw[12 * 8 + 5] = 1.0f;
    /* 两个头的 V 都原样搬运 x 的 dim0..3（角色部分） */
    for (int i = 0; i < 4; i++) {
        qkvw[(16 + i) * 8 + i] = 1.0f;
        qkvw[(20 + i) * 8 + i] = 1.0f;
    }
    float qkvb[24] = {0};

    /* W_proj (8×8)：前 4 行把头0/头1 的同一角色维相加 —— 两份看法叠加确认 */
    float projw[8 * 8] = {0};
    for (int i = 0; i < 4; i++) {
        projw[i * 8 + i] = 1.0f;        /* 头0 的 dim i */
        projw[i * 8 + 4 + i] = 1.0f;    /* 头1 的 dim i */
    }
    float projb[8] = {0};

    float qkv[3 * 24] = {0};
    float preatt[2 * 3 * 3] = {0};   /* (NH, T, T) */
    float att[2 * 3 * 3] = {0};
    float att_out[3 * 8] = {0};
    float proj_out[3 * 8] = {0};

    /* ═══ 跳点 ① 输入与一个 QKV ═══ */
    line_hop(ctx, "输入 x（T=3, C=8）与合并的 W_QKV（24×8）");
    line_mat(ctx, "x", x, T, C);
    line_note(ctx, "QKV 投影还是一次 matmul（见线 ln_attn）——多头发生在投影【之后】的切分上：");
    line_note(ctx, "  Q 的前 4 维 = 头0 的 Q，后 4 维 = 头1 的 Q（K、V 同理）");
    matmul_forward(qkv, x, qkvw, qkvb, B, T, C, 3 * C);
    for (int t = 0; t < T; t++) {
        char lbl[40];
        snprintf(lbl, sizeof(lbl), "Q(%s) 头0|头1", word[t]);
        line_vec(ctx, lbl, qkv + t * 24, C);
    }
    line_note(ctx, "头0 的 Q 装的是角色 one-hot；头1 的 Q 只装 +2t（小猫=0, 追=2, 它=4）。");

    /* ═══ 跳点 ② 两个头各自打分 ═══ */
    line_hop(ctx, "多头注意力：每个头在各自的 4 维子空间里独立打分");
    attention_forward(att_out, preatt, att, qkv, B, T, C, NH);
    for (int h = 0; h < NH; h++) {
        char title[64];
        snprintf(title, sizeof(title), "att 头 %d（%s）", h,
                 h == 0 ? "自扫门前雪" : "朝拜句首");
        line_attn_mat(ctx, title, att + h * T * T, T, word, word);
    }
    line_note(ctx, "头0：行「它」= [0.274, 0.274, 0.452] —— 自己的 0.452 最大，各扫门前雪。");
    line_note(ctx, "头1：行「它」= [0.982, 0.018, 0.000] —— 越靠句首权重越大，整齐朝拜「小猫」。");
    line_note(ctx, "同一个词，两个头看到的景象完全不同 —— 这就是「多头」的意义：");
    line_note(ctx, "  注意力打分只是 4 维子空间里的一次点积，一种 Q/K 投影只能问一类问题；");
    line_note(ctx, "  把 C 维切成 NH 份，等于让模型同时问 NH 类问题（语法？指代？位置？）。");
    {
        /* 头0 行「它」：角色点积 self=1, cross=0 → softmax([0, 0, 0.5]) */
        line_check_close(ctx, att[0 * T * T + 2 * T + 2], 0.4518f, 1e-3f, "头0: att[它→它] ≈ 0.452（自己最大）");
        line_check_close(ctx, att[0 * T * T + 2 * T + 0], 0.2741f, 1e-3f, "头0: att[它→小猫] ≈ 0.274");
        /* 头1 行「它」：分数 = -2t·s = [0, -4, -8] → softmax([0,-4,-8]) */
        line_check_close(ctx, att[1 * T * T + 2 * T + 0], 0.9817f, 1e-3f, "头1: att[它→小猫] ≈ 0.982（句首 dominate）");
        line_check_close(ctx, att[1 * T * T + 2 * T + 1], 0.0180f, 1e-3f, "头1: att[它→追] ≈ 0.018（距离指数衰减）");
    }

    /* ═══ 跳点 ③ 拼接：两个头的输出按维度首尾相接 ═══ */
    line_hop(ctx, "拼接：att_out = [头0 的输出 | 头1 的输出]（各 4 维，拼回 8 维）");
    line_mat(ctx, "att_out (3×8)", att_out, T, C);
    line_note(ctx, "前 4 维是头0 的加权 V（角色混合），后 4 维是头1 的加权 V（朝拜结果的复制）。");
    line_note(ctx, "到此为止两个头都没有说过话 —— 拼接只是把它们并排摆好，还没混合。");
    line_check_close(ctx, att_out[2 * 8 + 2], 0.4518f, 1e-3f, "att_out(它)[dim2] = 头0 的自身角色");
    line_check_close(ctx, att_out[2 * 8 + 4], 0.9817f, 1e-3f, "att_out(它)[dim4] = 头1 看到的「小猫」信号");

    /* ═══ 跳点 ④ W_proj 混合：两份看法叠加确认 ═══ */
    line_hop(ctx, "输出投影 W_proj：把两个头的看法重新混合成一份表示");
    matmul_forward(proj_out, att_out, projw, projb, B, T, C, C);
    line_mat(ctx, "proj_out (3×8)", proj_out, T, C);
    line_note(ctx, "本线 W_proj 把同一角色维在两个头里的值相加（行 i = 头0.dim_i + 头1.dim_i）：");
    line_note(ctx, "  proj_out(它)[小猫维] = 0.274(头0) + 0.982(头1) = 1.256 —— 两个头双重确认「小猫」");
    line_note(ctx, "  proj_out(它)[自身维] = 0.452(头0) + 0.000(头1) = 0.452 —— 头1 几乎不认「它自己」");
    line_check_close(ctx, proj_out[2 * 8 + 0], 1.2558f, 1e-3f, "proj_out(它)[小猫维] = 0.274 + 0.982 ≈ 1.256");
    line_check_true(ctx, proj_out[2 * 8 + 0] > proj_out[2 * 8 + 2],
                    "两个头混合后：小猫维(1.26) > 自身维(0.45) —— 句首信号被叠加放大");
    line_note(ctx, "");
    line_note(ctx, "真实模型里 W_proj 也是学出来的（它是注意力之后第二个大权重矩阵）。");
    line_note(ctx, "完整多头公式：MultiHead = Concat(head_1..head_N) · W_proj，");
    line_note(ctx, "本线 head_i = softmax(Q_i·K_i/√d)·V_i，每个 head_i 都在 c/attention.c 里真跑。");

    return line_summary(ctx);
}
