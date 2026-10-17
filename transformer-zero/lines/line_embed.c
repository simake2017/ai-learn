/* =============================================================================
 * line_embed.c - 学习线：token id 怎么变成向量（embedding 机制）
 *
 * 📚 这条线贯穿的竖线（gpt_forward Step 1，model.c）：
 *
 *   inputs (B,T) 整数 ──▶ tok_emb = wte[id]  ──┐
 *                                              ├─▶ x = tok_emb + pos_emb
 *                    t ──▶ pos_emb = wpe[t]  ──┘
 *
 * 🎭 案例：4 个词的小世界 {小猫, 追, 它, 鱼}
 *   - wte（token embedding 表）：每个词一行，是「词的身份」——查表即得
 *   - wpe（position embedding 表）：每个位置一行，是「词的座次」
 *   - 关键洞察：同一个词坐在不同位置，x 是不一样的 ——
 *     注意力机制本身分不清先后，先后信息全靠 wpe 在这里注入
 *
 * 🔍 手工构造：词用 one-hot 身份（dim0..3），位置用 dim4..6 编码座次。
 *   可以精确手算：x[t] = wte[inputs[t]] + wpe[t]
 * =============================================================================
 */

#include <stdio.h>
#include "../c/transformer.h"
#include "_shared/line.h"

int line_embed_run(LineCtx* ctx) {
    /* ──────────────────────────────────────────────
     * 舞台：vocab=4, T=3 句话「小猫 追 它」, C=8
     * ────────────────────────────────────────────── */
    int B = 1, T = 3, C = 8;
    const char* word[4] = {"小猫", "追", "它", "鱼"};
    int inputs[3] = {0, 1, 2};          // 小猫 追 它
    const char* sent[3] = {"小猫", "追", "它"};

    /* wte (4×8)：每个词的身份卡。dim0..3 是 one-hot 身份，其余维度留给"语义生长" */
    float wte[4 * 8] = {
        /* 小猫 */ 1, 0, 0, 0,  0, 0, 0, 0,
        /* 追   */ 0, 1, 0, 0,  0, 0, 0, 0,
        /* 它   */ 0, 0, 1, 0,  0, 0, 0, 0,
        /* 鱼   */ 0, 0, 0, 1,  0, 0, 0, 0,
    };

    /* wpe (T×8 就够本例用)：dim4..6 编码座次，座次越高亮得越多。
     * 真实模型里 wpe 是学习出来的（trained_attn 线会看到它学成近反向结构），
     * 这里手工给可读的值。 */
    float wpe[3 * 8] = {
        /* pos 0 */ 0, 0, 0, 0,  0.1f, 0,    0,    0,
        /* pos 1 */ 0, 0, 0, 0,  0.1f, 0.2f, 0,    0,
        /* pos 2 */ 0, 0, 0, 0,  0.1f, 0.2f, 0.4f, 0,
    };

    float tok_emb[3 * 8] = {0};
    float x[3 * 8] = {0};

    /* ═══ 跳点 ① 输入：一串整数 id ═══ */
    line_hop(ctx, "输入：token id（整数，模型的\"原材料\"）");
    line_note(ctx, "文本经分词后变成整数下标，模型不识字，只认下标：");
    {
        char line[64] = {0};
        for (int t = 0; t < T; t++) {
            strcat(line, word[inputs[t]]);
            strcat(line, " ");
        }
        line_note(ctx, "  「%s」 → id = [%d, %d, %d]", line, inputs[0], inputs[1], inputs[2]);
    }
    line_note(ctx, "整数本身没有语义（0 和 1 的距离不比 0 和 3 更近），");
    line_note(ctx, "语义必须先从一张「查词表」里取出来 —— 这就是 wte。");

    /* ═══ 跳点 ② wte 查表：id → 词向量 ═══ */
    line_hop(ctx, "token embedding：wte 查表，id → 词的身份向量");
    line_mat(ctx, "wte (4×8)", wte, 4, C);
    embedding_forward(x, inputs, wte, wpe, B, T, C);   /* 一次调用把两步都做了， */
    memcpy(tok_emb, x, sizeof(tok_emb));               /* 这里先存档待会拆解 */
    line_note(ctx, "查表逻辑：tok_emb[t] = wte[inputs[t]] —— 第 t 个词直接抄表的第 inputs[t] 行。");
    for (int t = 0; t < T; t++) {
        char lbl[32];
        snprintf(lbl, sizeof(lbl), "tok_emb[%s]", sent[t]);
        line_vec(ctx, lbl, tok_emb + t * C, C);
    }
    line_note(ctx, "「小猫」「它」的身份卡互不重叠（one-hot），谁的特征在哪一维一目了然。");
    line_check_close(ctx, tok_emb[0], 1.0f, 1e-5f, "tok_emb[小猫][dim0] = 1（抄的 wte 第 0 行）");
    line_check_close(ctx, tok_emb[2 * 8 + 2], 1.0f, 1e-5f, "tok_emb[它][dim2] = 1");
    line_check_close(ctx, tok_emb[2 * 8 + 0], 0.0f, 1e-5f, "tok_emb[它][dim0] = 0（它身上没有「小猫」）");

    /* ═══ 跳点 ③ wpe 注入座次：同词不同位 ═══ */
    line_hop(ctx, "position embedding：wpe 注入「座次」，x = tok_emb + wpe");
    line_mat(ctx, "wpe (3×8)", wpe, T, C);
    line_note(ctx, "embedding_forward 的完整动作：x[t] = wte[inputs[t]] + wpe[t] —— 逐元素相加。");
    for (int t = 0; t < T; t++) {
        char lbl[32];
        snprintf(lbl, sizeof(lbl), "x[%d]=%s", t, sent[t]);
        line_vec(ctx, lbl, x + t * C, C);
    }
    line_note(ctx, "对比跳点 ②：dim0..3 的身份没变，变的是 dim4..6 —— 位置签名叠了上去。");
    line_note(ctx, "");
    line_note(ctx, "关键洞察：注意力打分是点积，对「词的排列顺序」天然盲视。");
    line_note(ctx, "  把「小猫 追 它」打乱成「它 追 小猫」，只要每行的 x 不变，注意力算出一模一样的分数。");
    line_note(ctx, "  「谁先谁后」这条信息，必须靠 wpe 在进门这一刻写进向量，之后所有计算才有先后可言。");
    line_check_close(ctx, x[0 * 8 + 4], 0.1f, 1e-5f, "x[0][dim4] = 0 + 0.1（位置 0 的签名）");
    line_check_close(ctx, x[2 * 8 + 6], 0.4f, 1e-5f, "x[2][dim6] = 0 + 0.4（位置 2 的签名）");

    /* ═══ 跳点 ④ 验证：同词坐不同位，x 不同 ═══ */
    line_hop(ctx, "验证：同一个词换个位置，还是同一个词吗？");
    int inputs2[2] = {0, 0};      // 「小猫 小猫」
    float x2[2 * 8] = {0};
    float wpe2[2 * 8] = {0};      /* 复用前两个位置 */
    memcpy(wpe2, wpe, sizeof(wpe2));
    embedding_forward(x2, inputs2, wte, wpe2, 1, 2, C);
    line_vec(ctx, "x2[0]=小猫@位置0", x2 + 0 * C, C);
    line_vec(ctx, "x2[1]=小猫@位置1", x2 + 1 * C, C);
    line_note(ctx, "身份部分(dim0=1)完全相同，座次部分(dim4/5)不同 ——");
    line_note(ctx, "对模型来说，这是「两个共享同一身份、但坐在不同位置的小猫」。");
    line_check_close(ctx, x2[0], x2[8], 1e-5f, "身份维相同：两个「小猫」dim0 都是 1");
    line_check_true(ctx, x2[5] != x2[8 + 5], "座次维不同：位置 1 比位置 0 在 dim5 上多 0.2");
    line_note(ctx, "");
    line_note(ctx, "至此 x 已备齐，接下来进入 block：LN → QKV → 注意力（见线 ln_attn），");
    line_note(ctx, "训练时 wte/wpe 也会一起被梯度更新 —— 词的「身份卡」和「座次表」都是学出来的。");

    return line_summary(ctx);
}
