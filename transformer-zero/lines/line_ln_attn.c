/* =============================================================================
 * line_ln_attn.c - 学习线：LN 的数据怎么走进 QKV，注意力又怎么把语义搬走
 *
 * 📚 这条线贯穿的竖线（对应 block_forward 子层 1 的真实接线，model.c）：
 *
 *   x ──LN──▶ ln1 ──matmul(W_QKV)──▶ qkv ──attention──▶ att_out
 *       ──matmul(W_proj)──▶ ──+x(residual)──▶ residual2
 *
 * 🎭 真实案例：代词消解「小猫 追 它」
 *   人在读这句话时知道「它 = 小猫」，因为大脑把「它」的指针对回了「小猫」。
 *   我们要手工搭一个迷你注意力，让它也完成这件事：
 *   读到「它」时，把注意力集中到「小猫」身上，把「小猫」的语义搬进「它」的表示。
 *
 * 🧮 手工构造思路（C=8, NH=1, T=3）：
 *   - 每个词的原始 embedding 是其"角色"维被点亮：小猫=dim0, 追=dim1, 它=dim2
 *     且三个词的点亮量级故意不同（8 / 2 / 1）——用来展示 LN 的尺度不变性
 *   - LN 后每个词变成同一结构：自己维度 sqrt(7)≈2.6458，其余 -1/sqrt(7)≈-0.378
 *   - W_Q 只留一行（取 dim2）：「它」的 Q 在 dim0 上出现大值 a
 *     W_K 只留一行（取 dim0）：「小猫」的 K 在 dim0 上出现大值 a
 *     → 「它」的 Q · 「小猫」的 K = a*a = 7，远大于其它组合 → softmax 后 ≈ 0.894
 *   - W_V = 单位阵：V 就是 LN 输出本身，谁的 V 被加权，谁的语义就搬进输出
 *
 * 🔍 预期结果：
 *   注意力矩阵「它」那一行 = [0.894, 0.053, 0.053] ——「它」89% 在看「小猫」
 *   残差相加后「它」的表示里，小猫特征(dim0=2.33) 反超自身特征(dim2=0.78)
 * =============================================================================
 */

#include <stdio.h>
#include "../c/transformer.h"
#include "_shared/line.h"

int line_ln_attn_run(LineCtx* ctx) {
    /* ──────────────────────────────────────────────
     * 舞台：B=1 句话，T=3 个词，C=8 维特征，NH=1 个头
     * ────────────────────────────────────────────── */
    int B = 1, T = 3, C = 8, NH = 1;
    const char* word[3] = {"小猫", "追", "它"};

    /* 原始 embedding：角色维点亮，量级故意悬殊（8 / 2 / 1）。
     * 为什么要 LN？—— 不先归一化的话，「小猫」的 8 会淹没一切点积。 */
    float x[3 * 8] = {
        /* 小猫: dim0 点亮，量级 8 */
        8, 0, 0, 0, 0, 0, 0, 0,
        /* 追:   dim1 点亮，量级 2 */
        0, 2, 0, 0, 0, 0, 0, 0,
        /* 它:   dim2 点亮，量级 1 */
        0, 0, 1, 0, 0, 0, 0, 0,
    };

    /* LN 参数：恒等仿射（先不看 gamma/beta 的作用，专注归一化本身） */
    float ln1w[8] = {1,1,1,1,1,1,1,1};
    float ln1b[8] = {0,0,0,0,0,0,0,0};

    /* W_QKV (3C × C) = (24 × 8)：前 8 行 W_Q，中 8 行 W_K，后 8 行 W_V。
     * 这是「一次 matmul 算出 Q/K/V」的合并权重，行号 = 输出维度。 */
    float qkvw[24 * 8] = {0};
    /* W_Q 只留 row0 = e_2：Q 的 dim0 = LN 输出的 dim2（「它」的自身维） */
    qkvw[0 * 8 + 2] = 1.0f;
    /* W_K 只留 row8 = e_0（第 8 行是 W_K 的第 0 行）：K 的 dim0 = LN 输出的 dim0（「小猫」维） */
    qkvw[8 * 8 + 0] = 1.0f;
    /* W_V = 单位阵：V 原样搬运 LN 输出 */
    for (int i = 0; i < 8; i++) qkvw[(16 + i) * 8 + i] = 1.0f;

    float qkvb[24] = {0};

    /* 输出投影 W_proj = 单位阵（本线聚焦注意力，投影先直通） */
    float projw[8 * 8] = {0};
    for (int i = 0; i < 8; i++) projw[i * 8 + i] = 1.0f;
    float projb[8] = {0};

    /* 激活缓存 */
    float ln1[3 * 8] = {0};        // LN 输出
    float qkv[3 * 24] = {0};       // QKV 拼接 (T, 3C)
    float preatt[3 * 3] = {0};     // 注意力原始分数（上三角是掩码区，打印恒为 0）
    float att[3 * 3] = {0};        // softmax 后的注意力权重
    float att_out[3 * 8] = {0};    // 加权 V 后的注意力输出
    float residual2[3 * 8] = {0};  // 残差相加后的 block 输出

    /* ═══ 跳点 ① 输入：一句有指代关系的话 ═══ */
    line_hop(ctx, "输入 x：「小猫 追 它」的原始 embedding (T=3, C=8)");
    line_note(ctx, "每行是一个词的 8 维向量，哪个维度亮哪个维度就是它的「角色」：");
    for (int t = 0; t < T; t++) line_vec(ctx, word[t], x + t * C, C);
    line_note(ctx, "注意三个词的点亮量级完全不同：小猫=8, 追=2, 它=1");
    line_note(ctx, "问题：如果直接拿它做点积，「小猫」的量级会压过一切，模型只看得见小猫。");

    /* ═══ 跳点 ② LayerNorm：把量级拉平 ═══ */
    line_hop(ctx, "LN：x → ln1（每个词单独归一化：减均值、除标准差）");
    layernorm_forward(ln1, x, ln1w, ln1b, B, T, C);
    for (int t = 0; t < T; t++) line_vec(ctx, word[t], ln1 + t * C, C);
    line_note(ctx, "LN 是「逐词」做的：每个词减自己的均值、除自己的标准差。");
    line_note(ctx, "结果：量级 8/2/1 消失了，三个词变成同一结构 ——");
    line_note(ctx, "  自己角色维 = sqrt(7) ≈ 2.6458，其余 7 维 = -1/sqrt(7) ≈ -0.378");
    line_note(ctx, "这就是 LN 的尺度不变性：8 倍的输入，归一化后一模一样。");
    {
        float a = sqrtf(7.0f), b = -1.0f / sqrtf(7.0f);
        line_check_close(ctx, ln1[0], a, 1e-3f, "LN(小猫)[dim0=小猫维] = sqrt(7) ≈ 2.6458");
        line_check_close(ctx, ln1[1], b, 1e-3f, "LN(小猫)[dim1] = -1/sqrt(7) ≈ -0.378");
        line_check_close(ctx, ln1[2], b, 1e-3f, "LN(小猫)[dim2=它维] 也是 -0.378（小猫身上没有「它」）");
        line_check_close(ctx, ln1[2 * 8 + 2], a, 1e-3f, "LN(它)[dim2=自身维] 同样 = sqrt(7) —— 量级 1 和 8 归一化后无差别");
    }

    /* ═══ 跳点 ③ QKV 投影：ln1 一次 matmul 分成 Q/K/V 三份 ═══ */
    line_hop(ctx, "QKV 投影：ln1 (3×8) ──×W_QKV(24×8)──▶ qkv (3×24)");
    matmul_forward(qkv, ln1, qkvw, qkvb, B, T, C, 3 * C);
    line_note(ctx, "内存里每词占 24 维连续空间 [Q(8) | K(8) | V(8)]，指针一偏就拆开：");
    line_note(ctx, "  Q_t = qkv + t*24,  K_t = Q_t + 8,  V_t = K_t + 8");
    for (int t = 0; t < T; t++) {
        char lbl[32];
        float* Q = qkv + t * 3 * C;
        float* K = Q + C;
        snprintf(lbl, sizeof(lbl), "Q(%s)", word[t]); line_vec(ctx, lbl, Q, C);
        snprintf(lbl, sizeof(lbl), "K(%s)", word[t]); line_vec(ctx, lbl, K, C);
        snprintf(lbl, sizeof(lbl), "V(%s)", word[t]); line_vec(ctx, lbl, K + C, C);
    }
    line_note(ctx, "Q 只留 dim0（取的是各词身上的「它」维），K 只留 dim0（取的是「小猫」维）：");
    line_note(ctx, "  「它」的 Q dim0 = 2.6458（它自己身上「它」维很亮）");
    line_note(ctx, "  「小猫」的 K dim0 = 2.6458（小猫身上「小猫」维很亮）");
    line_note(ctx, "  → 等一下点积：Q(它)·K(小猫) = 2.6458² = 7.0，全场最高。指代指针已埋好。");
    line_check_close(ctx, qkv[2 * 3 * C + 0], sqrtf(7.0f), 1e-3f, "Q(它)[dim0] = sqrt(7)");
    line_check_close(ctx, qkv[0 * 3 * C + C + 0], sqrtf(7.0f), 1e-3f, "K(小猫)[dim0] = sqrt(7)");

    /* ═══ 跳点 ④ Attention：Q·K 打分 → softmax → 加权 V ═══ */
    line_hop(ctx, "Attention：preatt 打分 → softmax 归一 → 加权求和 V");
    attention_forward(att_out, preatt, att, qkv, B, T, C, NH);
    line_note(ctx, "preatt = Q·K / sqrt(8)，上三角被因果掩码清零（每个词看不到未来）：");
    line_mat(ctx, "preatt (3×3)", preatt, T, T);
    line_note(ctx, "看「它」那行：对「小猫」的分数 2.47，对「追」「它自己」都只有 -0.35 —— 差距悬殊。");
    line_check_close(ctx, preatt[2 * 3 + 0], 7.0f / sqrtf(8.0f), 1e-3f, "preatt[它→小猫] = 7/sqrt(8) ≈ 2.475");
    line_check_close(ctx, preatt[2 * 3 + 1], -1.0f / sqrtf(8.0f), 1e-3f, "preatt[它→追] = -1/sqrt(8) ≈ -0.354");
    line_note(ctx, "softmax 把分数压成「关注比例」，每行和 = 1：");
    line_attn_mat(ctx, "att 注意力权重", att, T, word, word);
    line_note(ctx, "「它」行 = [0.894, 0.053, 0.053]：89% 的注意力给了「小猫」—— 代词消解发生了！");
    line_check_close(ctx, att[2 * 3 + 0], 0.8944f, 1e-3f, "att[它→小猫] ≈ 0.894");
    line_check_true(ctx, att[2 * 3 + 0] > 0.85f, "「它」对「小猫」的关注 > 85%");
    line_check_true(ctx, att[2 * 3 + 1] < 0.1f && att[2 * 3 + 2] < 0.1f, "「它」对「追」和「自己」的关注都 < 10%");

    /* ═══ 跳点 ⑤ 加权 V：语义被搬进输出 ═══ */
    line_hop(ctx, "加权求和 V：谁被关注，谁的语义就进 att_out");
    line_mat(ctx, "att_out (3×8)", att_out, T, C);
    line_note(ctx, "att_out(它) = 0.894×V(小猫) + 0.053×V(追) + 0.053×V(它) ≈ V(小猫)");
    line_note(ctx, "V = LN 输出，所以 V(小猫) 的「小猫维」(dim0) 是全场最亮的 2.6458：");
    line_note(ctx, "  → att_out(它)[dim0] ≈ 2.33 被点亮 —— 「小猫」的语义搬进了「它」！");
    line_check_close(ctx, att_out[2 * 8 + 0], 2.3265f, 1e-3f, "att_out(它)[小猫维] ≈ 2.327（全场最大）");
    line_check_true(ctx, att_out[2 * 8 + 0] > att_out[2 * 8 + 2],
                    "「它」的输出里，小猫维(2.33) 亮过自身维(-0.22)");

    /* ═══ 跳点 ⑥ 输出投影 + 残差：和原路信息汇合 ═══ */
    line_hop(ctx, "输出投影(W=单位阵, 直通) + 残差：residual2 = x + att_out");
    matmul_forward(att_out, att_out, projw, projb, B, T, C, C);
    residual_forward(residual2, x, att_out, B * T * C);
    line_mat(ctx, "residual2 (3×8)", residual2, T, C);
    line_note(ctx, "残差 = 抄近道：注意力只负责「搬运增量」，原词信息(x)一条高速直连始终保留。");
    line_note(ctx, "对比「它」残差前后的自身维：x(它)[dim2]=1 → residual2(它)[dim2]≈0.78，");
    line_note(ctx, "  而小猫维从 0 → 2.33。一句话读完，「它」的表示里「小猫」已经反超「自己」。");
    line_check_close(ctx, residual2[2 * 8 + 0], 2.3265f, 1e-3f, "residual2(它)[小猫维] ≈ 2.327");
    line_check_close(ctx, residual2[2 * 8 + 2], 0.7816f, 1e-3f, "residual2(它)[自身维] ≈ 1 + (-0.218) = 0.782");
    line_check_true(ctx, residual2[2 * 8 + 0] > residual2[2 * 8 + 2],
                    "残差后：「它」的小猫维(2.33) > 自身维(0.78) —— 指代消解写进了向量");
    line_note(ctx, "「小猫」和「追」的行也各加回了各自的 x —— 残差对每词都生效。");

    line_note(ctx, "");
    line_note(ctx, "至此子层 1 走完。真实模型里 W_QKV/W_proj 是学习出来的（见线 tiny_train），");
    line_note(ctx, "但数据流一模一样：LN → 一次 matmul 出 QKV → 打分/归一/加权 → 投影 → +x。");

    return line_summary(ctx);
}
