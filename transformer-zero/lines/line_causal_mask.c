/* =============================================================================
 * line_causal_mask.c - 学习线：因果掩码到底在防什么
 *
 * 📚 这条线贯穿的竖线（attention_forward 内部那个 t2 <= t 的循环，attention.c）：
 *
 *   掩码版   attention_forward:               行 t 只在 {0..t}   上归一化
 *   双向版   attention_forward_bidirectional: 行 t 在 {0..T-1} 全列上归一化
 *
 * 🎭 案例：「小猫 追 它」。把 Q/K 全置零（所有分数相等 → 均匀关注），
 *   这样两种掩码的差别就是【纯结构差异】，不被任何打分偏好掩盖：
 *   - 掩码版：第 0 行 = [1, 0, 0] —— 「小猫」的输出只含自己
 *   - 双向版：第 0 行 = [1/3, 1/3, 1/3] —— 「小猫」被迫吸收「追」和「它」
 *     而它俩在生成场景里【还没出生】
 *
 * 🔍 训练视角的真相：训练时 inputs[t+1] 恰好就是 targets[t] 的标准答案 ——
 *   不掩码 = 开卷考试；且 GPT 逐字生成时未来根本不存在，
 *   训练/推理必须同分布，这就是因果掩码存在的理由
 *   （也是 GPT 与 BERT 的分野：BERT 双向，所以训练要用 [MASK] 主动遮题）。
 * =============================================================================
 */

#include <stdio.h>
#include "../c/transformer.h"
#include "_shared/line.h"

int line_causal_mask_run(LineCtx* ctx) {
    int B = 1, T = 3, C = 4, NH = 1;
    const char* word[3] = {"小猫", "追", "它"};

    /* x：角色 one-hot。Q/K 打分无关（全零权重 → 分数全相等），V 原样搬运角色 */
    float x[3 * 4] = {
        1, 0, 0, 0,   /* 小猫 */
        0, 1, 0, 0,   /* 追   */
        0, 0, 1, 0,   /* 它   */
    };

    /* W_QKV (12×4)：Q/K 全部置零（均匀关注的来源），V = 单位阵 */
    float qkvw[12 * 4] = {0};
    for (int i = 0; i < 4; i++) qkvw[(8 + i) * 4 + i] = 1.0f;   /* 后 4 行 = W_V */
    float qkvb[12] = {0};

    float qkv[3 * 12] = {0};
    float preatt_m[3 * 3] = {0}, att_m[3 * 3] = {0}, out_m[3 * 4] = {0};
    float preatt_b[3 * 3] = {0}, att_b[3 * 3] = {0}, out_b[3 * 4] = {0};

    /* ═══ 跳点 ① 输入：一个 Q/K 无偏好的世界 ═══ */
    line_hop(ctx, "输入：「小猫 追 它」，Q/K 置零 → 所有打分相等 → 均匀关注");
    line_mat(ctx, "x (3×4)", x, T, C);
    line_note(ctx, "把 Q/K 权重清零是个刻意的实验设计：");
    line_note(ctx, "  softmax(全零分数) = 均匀分布，此时两种掩码的差别不再被任何");
    line_note(ctx, "  「谁更相关」的偏好遮盖，掩码本身的形状差异会裸露出来。");
    matmul_forward(qkv, x, qkvw, qkvb, B, T, C, 3 * C);
    line_note(ctx, "QKV 投影照常跑一次（Q=0, K=0, V=x）—— 线 ln_attn 的那次 matmul。");

    /* ═══ 跳点 ② 掩码版 ═══ */
    line_hop(ctx, "掩码版 attention_forward：行 t 只在「过去」上归一化");
    attention_forward(out_m, preatt_m, att_m, qkv, B, T, C, NH);
    line_attn_mat(ctx, "att 掩码版", att_m, T, word, word);
    line_mat(ctx, "att_out 掩码版 (3×4)", out_m, T, C);
    line_note(ctx, "第 0 行 = [1, 0, 0]：「小猫」只看得见自己，输出 = 自己的角色卡。");
    line_note(ctx, "第 1 行 = [0.5, 0.5, 0]：「追」平均分给小猫和自己，未来的「它」被清零。");
    line_check_close(ctx, att_m[0 * 3 + 0], 1.0f, 1e-5f, "掩码版 att[0][0] = 1");
    line_check_close(ctx, att_m[0 * 3 + 1], 0.0f, 1e-5f, "掩码版 att[0][1] = 0（未来被掩）");
    line_check_close(ctx, att_m[0 * 3 + 2], 0.0f, 1e-5f, "掩码版 att[0][2] = 0");
    line_check_close(ctx, att_m[1 * 3 + 0], 0.5f, 1e-5f, "掩码版 att[1][0] = 1/2");
    line_check_close(ctx, out_m[0], 1.0f, 1e-5f, "掩码版 att_out(小猫)[dim0] = 1（干净）");

    /* ═══ 跳点 ③ 双向版 ═══ */
    line_hop(ctx, "双向版 attention_forward_bidirectional：行 t 在「全部」位置上归一化");
    attention_forward_bidirectional(out_b, preatt_b, att_b, qkv, B, T, C, NH);
    line_attn_mat(ctx, "att 双向版", att_b, T, word, word);
    line_mat(ctx, "att_out 双向版 (3×4)", out_b, T, C);
    line_note(ctx, "第 0 行 = [1/3, 1/3, 1/3]：「小猫」的输出混入了「追」和「它」——");
    line_note(ctx, "  在逐字生成「小猫」那一刻，后面两个词根本不存在，这份信息是幽灵。");
    line_check_close(ctx, att_b[0 * 3 + 1], 1.0f / 3.0f, 1e-4f, "双向版 att[0][1] = 1/3（未来泄漏）");
    line_check_close(ctx, out_b[0 * 4 + 1], 1.0f / 3.0f, 1e-4f,
                     "双向版 att_out(小猫)[dim1=追维] = 1/3（被「追」污染）");
    line_check_true(ctx, out_m[0] > out_b[0 * 4 + 0] && out_b[0 * 4 + 1] > 0.0f,
                    "同一输入两种输出：掩码版干净，双向版被未来稀释");

    /* ═══ 跳点 ④ 为什么训练必须掩码 ═══ */
    line_hop(ctx, "语义：掩码防的不是「看」，是「泄题」");
    line_note(ctx, "语言模型训练时，位置 t 的预测目标是 targets[t] = inputs[t+1] ——");
    line_note(ctx, "  下一个词就躺在输入序列里！不掩码，等于把答案放在桌上考开卷。");
    line_note(ctx, "loss 会假降：模型学会的不是「预测下一个词」，而是「抄写桌上的答案」。");
    line_note(ctx, "更根本的是：GPT 推理（generate 线）是逐字生成的，未来永远不存在；");
    line_note(ctx, "  训练若不掩码，训练分布 ≠ 推理分布，学出来的东西推理时用不上。");
    line_note(ctx, "对照：BERT 是双向的，所以它训练时要把被预测词替换成 [MASK] 遮题 ——");
    line_note(ctx, "  同一件事的两份答案：要么遮住未来（GPT），要么遮住答案（BERT）。");
    line_check_true(ctx, 1, "结论已述：因果掩码 = 训练/推理同分布 + 防泄题");

    return line_summary(ctx);
}
