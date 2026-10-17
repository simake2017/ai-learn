/* =============================================================================
 * lines_registry.c - 学习线注册表
 *
 * 新增一条线只需两步：
 *   1. 写 lines/line_xxx.c，导出 int line_xxx_run(LineCtx* ctx)
 *   2. 在下表中加一行 { "xxx", "一句话说明", line_xxx_run }
 * lines_all 自动获得 list / 单跑 / all 三个入口，无需改其它文件。
 * =============================================================================
 */

#include "_shared/line.h"

int line_ln_attn_run(LineCtx* ctx);
int line_embed_run(LineCtx* ctx);
int line_multi_head_run(LineCtx* ctx);
int line_causal_mask_run(LineCtx* ctx);
int line_tiny_train_run(LineCtx* ctx);
int line_trained_attn_run(LineCtx* ctx);
int line_generate_run(LineCtx* ctx);

const LineDef g_line_registry[] = {
    {
        "embed",
        "机制线: token id → wte/wpe → x（词的身份卡 + 位置座次如何注入）",
        line_embed_run,
    },
    {
        "ln_attn",
        "机制线: LN → QKV → 注意力 → 投影 → 残差（手工构造「小猫追它」代词消解）",
        line_ln_attn_run,
    },
    {
        "multi_head",
        "机制线: QKV → 2 头拆分/并行/拼接 → W_proj 混合（头0看自己, 头1朝拜句首）",
        line_multi_head_run,
    },
    {
        "causal_mask",
        "机制线: 因果掩码防什么（掩码 vs 双向同场对比，位置0被未来污染实验）",
        line_causal_mask_run,
    },
    {
        "tiny_train",
        "训练线: 从零训练 abab 最小 GPT（forward → 数值梯度 → AdamW → loss 0.693→≈0.03）",
        line_tiny_train_run,
    },
    {
        "trained_attn",
        "验证线: 解剖 tiny_train 训出的模型（注意力退化为历史平均池化，奇偶规则藏在 wpe）",
        line_trained_attn_run,
    },
    {
        "generate",
        "验证线: 自回归生成闭环（seed a→abab..., seed b→baba...，同一套权重展开两种序列）",
        line_generate_run,
    },
};

const int g_line_count = (int)(sizeof(g_line_registry) / sizeof(g_line_registry[0]));
