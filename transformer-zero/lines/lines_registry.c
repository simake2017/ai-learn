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
int line_tiny_train_run(LineCtx* ctx);

const LineDef g_line_registry[] = {
    {
        "ln_attn",
        "机制线: LN → QKV → 注意力 → 投影 → 残差（手工构造「小猫追它」代词消解）",
        line_ln_attn_run,
    },
    {
        "tiny_train",
        "训练线: 从零训练 abab 最小 GPT（forward → 数值梯度 → AdamW → loss 0.693→≈0）",
        line_tiny_train_run,
    },
};

const int g_line_count = (int)(sizeof(g_line_registry) / sizeof(g_line_registry[0]));
