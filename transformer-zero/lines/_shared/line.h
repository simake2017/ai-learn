/* =============================================================================
 * line.h - 学习线公共 harness
 *
 * 📚 什么是「线」？
 *   一条线 = 一个端到端调用链，是可运行的业务代码 + 逐跳日志，不是单元测试。
 *   c/ 目录里的算子按机制类别横着放（layernorm.c / attention.c / model.c ...），
 *   而一条真实的数据流是竖着穿过它们的：
 *
 *     x ──LN──▶ ln1 ──matmul──▶ qkv ──attention──▶ att_out ──proj──▶ +x ──▶ 残差
 *
 *   这条竖线不属于任何一个文件 —— attention.c 不知道自己是被 LN 的输出喂进来的。
 *   学习时最难啃的恰恰是它，lines/ 就是把这些竖线一根根抽出来单跑。
 *
 * 🧱 铁律（继承自 myos/lines）：
 *   lines/ 里不出现任何模型逻辑。线里只准调用 c/ 里的真算子 + 打印 + 校验。
 *   如果需要"顺手实现一下"，说明 c/ 缺东西 —— 补到 c/，再回来串线。
 *
 * 🔇 安静模式：
 *   环境变量 LINE_QUIET=1 时，过程叙述全部静默，只有 check 失败才打印一行。
 *   这样可以把所有线塞进 CI / run_all.sh，全绿时一声不吭。
 * =============================================================================
 */

#ifndef LINE_H
#define LINE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>

/* === ANSI 颜色 === */
#define LN_GREEN  "\033[32m"
#define LN_RED    "\033[31m"
#define LN_YELLOW "\033[33m"
#define LN_CYAN   "\033[36m"
#define LN_BOLD   "\033[1m"
#define LN_DIM    "\033[2m"
#define LN_RESET  "\033[0m"

/* === 线上下文 === */
typedef struct {
    const char* name;   // 线名（如 "ln_attn"）
    int quiet;          // 安静模式：1 = 只打印 check 失败
    int hop;            // 当前跳点序号
    int pass, fail, total;  // check 计数
} LineCtx;

/* === 注册表条目（每条线文件导出一个 run 函数） === */
typedef struct {
    const char* name;    // 命令行名字: ./lines_all <name>
    const char* title;   // 一句话说明这条线贯穿了什么
    int (*run)(LineCtx* ctx);
} LineDef;

/* === 生命周期 === */
void line_ctx_init(LineCtx* ctx, const char* name);

/* === 逐跳日志 ===
 * line_hop  : 大标题，跳点编号自动递增 —— 每一"跳"就是数据穿过的一次算子
 * line_note : 跳点内的叙述行（缩进）
 * 安静模式下两者都不落屏 */
void line_hop(LineCtx* ctx, const char* fmt, ...);
void line_note(LineCtx* ctx, const char* fmt, ...);

/* === 数据打印（安静模式下不落屏） === */
void line_vec(LineCtx* ctx, const char* name, const float* v, int n);
void line_ivec(LineCtx* ctx, const char* name, const int* v, int n);
void line_mat(LineCtx* ctx, const char* name, const float* m, int rows, int cols);
/* 注意力方阵专用：行 = 查询位置，列 = 被关注的 key 位置，打印权重热力文本 */
void line_attn_mat(LineCtx* ctx, const char* name, const float* att,
                   int T, const char* const* row_labels, const char* const* col_labels);

/* === 校验（安静模式下只打印失败行） === */
void line_check_close(LineCtx* ctx, float actual, float expected, float tol, const char* msg);
void line_check_true(LineCtx* ctx, int cond, const char* msg);

/* === 汇总（安静模式下不落屏；返回值：fail 数） === */
int line_summary(const LineCtx* ctx);

#endif /* LINE_H */
