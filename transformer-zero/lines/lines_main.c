/* =============================================================================
 * lines_main.c - 学习线合集入口
 *
 * 用法：
 *   ./lines_all list          列出所有线
 *   ./lines_all ln_attn       跑一条线（逐跳日志 + 校验）
 *   ./lines_all all           按注册顺序全跑
 *
 * 环境变量：
 *   LINE_QUIET=1    安静模式：过程不打印，只有 check 失败才冒一行（适合 CI）
 *
 * 退出码：失败的 check 总数（0 = 全过）
 * =============================================================================
 */

#include <stdio.h>
#include "_shared/line.h"

extern const LineDef g_line_registry[];
extern const int g_line_count;

static int run_one(const LineDef* def) {
    LineCtx ctx;
    line_ctx_init(&ctx, def->name);
    if (!ctx.quiet) {
        printf("\033[36m\033[1m╔══════════════════════════════════════════╗\033[0m\n");
        printf("\033[36m\033[1m║  线: %-33s║\033[0m\n", def->name);
        printf("\033[36m\033[1m╚══════════════════════════════════════════╝\033[0m\n");
        printf("\033[2m%s\033[0m\n", def->title);
    }
    return def->run(&ctx);
}

int main(int argc, char** argv) {
    if (argc < 2 || strcmp(argv[1], "list") == 0 || strcmp(argv[1], "help") == 0) {
        printf("学习线列表（用法: ./lines_all <名字> | all | list）:\n\n");
        for (int i = 0; i < g_line_count; i++) {
            printf("  %-12s %s\n", g_line_registry[i].name, g_line_registry[i].title);
        }
        printf("\n提示: LINE_QUIET=1 安静模式，只有校验失败才打印\n");
        return 0;
    }

    if (strcmp(argv[1], "all") == 0) {
        int fails = 0;
        for (int i = 0; i < g_line_count; i++) {
            fails += run_one(&g_line_registry[i]);
        }
        if (fails == 0) {
            printf("\n\033[32m\033[1m✅ 全部 %d 条线通过\033[0m\n", g_line_count);
        } else {
            printf("\n\033[31m\033[1m❌ 共 %d 项校验失败\033[0m\n", fails);
        }
        return fails;
    }

    for (int i = 0; i < g_line_count; i++) {
        if (strcmp(argv[1], g_line_registry[i].name) == 0) {
            return run_one(&g_line_registry[i]);
        }
    }

    fprintf(stderr, "查无此线: %s（用 ./lines_all list 看有哪些）\n", argv[1]);
    return 255;
}
