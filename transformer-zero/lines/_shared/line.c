/* =============================================================================
 * line.c - 学习线公共 harness 实现
 * =============================================================================
 */

#include "line.h"

void line_ctx_init(LineCtx* ctx, const char* name) {
    ctx->name = name;
    ctx->quiet = (getenv("LINE_QUIET") != NULL && strcmp(getenv("LINE_QUIET"), "1") == 0);
    ctx->hop = 0;
    ctx->pass = ctx->fail = ctx->total = 0;
}

void line_hop(LineCtx* ctx, const char* fmt, ...) {
    if (ctx->quiet) return;
    ctx->hop++;
    printf("\n");
    printf(LN_YELLOW "─── 跳点 %d ─── " LN_RESET, ctx->hop);
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n" LN_DIM "────────────────────────────────────────────" LN_RESET "\n");
}

void line_note(LineCtx* ctx, const char* fmt, ...) {
    if (ctx->quiet) return;
    printf("  ");
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
}

void line_vec(LineCtx* ctx, const char* name, const float* v, int n) {
    if (ctx->quiet) return;
    printf("  " LN_CYAN "%s" LN_RESET " [%d]: ", name, n);
    int show = (n > 16) ? 16 : n;
    for (int i = 0; i < show; i++) printf("%.4f ", v[i]);
    if (n > 16) printf("...");
    printf("\n");
}

void line_ivec(LineCtx* ctx, const char* name, const int* v, int n) {
    if (ctx->quiet) return;
    printf("  " LN_CYAN "%s" LN_RESET " [%d]: ", name, n);
    int show = (n > 16) ? 16 : n;
    for (int i = 0; i < show; i++) printf("%d ", v[i]);
    if (n > 16) printf("...");
    printf("\n");
}

void line_mat(LineCtx* ctx, const char* name, const float* m, int rows, int cols) {
    if (ctx->quiet) return;
    printf("  " LN_CYAN "%s" LN_RESET " (%d×%d):\n", name, rows, cols);
    int show_r = (rows > 8) ? 8 : rows;
    int show_c = (cols > 8) ? 8 : cols;
    for (int i = 0; i < show_r; i++) {
        printf("    ");
        for (int j = 0; j < show_c; j++) printf("%8.4f ", m[i * cols + j]);
        if (cols > 8) printf(" ...");
        printf("\n");
    }
    if (rows > 8) printf("    ...\n");
}

void line_attn_mat(LineCtx* ctx, const char* name, const float* att,
                   int T, const char* const* row_labels, const char* const* col_labels) {
    if (ctx->quiet) return;
    printf("  " LN_CYAN "%s" LN_RESET " (%d×%d, 行=查询, 列=被关注):\n", name, T, T);
    printf("          ");
    for (int j = 0; j < T; j++) printf("%8s ", col_labels[j]);
    printf("\n");
    for (int i = 0; i < T; i++) {
        printf("  %6s  ", row_labels[i]);
        for (int j = 0; j < T; j++) {
            float w = att[i * T + j];
            /* 权重越大颜色越亮，一眼看出"谁在看谁" */
            if (w > 0.5f)      printf(LN_GREEN "%8.4f " LN_RESET, w);
            else if (w > 0.2f) printf(LN_CYAN "%8.4f " LN_RESET, w);
            else               printf(LN_DIM  "%8.4f " LN_RESET, w);
        }
        printf("\n");
    }
}

void line_check_close(LineCtx* ctx, float actual, float expected, float tol, const char* msg) {
    ctx->total++;
    float d = fabsf(actual - expected);
    if (d < tol) {
        ctx->pass++;
        if (!ctx->quiet)
            printf("  " LN_GREEN "✓" LN_RESET " %s: %.4f ≈ %.4f\n", msg, actual, expected);
    } else {
        ctx->fail++;
        /* 安静模式下这是唯一会落屏的行 —— 专为批量跑设计 */
        printf(LN_RED "✗ [线 %s] %s: %.4f ≠ %.4f (diff=%.6f)" LN_RESET "\n",
               ctx->name, msg, actual, expected, d);
    }
}

void line_check_true(LineCtx* ctx, int cond, const char* msg) {
    ctx->total++;
    if (cond) {
        ctx->pass++;
        if (!ctx->quiet) printf("  " LN_GREEN "✓" LN_RESET " %s\n", msg);
    } else {
        ctx->fail++;
        printf(LN_RED "✗ [线 %s] %s (条件不满足)" LN_RESET "\n", ctx->name, msg);
    }
}

int line_summary(const LineCtx* ctx) {
    if (ctx->quiet) return ctx->fail;
    printf("\n" LN_DIM "────────────────────────────────────────────" LN_RESET "\n");
    if (ctx->fail == 0) {
        printf(LN_GREEN LN_BOLD "✅ 线 %s 全部通过: %d/%d" LN_RESET "\n", ctx->name, ctx->pass, ctx->total);
    } else {
        printf(LN_RED LN_BOLD "❌ 线 %s 失败: %d/%d" LN_RESET " (通过 %d)\n",
               ctx->name, ctx->fail, ctx->total, ctx->pass);
    }
    return ctx->fail;
}
