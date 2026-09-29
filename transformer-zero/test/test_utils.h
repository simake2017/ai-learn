/* =============================================================================
 * test_utils.h - 单元测试通用工具
 *
 * 📚 使用方法：
 *   #include "test_utils.h"
 *   TEST_BEGIN("矩阵乘法测试");
 *   ...测试代码...
 *   ASSERT_CLOSE(actual, expected, 1e-5, "matmul[0]");
 *   TEST_END();
 *
 * 🎨 输出示例：
 *   ╔══════════════════════════════════════╗
 *   ║  矩阵乘法测试                         ║
 *   ╚══════════════════════════════════════╝
 *   ✅ matmul[0]: 1.00 ≈ 1.00
 *   ❌ matmul[1]: 3.14 ≠ 2.00 (diff=1.14)
 *   ──────────────────────────────────────
 *   结果: 1/2 通过
 * =============================================================================
 */

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <stdio.h>
#include <math.h>
#include <string.h>

/* === ANSI 颜色 === */
#define CLR_GREEN  "\033[32m"
#define CLR_RED    "\033[31m"
#define CLR_YELLOW "\033[33m"
#define CLR_CYAN   "\033[36m"
#define CLR_BOLD   "\033[1m"
#define CLR_DIM    "\033[2m"
#define CLR_RESET  "\033[0m"

/* === 测试计数器（每个测试文件独立） === */
static int _test_pass = 0;
static int _test_fail = 0;
static int _test_total = 0;
static char _test_name[128] = {0};

/* === 测试开始 === */
#define TEST_BEGIN(name) \
    do { \
        strncpy(_test_name, name, sizeof(_test_name) - 1); \
        _test_pass = 0; _test_fail = 0; _test_total = 0; \
        printf("\n"); \
        printf(CLR_CYAN "╔══════════════════════════════════════════╗" CLR_RESET "\n"); \
        printf(CLR_CYAN "║  " CLR_BOLD "%-38s" CLR_CYAN "║" CLR_RESET "\n", name); \
        printf(CLR_CYAN "╚══════════════════════════════════════════╝" CLR_RESET "\n\n"); \
    } while (0)

/* === 测试结束：打印汇总 === */
#define TEST_END() \
    do { \
        printf("\n"); \
        printf(CLR_DIM "────────────────────────────────────────────" CLR_RESET "\n"); \
        if (_test_fail == 0) { \
            printf(CLR_GREEN CLR_BOLD "✅ 全部通过: %d/%d" CLR_RESET "\n", \
                   _test_pass, _test_total); \
        } else { \
            printf(CLR_RED CLR_BOLD "❌ 失败: %d/%d" CLR_RESET " (通过 %d)\n", \
                   _test_fail, _test_total, _test_pass); \
        } \
        printf("\n"); \
        return (_test_fail > 0) ? 1 : 0; \
    } while (0)

/* === 断言：浮点数接近 === */
#define ASSERT_CLOSE(actual, expected, tol, msg) \
    do { \
        _test_total++; \
        float _a = (float)(actual); \
        float _e = (float)(expected); \
        float _d = fabsf(_a - _e); \
        if (_d < (tol)) { \
            _test_pass++; \
            printf("  " CLR_GREEN "✅" CLR_RESET " %s: %.4f ≈ %.4f\n", msg, _a, _e); \
        } else { \
            _test_fail++; \
            printf("  " CLR_RED "❌" CLR_RESET " %s: %.4f ≠ %.4f " CLR_DIM "(diff=%.6f)" CLR_RESET "\n", \
                   msg, _a, _e, _d); \
        } \
    } while (0)

/* === 断言：整数相等 === */
#define ASSERT_EQ(actual, expected, msg) \
    do { \
        _test_total++; \
        int _a = (int)(actual); \
        int _e = (int)(expected); \
        if (_a == _e) { \
            _test_pass++; \
            printf("  " CLR_GREEN "✅" CLR_RESET " %s: %d == %d\n", msg, _a, _e); \
        } else { \
            _test_fail++; \
            printf("  " CLR_RED "❌" CLR_RESET " %s: %d ≠ %d\n", msg, _a, _e); \
        } \
    } while (0)

/* === 断言：条件为真 === */
#define ASSERT_TRUE(cond, msg) \
    do { \
        _test_total++; \
        if (cond) { \
            _test_pass++; \
            printf("  " CLR_GREEN "✅" CLR_RESET " %s\n", msg); \
        } else { \
            _test_fail++; \
            printf("  " CLR_RED "❌" CLR_RESET " %s (条件不满足)\n", msg); \
        } \
    } while (0)

/* === 打印向量（最多前 n 个元素） === */
static void print_vec(const char* name, const float* v, int n) {
    printf("  " CLR_CYAN "%s" CLR_RESET " [%d]: ", name, n);
    int show = (n > 16) ? 16 : n;
    for (int i = 0; i < show; i++) {
        printf("%.3f ", v[i]);
    }
    if (n > 16) printf("...");
    printf("\n");
}

/* === 打印整数向量 === */
static void print_ivec(const char* name, const int* v, int n) {
    printf("  " CLR_CYAN "%s" CLR_RESET " [%d]: ", name, n);
    int show = (n > 16) ? 16 : n;
    for (int i = 0; i < show; i++) {
        printf("%d ", v[i]);
    }
    if (n > 16) printf("...");
    printf("\n");
}

/* === 打印矩阵（行优先，最多 8×8） === */
static void print_mat(const char* name, const float* m, int rows, int cols) {
    printf("  " CLR_CYAN "%s" CLR_RESET " (%d×%d):\n", name, rows, cols);
    int show_r = (rows > 8) ? 8 : rows;
    int show_c = (cols > 8) ? 8 : cols;
    for (int i = 0; i < show_r; i++) {
        printf("    ");
        for (int j = 0; j < show_c; j++) {
            printf("%8.3f ", m[i * cols + j]);
        }
        if (cols > 8) printf(" ...");
        printf("\n");
    }
    if (rows > 8) printf("    ...\n");
}

/* === 分隔线 === */
static void print_sep(void) {
    printf("\n");
}

#endif /* TEST_UTILS_H */
