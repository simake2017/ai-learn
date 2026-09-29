#!/bin/bash
# =============================================================================
# build_tests.sh - 编译所有单元测试
#
# 用法: bash test/build_tests.sh
# 输出: build/test_XX_* 可执行文件
# =============================================================================

set -e

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

mkdir -p build

# 编译标志
CFLAGS="-O0 -g -Wall -Wno-unused-result -I."

# 源文件（排除 main.c，避免 main 符号冲突）
SRC_FILES="c/matrix.c c/activations.c c/layernorm.c c/attention.c c/model.c c/autodiff.c c/optimizer.c c/dataloader.c"

# 测试文件列表
TESTS=(
    "test_01_embedding"
    "test_02_matmul"
    "test_03_layernorm"
    "test_04_qkv"
    "test_05_attention"
    "test_06_gelu"
    "test_07_softmax"
    "test_08_crossentropy"
    "test_09_residual"
    "test_10_forward"
    "test_11_gradient"
    "test_12_adamw"
)

echo "🔨 编译单元测试..."
echo ""

PASS=0
FAIL=0

for test in "${TESTS[@]}"; do
    printf "  %-30s " "$test"
    if gcc $CFLAGS "test/${test}.c" $SRC_FILES -o "build/${test}" -lm 2>/dev/null; then
        echo "✅"
        PASS=$((PASS + 1))
    else
        echo "❌"
        FAIL=$((FAIL + 1))
        # 显示编译错误
        gcc $CFLAGS "test/${test}.c" $SRC_FILES -o "build/${test}" -lm 2>&1 | head -5
    fi
done

echo ""
echo "──────────────────────────────────────"
echo "编译结果: ${PASS} 成功, ${FAIL} 失败 (共 ${#TESTS[@]})"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "✅ 全部编译通过！运行测试: bash test/run_all.sh"
else
    echo "❌ 有编译错误，请检查上面的输出"
    exit 1
fi
