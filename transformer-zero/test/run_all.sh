#!/bin/bash
# =============================================================================
# run_all.sh - 运行所有单元测试
#
# 用法: bash test/run_all.sh
# 前提: 先运行 bash test/build_tests.sh
# =============================================================================

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

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

PASS=0
FAIL=0
SKIP=0

echo ""
echo "🧪 运行全部单元测试"
echo "════════════════════════════════════════════"

for test in "${TESTS[@]}"; do
    if [ ! -f "build/${test}" ]; then
        echo ""
        echo "⚠️  跳过 ${test} (未编译)"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo ""
    if "build/${test}" 2>&1; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "════════════════════════════════════════════"
echo "📊 总计: ${#TESTS[@]} 个测试"
echo "   ✅ 通过: ${PASS}"
echo "   ❌ 失败: ${FAIL}"
if [ $SKIP -gt 0 ]; then
    echo "   ⚠️  跳过: ${SKIP}"
fi
echo "════════════════════════════════════════════"
echo ""

if [ $FAIL -gt 0 ]; then
    exit 1
fi
