#!/bin/bash
# 一键编译脚本
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

CFLAGS="-O2 -g -Wall -Wno-unused-result -march=native"
SRC_FILES="c/matrix.c c/activations.c c/layernorm.c c/attention.c c/model.c c/autodiff.c c/optimizer.c c/dataloader.c c/main.c"

echo "Compiling..."
gcc $CFLAGS $SRC_FILES -o build/transformer -lm -lpthread

echo "Build successful: build/transformer"
ls -la build/transformer

echo ""
echo "运行模式:"
echo "  ./build/transformer train     # 训练"
echo "  ./build/transformer generate  # 推理"
echo "  ./build/transformer overfit   # 过拟合测试"
echo "  ./build/transformer demo      # 教学演示"
echo ""
echo "单元测试: bash test/build_tests.sh && bash test/run_all.sh"