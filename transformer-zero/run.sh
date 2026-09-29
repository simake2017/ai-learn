#!/bin/bash
# 一键训练 + 推理脚本
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# 编译
bash build.sh

# 训练
echo ""
echo "=== 训练 ==="
./build/transformer train

# 推理
echo ""
echo "=== 推理 ==="
./build/transformer generate