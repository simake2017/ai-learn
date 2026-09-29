#!/bin/bash
# 初始化一个临时 git 仓库用于演示 commit 命令
set -e

cd "$(dirname "$0")"

# 初始化 git（如果还没初始化）
if [ ! -d .git ]; then
    git init -q
    git config user.email "demo@example.com"
    git config user.name "Demo User"
    echo "# demo" > README.md
    git add README.md
    git commit -q -m "initial commit"
    echo "✓ git 仓库初始化完成"
else
    echo "✓ git 仓库已存在"
fi

# 做点改动
cat > hello.go << 'EOF'
package main

import "fmt"

func greet(name string) string {
    if name == "" {
        return "Hello, World!"
    }
    return fmt.Sprintf("Hello, %s!", name)
}

func main() {
    fmt.Println(greet(""))
    fmt.Println(greet("Claude"))
}
EOF

echo "✓ 已创建 hello.go，可以跑 /commit 了"
echo ""
echo "下一步：claude 然后输入 /commit"
