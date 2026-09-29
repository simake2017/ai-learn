#!/bin/bash
# 06-commit-commands 演示命令
# 先跑 setup.sh 初始化 git + 创建 hello.go

# 1. 只看建议不真提交（推荐先试这个）
/commit --dry-run

# 2. 真提交
/commit

# 3. 提交并 push（需要远程仓库）
/commit --push

# 4. 开 PR（需要 GitHub 远程）
/commit --pr
