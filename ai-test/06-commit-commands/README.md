# commit-commands — git commit/push/PR 一条龙

## 作用
自动分析 git diff，生成规范的 commit message，
可选直接 push 或开 PR。

## 演示场景
本目录已初始化 git，并改了个文件。演示一键 commit。

## 怎么玩

```bash
cd /home/magene/runtime/ai/ai-learn/ai-test/06-commit-commands
./setup.sh        # 初始化 git + 做点改动
claude
```

### 命令 1：只看建议（不真提交）

```
/commit --dry-run
```

### 命令 2：直接提交

```
/commit
```

### 命令 3：push 到远程

```
/commit --push
```

### 命令 4：开 PR（需要有远程仓库）

```
/commit --pr
```

## 预期输出
- 自动分析 `git diff`
- 生成 Conventional Commits 风格的 message（feat/fix/chore...）
- 显示预览让你确认
- 确认后执行 `git add` + `git commit`

## 适用场景
- 写完代码懒得想 commit message
- 想让 message 规范统一
- CI 要求 message 格式
