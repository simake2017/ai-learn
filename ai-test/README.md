# Claude Code 插件演示

每个子目录演示一个插件，按编号顺序玩。

## 玩法

```bash
cd 到对应子目录
claude   # 在该目录下启动 Claude Code
# 然后在 Claude 里粘贴 try.sh 里的斜杠命令
```

## 插件清单

| # | 插件 | 一句话 |
|---|---|---|
| 01 | frontend-design | 生成前端 UI 代码 |
| 02 | code-review | 多 agent 自动审代码 |
| 03 | context7 | 查最新版 API 文档 |
| 04 | code-simplifier | 简化重构代码 |
| 05 | feature-dev | 多 agent 协作开发功能 |
| 06 | commit-commands | git commit/push/PR 一条龙 |
| 07 | claude-md-management | 维护 CLAUDE.md |
| 08 | skill-creator | 创建/测试 Skill |

## 注意

- 每个演示独立，互不影响
- 斜杠命令输入后会提示确认，回车即可
- 部分插件（code-review、feature-dev）会跑多 agent，耗时较长
