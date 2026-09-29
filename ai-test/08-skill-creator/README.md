# skill-creator — 创建/测试 Skill

## 作用
Skill 是 Claude Code 的可复用能力模块（比 CLAUDE.md 更结构化）。
这个插件帮你：
- 从零创建新 Skill（带 agents、references 目录结构）
- 改进已有 Skill
- 跑评估集测试 Skill 性能

## 什么是 Skill
Skill = 一段可触发的"专家指令包"，包含：
- SKILL.md — 主指令
- agents/ — 子 agent 角色定义
- references/ — 参考资料
- 用斜杠命令触发（如 `/my-skill`）

## 演示场景
创建一个 "go-test-writer" skill，专门给 Go 代码写 table-driven 测试。

## 怎么玩

```bash
cd /home/magene/runtime/ai/ai-learn/ai-test/08-skill-creator
claude
```

### 命令 1：从零创建 Skill

```
/skill-creator 创建一个 go-test-writer 技能，功能：
- 输入一个 Go 文件路径
- 自动生成 table-driven 测试文件
- 覆盖正常、边界、错误场景
- 测试命名符合 TestXxx_Yyy 规范
```

### 命令 2：改进已有 Skill

```
/skill-creator improve go-test-writer
```

### 命令 3：跑评估测试

```
/skill-creator eval go-test-writer --cases 5
```

## 预期输出
```
.claude/skills/go-test-writer/
├── SKILL.md           # 主指令
├── agents/
│   ├── analyzer.md    # 分析源码的 agent
│   └── writer.md      # 写测试的 agent
└── references/
    └── patterns.md    # table-driven 模式参考
```

## 适用场景
- 经常做同类任务（写测试、生成 mock、迁移代码）
- 想把某个"专家流程"固化下来给团队复用
- 想用 eval 量化 Skill 效果
