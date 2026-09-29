# claude-md-management — 维护 CLAUDE.md

## 作用
CLAUDE.md 是 Claude Code 的项目指令文件，决定 Claude 在本目录下的行为。
这个插件帮你：
- 分析现有 CLAUDE.md 的完整性
- 建议补充缺失的规则
- 优化结构让 Claude 更容易理解
- 随项目演进更新

## 演示场景
本目录有一份简陋的 CLAUDE.md，让插件改进它。

## 怎么玩

```bash
cd /home/magene/runtime/ai/ai-learn/ai-test/07-claude-md-management
claude
```

### 命令 1：分析当前 CLAUDE.md

```
/claude-md analyze
```

### 命令 2：自动改进

```
/claude-md improve
```

### 命令 3：从零生成

```
/claude-md generate
```

## 预期输出
- 指出缺失项（测试规范、构建命令、风格指南等）
- 建议补充的规则
- 重写后的 CLAUDE.md（带 diff 预览）

## 什么时候用
- 新项目初始化后
- Claude 总是做出不符合预期的事（说明 CLAUDE.md 没写清楚）
- 项目结构/技术栈发生变化时
