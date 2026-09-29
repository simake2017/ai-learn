# Claude Code + claude-mem 架构与使用指南

> 基于 claude-mem v13.15.0，2026-08-12 实测

## 一、整体架构

Claude Code 是 Anthropic 的 CLI 编码助手。claude-mem 是第三方插件，为 Claude Code 提供**跨会话持久记忆**能力。

两者通过三种机制协作：

```
┌─────────────────────────────────────────────────────────┐
│                     Claude Code CLI                      │
│  (Node.js 进程，运行 LLM 对话 + 工具调用)                  │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐     │
│  │ MCP 协议  │   │  Hooks   │   │ CLAUDE.md 注入    │     │
│  │ (工具调用) │   │ (事件钩子)│   │ (system prompt)  │     │
│  └────┬─────┘   └────┬─────┘   └──────────────────┘     │
└───────┼──────────────┼──────────────────────────────────┘
        │ stdio        │ spawn 子进程
        ▼              ▼
┌─────────────────────────────────────────────────────────┐
│                    claude-mem 插件                        │
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │           MCP Server（每会话 1 对）               │      │
│  │  node wrapper → mcp-server.cjs                 │      │
│  │  提供 search / timeline / corpus 等工具          │      │
│  └──────────────────┬─────────────────────────────┘      │
│                     │                                    │
│  ┌──────────────────▼─────────────────────────────┐      │
│  │         bun daemon（全局 1 个，共享）             │      │
│  │  worker-service.cjs --daemon                   │      │
│  │  ├─ SQLite 存储（claude-mem.db）                │      │
│  │  ├─ Chroma 向量库（chroma/）                     │      │
│  │  └─ supervisor.json 进程管理                    │      │
│  └────────────────────────────────────────────────┘      │
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │         Hook 进程（每次触发 1 个，短命）           │      │
│  │  worker-service.cjs hook claude-code <action>  │      │
│  │  observation / file-context / summarize ...    │      │
│  └────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

## 二、三种协作机制详解

### 2.1 MCP（Model Context Protocol）— 工具调用

Claude Code 通过 MCP 协议调用 claude-mem 提供的工具，**像调用内置工具一样使用记忆功能**。

**注册方式**：插件的 `.mcp.json` 声明 MCP server，Claude Code 启动时自动注册。

```
~/.claude/plugins/cache/thedotmack/claude-mem/13.15.0/.mcp.json
```

**进程模型**：每个 Claude 会话启动时 spawn 一对进程：

```
Claude Code ←stdio→ node -e "wrapper脚本" ←spawn→ mcp-server.cjs
                      (PID A)                  (PID B)
```

- `node -e` 是 wrapper，负责查找正确版本的 mcp-server.cjs
- `mcp-server.cjs` 是实际 MCP server，处理工具请求
- 通过 stdio（stdin/stdout）与 Claude Code 通信
- 会话结束时自动退出

**提供的 MCP 工具**：

| 工具 | 功能 | 典型用途 |
|------|------|---------|
| `search` | 语义搜索记忆 | 查找过去的决策、bug 修复记录 |
| `timeline` | 按时间线查看记忆上下文 | 查看某个决策的前后事件 |
| `get_observations` | 按 ID 获取记忆详情 | 查看特定记录全文 |
| `build_corpus` | 构建知识语料库 | 筛选特定主题的记忆集合 |
| `prime_corpus` | 加载语料库到会话 | 让 LLM 具备特定领域知识 |
| `query_corpus` | 查询已加载的语料库 | 向语料库提问 |
| `smart_outline` | 文件结构大纲 | 快速了解文件符号结构 |
| `smart_search` | AST 级代码搜索 | 按符号名搜索代码 |
| `smart_unfold` | 展开特定符号源码 | 读取函数/类的完整实现 |
| `session_start_context` | 渲染会话启动上下文 | 注入项目历史到 prompt |

### 2.2 Hooks — 事件钩子

Claude Code 在特定生命周期事件时执行 hook 命令，claude-mem 借此**自动记录和分析**用户行为。

**注册方式**：插件的 `hooks/hooks.json`，Claude Code 启动时读取。

**当前注册的 hooks**：

| 事件 | matcher | async | 动作 | 说明 |
|------|---------|-------|------|------|
| `Setup` | `*` | 同步 | `version-check.js` | 检查版本、安装依赖 |
| `SessionStart` | `startup\|clear\|compact` | 异步 | `start` | 启动/连接 daemon |
| `SessionStart` | `startup\|clear\|compact` | 异步 | `context` | 注入项目记忆到 prompt |
| `UserPromptSubmit` | `*` | 异步 | `session-init` | 会话初始化 |
| `PostToolUse` | `*` | 异步 | `observation` | **每次工具调用后**自动记录观察 |
| `PreToolUse` | `Read` | 异步 | `file-context` | 读文件前注入上下文 |
| `Stop` | `*` | 异步 | `summarize` | 会话结束时总结 |

**进程模型**：每次 hook 触发 spawn 独立进程链：

```
Claude Code
  → /bin/sh -c "复杂的路径查找脚本"
    → node bun-runner.js worker-service.cjs hook claude-code <action>
      → bun worker-service.cjs hook claude-code <action>
```

这意味着：
- 调用 1 次 `Bash("ls")` → 触发 PostToolUse → spawn 3 个进程
- 调用 1 次 `Read("foo.java")` → 触发 PreToolUse + PostToolUse → spawn 6 个进程
- 这些进程理论上执行完就退出，但实测存在泄漏（见故障排查）

### 2.3 CLAUDE.md — 记忆文件注入

claude-mem 管理的记忆文件通过 CLAUDE.md 机制注入到 Claude 的 system prompt 中。

```
~/.claude/projects/<project-hash>/memory/MEMORY.md    ← 索引文件
~/.claude/projects/<project-hash>/memory/*.md          ← 具体记忆文件
```

MEMORY.md 包含一行一条记忆的摘要，Claude 启动时自动读取。需要详情时通过 MCP 工具查询。

## 三、Daemon 进程管理

### 3.1 自管理模式（默认）

```
Claude 启动
  └─ SessionStart hook（异步）
       └─ node bun-runner.js worker-service.cjs start
            └─ bun worker-service.cjs --daemon     ← 全局唯一
                 ├─ SQLite（claude-mem.db，324KB）
                 ├─ Chroma 向量库（chroma/，192KB）
                 ├─ supervisor.json（进程注册表）
                 └─ spawn.lock（防重复启动锁）
```

- **第一个**打开的 Claude 会话启动 daemon
- 后续会话检测到 daemon 已运行，直接连接
- `spawn.lock` 防止并发启动
- `supervisor.json` 记录 daemon 和 chroma-mcp 的 PID

### 3.2 systemd 模式（可选，当前未启用）

```ini
# /etc/systemd/system/claude-mem.service
[Service]
Type=simple
ExecStart=/root/.bun/bin/bun .../worker-service.cjs --daemon
Restart=on-failure
```

当前状态：`inactive (dead)` + `disabled`。启动失败原因是 `Type=simple` 与 daemon 的 fork 行为不兼容（`protocol` 错误）。

**不需要 systemd**，自管理模式已够用。

## 四、数据存储

```
~/.claude-mem/
├── claude-mem.db          # SQLite 主数据库（observations、sessions）
├── claude-mem.db-shm      # WAL 共享内存
├── claude-mem.db-wal      # WAL 日志
├── chroma/                # ChromaDB 向量存储（语义搜索用）
├── corpora/               # 知识语料库缓存
├── logs/                  # 运行日志
│   └── claude-mem-YYYY-MM-DD.log
├── state/                 # 状态文件
├── supervisor.json        # daemon 进程注册表
├── spawn.lock             # 启动锁
├── settings.json          # 插件配置
└── backfill.json          # 回填配置
```

### 记忆数据模型

```
Observation（观察记录）
├── id          唯一标识
├── type        bugfix | feature | decision | refactor | discovery | change
├── content     记忆内容
├── project     所属项目
├── platform    来源（claude / codex / cursor）
├── concepts    关联概念标签
├── files       关联文件路径
└── created_at  创建时间
```

## 五、配置文件

### 5.1 全局设置

```
~/.claude/settings.json
├── env              # 环境变量（API key、模型、超时）
├── permissions      # 工具权限白名单
├── model            # 默认模型
├── enabledPlugins   # 启用的插件列表
├── mcpServers       # 额外 MCP server（非插件的）
└── extraKnownMarketplaces  # 第三方插件源
```

### 5.2 项目设置

```
~/.claude/projects/<project-hash>/
├── CLAUDE.md          # 项目级 system prompt
├── memory/            # claude-mem 记忆文件
│   ├── MEMORY.md      # 记忆索引
│   └── *.md           # 具体记忆
└── settings.json      # 项目级设置（可覆盖全局）
```

`<project-hash>` 格式：`-` + 项目路径，`/` 替换为 `-`。
例：`/home/magene/runtime/ai/mini-qmd` → `-home-magene-runtime-ai-mini-qmd`

## 六、性能特征

### 6.1 资源占用（实测）

| 组件 | RSS 内存 | CPU | 数量 |
|------|---------|-----|------|
| bun daemon | ~110MB | <1% | 1 |
| chroma-mcp (Python) | ~210MB | <1% | 1 |
| MCP server (Node) | ~65MB | ~0 | 每会话 |
| node wrapper | ~40MB | ~0 | 每会话 |
| hook bun 进程 | ~100MB | 短暂 | 每次触发 |
| **总计（1 个会话）** | **~525MB** | | |

### 6.2 启动耗时（优化后）

| 阶段 | 耗时 | 说明 |
|------|------|------|
| Setup (version-check) | ~55ms | 依赖已装好 |
| SessionStart (start) | **异步** | 改为 async，不阻塞启动 |
| SessionStart (context) | 异步 | 后台注入记忆 |
| MCP server 启动 | ~100ms | node spawn |
| **总阻塞时间** | **~200ms** | |

优化前 SessionStart start 是同步阻塞 35 秒。

### 6.3 运行时开销

每次工具调用的 hook 链：

```
工具调用 → shell 路径解析(~5ms) → node bun-runner(~100ms) → bun hook(~500ms-2s)
```

PostToolUse observation 最重，因为要分析工具输出并决定是否值得记录。

## 七、故障排查

### 7.1 hook 进程泄漏

**症状**：`ps` 看到大量 `worker-service.cjs hook claude-code observation` 进程。

**原因**：async hook 理论上执行完退出，但 bun 进程有时不退出。

**修复**：
```bash
# 清理泄漏进程
ps aux | grep "worker-service.cjs hook" | grep -v grep | awk '{print $2}' | xargs kill -9
```

### 7.2 启动慢

**症状**：Claude 启动后等 30+ 秒才能交互。

**原因**：SessionStart hook 的 `worker-service.cjs start` 是同步阻塞的。

**修复**：编辑 `hooks.json`，给第一个 SessionStart hook 加 `"async": true`：
```bash
# 文件位置
~/.claude/plugins/cache/thedotmack/claude-mem/13.15.0/hooks/hooks.json

# SessionStart → hooks[0] 加上 "async": true
```

⚠️ 插件升级会覆盖此修改，需重新应用。

### 7.3 MCP server 路径错误

**症状**：启动时报 `Cannot access directory /Users/xxx`。

**原因**：`settings.json` 的 `mcpServers` 引用了其他机器的路径。

**修复**：删除或修正 `~/.claude/settings.json` 中的 `mcpServers` 配置。

### 7.4 daemon 未启动

**症状**：MCP 工具调用返回错误或超时。

**诊断**：
```bash
# 检查 daemon
ps aux | grep "worker-service.cjs --daemon" | grep -v grep

# 检查 supervisor
cat ~/.claude-mem/supervisor.json

# 手动启动
node ~/.claude/plugins/cache/thedotmack/claude-mem/13.15.0/scripts/bun-runner.js \
     ~/.claude/plugins/cache/thedotmack/claude-mem/13.15.0/scripts/worker-service.cjs start
```

## 八、使用方式

### 8.1 自动记忆（无需干预）

claude-mem 的 PostToolUse hook 自动观察你的操作，将有价值的信息记录为 observation。你不需要做任何事。

### 8.2 手动查询记忆

在 Claude 对话中直接使用 MCP 工具：

```
# 搜索记忆
→ 调用 mcp__plugin_claude-mem_mcp-search__search(query="如何配置 React hooks")

# 查看时间线
→ 调用 mcp__plugin_claude-mem_mcp-search__timeline(query="分块算法")

# 构建知识语料库
→ 调用 mcp__plugin_claude-mem_mcp-search__build_corpus(name="rag-pipeline", query="RAG search embedding")
→ 调用 mcp__plugin_claude-mem_mcp-search__prime_corpus(name="rag-pipeline")
→ 调用 mcp__plugin_claude-mem_mcp-search__query_corpus(name="rag-pipeline", question="RRF 融合算法是什么")
```

### 8.3 记忆文件管理

记忆以 Markdown 文件形式存储在 `~/.claude/projects/<project-hash>/memory/`：

```bash
# 查看当前项目的所有记忆
ls ~/.claude/projects/-home-magene-runtime-ai-mini-qmd/memory/

# 查看记忆索引
cat ~/.claude/projects/-home-magene-runtime-ai-mini-qmd/memory/MEMORY.md

# 查看某条记忆详情
cat ~/.claude/projects/-home-magene-runtime-ai-mini-qmd/memory/mini-qmd-tech-stack.md
```

### 8.4 代码智能工具

claude-mem 还提供 AST 级代码分析工具（基于 tree-sitter）：

```
# 查看文件结构大纲（只看符号，不读全文）
→ mcp__plugin_claude-mem_mcp-search__smart_outline(file_path="src/chunk/Chunker.java")

# 按符号名搜索
→ mcp__plugin_claude-mem_mcp-search__smart_search(query="findBestCutoff")

# 展开某个函数的完整实现
→ mcp__plugin_claude-mem_mcp-search__smart_unfold(file_path="...", symbol_name="findBestCutoff")
```

## 九、已安装插件清单

| 插件 | 来源 | 功能 |
|------|------|------|
| claude-mem | thedotmack | 跨会话持久记忆 |
| superpowers | official | 增强 Claude 能力 |
| github | official | GitHub 操作 |
| playwright | official | 浏览器自动化 |
| context7 | official | 文档查询 |
| code-review | official | 代码审查 |
| code-simplifier | official | 代码简化 |
| clangd-lsp | official | C/C++ LSP |
| gopls-lsp | official | Go LSP |
| jdtls-lsp | official | Java LSP |
| pyright-lsp | official | Python LSP |
| frontend-design | official | 前端设计 |
| skill-creator | official | 技能创建 |
| commit-commands | official | Git 提交辅助 |
| claude-md-management | official | CLAUDE.md 管理 |
