# Claude Code Best (CCB) 使用教程

> 基于 CCB V5，官方仓库：https://github.com/claude-code-best/claude-code

---

## 一、CCB 是什么

CCB 是 Claude Code 的开源逆向还原版本，用 Bun/TypeScript 重写。
核心区别：**可以接任意模型 API**（OpenAI、Gemini、Grok、中转站等），不绑定 Anthropic。

> **关于 Claude Code 源码**：官方 Claude Code 不是真正的开源产品。GitHub 仓库 ([anthropics/claude-code](https://github.com/anthropics/claude-code)) 只有文档和命令模板，没有 CLI 核心源码。2026年3月，有人发现 `@anthropic-ai/claude-agent-sdk` 包里附带了约 13,800 行**压缩混淆**的 JS 代码，被社区提取并传播。CCB 大概率是基于这份泄露代码逆向还原的。

---

## 二、安装与启动

```bash
# 克隆仓库
git clone https://github.com/claude-code-best/claude-code.git
cd claude-code

# 安装依赖
bun install

# 启动
bun run start
```

首次启动会进入 `/login` 流程，配置 API 端点和密钥。

---

## 三、配置模型

启动后输入 `/login`，选择模型提供商：

| 提供商 | 说明 |
|--------|------|
| Anthropic | 官方 API，需 Anthropic Key |
| OpenAI | GPT-4o 等 |
| Gemini | Google Gemini |
| Grok | xAI Grok |
| Custom | 任意 OpenAI 兼容 API（中转站、自建） |

**示例：配置中转站**
```
Provider: Custom
Base URL: https://your-relay.com/v1
API Key: sk-xxxx
Model: claude-sonnet-4-20250514
```

---

## 四、核心命令速查

| 命令 | 作用 |
|------|------|
| `/login` | 配置模型和 API |
| `/goal` | 设定跨轮次目标，自动推进 |
| `/fast` | 切换快速模式 |
| `/model` | 切换模型 |
| `/compact` | 压缩上下文 |
| `/clear` | 清空会话 |
| `/help` | 帮助 |

---

## 五、特色功能详解

### 5.1 `/goal` — 目标驱动执行

适合复杂的多步骤任务，设定目标后 CCB 自动推进，支持暂停/恢复。

```
你: /goal 重构 src/auth 模块，把所有 JWT 验证提取到中间件

CCB: [Goal] 重构 src/auth 模块
     步骤 1/5: 分析现有 JWT 逻辑...
     步骤 2/5: 创建 auth middleware...
     ...
     [已暂停] token 预算已用 80%，继续？(y/n)
```

**要点：**
- 自动跨轮次推进，不用反复手动催
- 可设 token 预算，防止失控
- 暂停后可审查中间结果再继续

### 5.2 多智能体编排（Ultracode）

大任务拆给多个 agent 并行执行，带监控面板。

```
你: 用 ultracode 同时审计这 5 个微服务的安全漏洞

CCB: [Ultracode] 启动 5 个 agent
     ├─ agent-1: user-service ✓ (2 个漏洞)
     ├─ agent-2: order-service ✓ (0 个漏洞)
     ├─ agent-3: payment-service ⟳ 进行中...
     ├─ agent-4: notification-service ✓ (1 个漏洞)
     └─ agent-5: inventory-service ✓ (0 个漏洞)
```

### 5.3 Artifacts — 生成可分享的内容

生成 HTML 页面、图表、仪表盘，自动上传到公开 URL。

```
你: 画一个项目架构图，用 mermaid

CCB: [Artifact] 已生成
     URL: https://artifacts.ccb.dev/abc123 (24h 后过期)
```

### 5.4 Channels — 消息推送

把外部消息推送到当前 CCB 会话，适合异步工作。

```bash
# 配置飞书 webhook
# 之后飞书群消息可以推送到 CCB 会话中
```

支持：飞书、Slack、Discord、微信

### 5.5 Poor Mode — 低资源模式

网络差或机器性能低时启用，减少并发请求：

```
/poor
```

会关闭 memory 提取、建议等后台功能，集中资源给主任务。

### 5.6 Computer Use / Chrome Use

```
你: 打开浏览器访问 github.com，搜索 claude-code-best

CCB: [Chrome Use] 已打开浏览器...
     正在导航到 github.com...
     搜索 "claude-code-best"...
     找到 1 个结果: claude-code-best/claude-code ⭐ 2.3k
```

### 5.7 远程控制（Docker）

```bash
# 启动远程服务
docker compose up -d

# 手机浏览器访问 http://your-server:3000
# 即可远程使用 CCB
```

---

## 六、使用示例

### 示例 1：日常开发 — 快速修 bug

```
你: src/utils/parser.ts 第 42 行解析 JSON 时没处理 null 值，会 crash

CCB: 读取文件 → 定位问题 → 修复

你: 跑一下测试

CCB: bun test src/utils/parser.test.ts
     ✓ 12 passed, 0 failed
```

### 示例 2：`/goal` 做项目重构

```
你: /goal 把项目的 REST API 全部加上 OpenAPI 文档，覆盖 src/routes 下所有路由

CCB:
  [Goal] 生成 OpenAPI 文档
  ├── 扫描 src/routes/ (12 个路由文件)
  ├── 分析每个路由的 handler、参数、返回值
  ├── 生成 openapi.yaml
  ├── 添加 swagger-ui 中间件
  └── 启动服务验证 /docs 可访问

  进度: 3/5 完成，当前: 生成 openapi.yaml...
```

### 示例 3：多智能体并行审计

```
你: 用 ultracode 审查这个 PR 的代码质量、安全性、性能三个方面

CCB:
  [Ultracode] 启动 3 个 agent 并行审查
  ├─ 代码质量: 发现 2 个命名问题，1 个重复代码
  ├─ 安全性: 发现 SQL 注入风险 (严重)
  └─ 性能: N+1 查询问题

  汇总报告: ...
```

### 示例 4：生成可视化报告

```
你: 分析这个项目的代码行数、语言分布、最近 30 天提交趋势，生成仪表盘

CCB:
  [分析中]
  - 代码行数: 45,231 行
  - 语言: TypeScript 62%, Python 28%, Go 10%
  - 提交: 127 次 (30天)

  [Artifact] 仪表盘已生成
  URL: https://artifacts.ccb.dev/xyz789
```

### 示例 5：浏览器自动化

```
你: 打开 Jira，把 PROJ-123 的状态改成"待测试"，然后截图给我

CCB:
  [Chrome Use] 打开 Jira...
  导航到 PROJ-123...
  修改状态 → 待测试 ✓
  截图已保存: /tmp/jira-proj-123.png
```

### 示例 6：跨模型对比

```
你: /login  (切换到 GPT-4o)
你: 解释 Go 的 goroutine 调度模型

你: /login  (切回 Claude)
你: 同样解释 Go 的 goroutine 调度模型

# 对比两个模型的回答质量和准确性
```

---

## 七、与官方 Claude Code 对比

| 场景 | 推荐 |
|------|------|
| 需要最强推理能力 | 官方 Claude Code (Opus 4.8) |
| 省钱/接便宜中转 | CCB |
| 需要 Goal 跨轮推进 | CCB |
| 需要多智能体编排 | CCB |
| 需要浏览器/桌面控制 | CCB |
| 团队协作/远程控制 | CCB |
| 稳定可靠，少踩坑 | 官方 Claude Code |

---

## 八、常见问题

**Q: CCB 响应慢？**
A: 大概率是中转延迟。检查 Base URL 指向的服务，直连官方 API 最快。

**Q: 工具调用行为跟官方不一样？**
A: CCB 逆向的 prompt 和工具调度可能不完整，复杂场景（大量文件编辑、长上下文）可能有差异。

**Q: 安全吗？**
A: CCB 是开源项目可审计，但你把 API Key 交给了它和中转服务。敏感项目建议用官方。

**Q: 支持 MCP 吗？**
A: 支持，兼容 Claude Code 的 MCP 配置。
