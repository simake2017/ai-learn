# feature-dev — 多 agent 协作开发功能

## 作用
开发一个完整功能的工作流，多 agent 分工：
- 🎯 需求分析 agent — 拆解需求
- 🏗️ 架构设计 agent — 设计方案
- 💻 实现 agent — 写代码
- 🧪 测试 agent — 写单测
- 🔍 审查 agent — 审代码

## 演示场景
在当前目录开发一个小功能：**"Go 实现的 LRU Cache，带过期时间"**。

## 怎么玩

```bash
cd /home/magene/runtime/ai/ai-learn/ai-test/05-feature-dev
claude
```

```
/feature-dev 实现一个 LRU Cache，要求：
- 容量可配置
- 每个 key 可设过期时间（TTL），过期自动失效
- 线程安全
- Get/Put/Delete 操作
- 用 table-driven 单测覆盖
```

## 预期输出
- `lru.go` — LRU Cache 实现
- `lru_test.go` — 单测
- 可能还有 `doc.go` 或 README

## 跟普通开发比有什么不同
普通对话：Claude 一口气写出来。
feature-dev：**多个 agent 接力**，需求→设计→实现→测试→审查，质量更高。

## 适用场景
- 完整的功能模块（不是一个函数）
- 需要设计+实现+测试全套
- 想让 Claude 自己审自己
