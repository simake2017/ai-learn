# frontend-design — 生成高质量前端 UI

## 作用
让 Claude 按专业设计师水准生成前端代码（React / Vue / 纯 HTML）。
输出包含完整 CSS、响应式、可访问性。

## 演示场景
生成一个"服务器监控仪表盘"页面。

## 怎么玩

```bash
cd /home/magene/runtime/ai/ai-learn/ai-test/01-frontend-design
claude
```

然后在 Claude 里输入（粘贴 try.sh 里的命令）：

```
/frontend-design 设计一个服务器监控仪表盘，包含：
- 顶部 4 个指标卡（CPU、内存、磁盘、网络）
- 中间一个折线图（过去 1h CPU 使用率）
- 下面一个表格（运行中的服务列表，含状态灯）
用 React + Tailwind，深色主题
```

## 预期输出
- 生成 `dashboard.tsx`（或类似文件）
- 包含完整组件代码，可直接 `npm run dev` 跑
