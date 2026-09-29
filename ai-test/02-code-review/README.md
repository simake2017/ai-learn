# code-review — 多 agent 自动代码审查

## 作用
派多个 agent 从不同角度（bug、性能、安全、可读性）审代码，
每个发现带置信度评分，避免误报。

## 演示场景
`buggy.go` 里有 5 个故意埋的 bug，让 code-review 找出来。

## 怎么玩

```bash
cd /home/magene/runtime/ai/ai-learn/ai-test/02-code-review
claude
```

在 Claude 里输入：

```
/code-review buggy.go
```

## 预期输出
找到以下 bug（带置信度）：
1. `Inc()` 写 count 没加锁（数据竞争）
2. `Get()` 读 count 没加锁（数据竞争）
3. 闭包捕获循环变量 `item`（所有 goroutine 共享同一个 item）
4. `os.Open` 错误没处理（nil 指针 panic）
5. 并发写 map 没加锁（fatal error: concurrent map writes）

## 进阶
可指定审查维度：

```
/code-review buggy.go --focus security
```
