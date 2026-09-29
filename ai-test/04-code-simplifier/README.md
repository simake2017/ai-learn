# code-simplifier — 简化重构代码

## 作用
分析代码的复杂度，给出简化建议：
- 消除冗余判断
- 用更简洁的写法
- 提取公共逻辑
- 改善命名

## 演示场景
`messy.go` 是一段能跑但又臭又长的代码，让插件重构。

## 怎么玩

```bash
cd /home/magene/runtime/ai/ai-learn/ai-test/04-code-simplifier
claude
```

```
/code-simplifier messy.go
```

## 预期输出（简化点）

| 原写法 | 简化后 |
|---|---|
| `if x == false` | `if !x` |
| `if len(s) > 0` + `if s != ""` | 只留一个 |
| `foundFlag bool = false` | 用 `for range` + 早返回 |
| 多层嵌套 if | guard clause 早 return |
| `retryCount = retryCount + 1` | `retryCount++` |
| 手动 append 到 map | 直接 `result[l] = append(...)` |

## 进阶：只看某一函数

```
/code-simplifier messy.go --function processWithRetry
```
