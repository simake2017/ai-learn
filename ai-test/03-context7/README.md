# context7 — 查最新版 API 文档

## 作用
Claude 训练数据有截止日期，可能用过时的 API。
context7 实时拉取库的最新官方文档，确保生成的代码用当前推荐写法。

## 演示场景
查 Go 标准库 `net/http` 和 etcd clientv3 的最新用法。

## 怎么玩

```bash
cd /home/magene/runtime/ai/ai-learn/ai-test/03-context7
claude
```

### 例子 1：查 React 最新写法

```
/context7 react useeffect cleanup
```

预期：返回 React 18/19 的 useEffect cleanup 官方写法，而不是旧版的 componentWillUnmount 思路。

### 例子 2：查 Go 新特性

```
/context7 go 1.22 range over integers
```

预期：展示 `for i := range 10` 新语法，而不是旧版 `for i := 0; i < 10; i++`。

### 例子 3：查 etcd clientv3

```
/context7 etcd clientv3 watch example
```

预期：最新 etcd client-go 的 watch API 示例。

## 什么时候用
- 写不熟悉的库
- 升级老代码（查"现在推荐写法是什么"）
- 怀疑 Claude 给的 API 是旧版
