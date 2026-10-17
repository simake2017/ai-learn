# 学习线（lines/）

> 一条线 = 一个端到端调用链，**是可运行的业务代码 + 逐跳日志 + 轻量校验**，
> 不是单元测试。目标是自己能读、能单步、能改一行看效果。

## 一、为什么要「线」

`c/` 是按**机制类别**横着分层的（`matrix.c` `layernorm.c` `attention.c` `model.c` …），
而一条真实的数据流是**竖着穿过**这些层的：

```text
x ──LN──▶ ln1 ──matmul(W_QKV)──▶ qkv ──attention──▶ att_out ──matmul(W_proj)──▶ ──+x──▶ 残差
└────────────── block_forward（model.c）里的真实接线 ──────────────────────────────────┘
```

**这条竖线不属于任何一个文件** —— `attention.c` 不知道自己是被 LN 的输出喂进来的，
`layernorm.c` 也不知道自己的输出会被切成 Q/K/V。学习时最难啃的恰恰是它，
`lines/` 就是把这些竖线一根根抽出来单跑：每一「跳」对应数据穿过的一次真算子，
跳与跳之间打印数据本身，结尾用 check 锚定关键数值。

## 二、铁律（继承自 myos/lines）

1. **lines/ 里不出现任何模型逻辑。** 线里只准调用 `c/` 的真算子 + 打印 + 校验。
   写线时如果想"顺手实现一下"，说明 `c/` 缺东西 —— 补到 `c/`（比如本次给
   `compute_numerical_grads` 加了 `NUMGRAD_QUIET` 开关），再回来串线。
   这条规则防止 lines/ 长成一份平行实现，几个月后和 `c/` 脱钩。
2. **每条线用一个真实案例贯穿**，数据要小到能手算核对（如 ln_attn 的 sqrt(7)），
   或真实到可信（后续 char_pred 用真训练权重）。

## 三、线目录与推荐阅读顺序

**主线例子**：vocab=2 的 abab 续写（规则 a→b, b→a），1032 参数的最小 GPT。
从机制到训练到验证，七条线走完一个模型的全生命周期。

| 顺序 | 线名 | 类型 | 贯穿的竖线 | 案例与看点 |
|------|------|------|-----------|-----------|
| 1 | `embed` | 机制线 | token id → wte/wpe → x | 4 词小世界：词的身份卡(wte) + 座次(wpe) 如何相加；同词不同位，x 不同 |
| 2 | `ln_attn` | 机制线 | LN → QKV → 注意力 → 投影 → 残差 | 手工构造「小猫 追 它」代词消解：「它」以 0.894 权重指回「小猫」 |
| 3 | `multi_head` | 机制线 | QKV → 2 头拆分/并行/拼接 → W_proj 混合 | 头0「自扫门前雪」看自己，头1「朝拜句首」；W_proj 把两份看法叠加确认 |
| 4 | `causal_mask` | 机制线 | attention 内部的 t2<=t 循环 | 掩码 vs 双向同场对比：位置 0 被「还没出生的未来」污染；掩码防泄题 |
| 5 | `tiny_train` | 训练线 | forward → 数值梯度 → AdamW → 更新 | abab 现场训练：loss 0.693→0.03，8 位置预测全对（约 40 秒） |
| 6 | `trained_attn` | 验证线 | 解剖训练好的模型 | 意外发现：注意力退化为历史平均池化，奇偶规则藏在 wpe（相邻位置余弦 ≈ -0.95） |
| 7 | `generate` | 验证线 | 自回归生成闭环 | seed a→abab..., seed b→baba...：同一套权重展开两种序列（约 40 秒） |

读法：**机制线**（1-4）讲零件怎么转，**训练线**（5）讲零件怎么变聪明，
**验证线**（6-7）讲变聪明之后零件变成了什么样。
6 号线是全 repo 最值得先跑的一条 —— 它推翻了「注意力一定在盯上一个词」的直觉。

可选项：`LINE_QUIET=1 ./build/lines_all all` 全跑约 2 分钟（三条训练线各需现场训练）。

## 四、怎么跑

```bash
cmake -S . -B build && cmake --build build --target lines_all -j6   # 编译
./build/lines_all list        # 列出所有线
./build/lines_all ln_attn     # 跑一条线（逐跳日志 + 校验）
./build/lines_all all         # 全跑
LINE_QUIET=1 ./build/lines_all all   # 安静模式：只有校验失败才打印，可进 CI
```

退出码 = 失败的 check 数（0 = 全过）。配合 gdb / CLion 单步：
在 `line_xxx_run` 里打断点，变量窗口里就是真实张量。

## 五、怎么新增一条线

1. 写 `lines/line_xxx.c`，导出一个入口：

```c
#include "../c/transformer.h"
#include "_shared/line.h"

int line_xxx_run(LineCtx* ctx) {
    line_hop(ctx, "跳点标题");
    line_note(ctx, "叙述...");
    /* 调用 c/ 的真算子 */
    line_check_close(ctx, actual, expected, 1e-3f, "校验说明");
    return line_summary(ctx);   /* 返回失败数 */
}
```

2. 在 `lines/lines_registry.c` 的表里加一行 `{"xxx", "一句话说明", line_xxx_run}`；
3. 在 `CMakeLists.txt` 的 `LINE_SOURCES` 里加 `lines/line_xxx.c`。
   `lines_all` 自动获得 `list / xxx / all` 三个入口，无需改其它文件。

## 六、harness API 速查（lines/_shared/line.h）

| 函数 | 作用 | 安静模式 |
|------|------|---------|
| `line_hop(ctx, fmt, ...)` | 大标题，跳点编号自动递增 | 静默 |
| `line_note(ctx, fmt, ...)` | 跳点内的叙述行 | 静默 |
| `line_vec / line_ivec / line_mat` | 打印张量 | 静默 |
| `line_attn_mat` | 注意力方阵（行=查询，权重越大越亮） | 静默 |
| `line_check_close / line_check_true` | 校验并计数 | 只打印失败行 |
| `line_summary(ctx)` | 打印汇总，返回失败数 | 静默 |
