# Python 脚本说明

本目录包含 Python 胶水和可视化工具。

## 文件

| 文件 | 用途 |
|---|---|
| `train.py` | 一键训练（编译 + 调 C 跑训练 + 推理）|
| `inference_demo.py` | 推理演示（生成字符）|
| `visualize.py` | 画 loss 曲线 / attention 热力图 |

## 用法

```bash
# 训练
python python/train.py

# 推理
python python/inference_demo.py

# 画 loss 曲线（需要先保存训练日志）
./build/transformer train > /tmp/train.log 2>&1
python python/visualize.py --log /tmp/train.log --save loss.png

# 画 attention（需要先跑一次推理保存数据）
python python/visualize.py --attention
```

## 为什么 Python 脚本这么少？

本项目**核心在 C 代码**里（手写 forward + 数值微分 + AdamW）。

Python 只做辅助：
- 调 C 库
- 画图
- 日志解析

更复杂的应用（比如数据集准备）请参考 `nanoGPT/` 目录下的脚本。