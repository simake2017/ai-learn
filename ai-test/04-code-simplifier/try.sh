#!/bin/bash
# 04-code-simplifier 演示命令

/code-simplifier messy.go

# 只重构某个函数
/code-simplifier messy.go --function processWithRetry
