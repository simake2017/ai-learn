#!/bin/bash
# 02-code-review 演示命令

/code-review buggy.go

# 进阶：只看安全类问题
/code-review buggy.go --focus security
