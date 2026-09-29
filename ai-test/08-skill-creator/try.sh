#!/bin/bash
# 08-skill-creator 演示命令

# 1. 从零创建一个 Skill（go 测试生成器）
/skill-creator 创建一个 go-test-writer 技能，功能：输入一个 Go 文件路径，自动生成 table-driven 测试文件，覆盖正常、边界、错误场景，测试命名符合 TestXxx_Yyy 规范

# 2. 改进已有 Skill
/skill-creator improve go-test-writer

# 3. 跑评估测试（5 个用例）
/skill-creator eval go-test-writer --cases 5
