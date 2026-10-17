cmake_minimum_required(VERSION 3.16)
project(transformer-zero C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unused-result")

# ============================================================
# 核心源文件（所有算子，不含 main.c）
# ============================================================
set(CORE_SOURCES
    c/matrix.c
    c/activations.c
    c/layernorm.c
    c/attention.c
    c/model.c
    c/autodiff.c
    c/optimizer.c
    c/dataloader.c
)

# ============================================================
# 主程序: ./transformer
# ============================================================
add_executable(transformer ${CORE_SOURCES} c/main.c)
target_link_libraries(transformer m)

# ============================================================
# 单元测试: 每个 test_XX 独立链接 CORE_SOURCES
# ============================================================
set(TESTS
    test_01_embedding
    test_02_matmul
    test_03_layernorm
    test_04_qkv
    test_05_attention
    test_06_gelu
    test_07_softmax
    test_08_crossentropy
    test_09_residual
    test_10_forward
    test_11_gradient
    test_12_adamw
)

foreach(test_name ${TESTS})
    add_executable(${test_name} test/${test_name}.c ${CORE_SOURCES})
    target_include_directories(${test_name} PRIVATE .)
    target_link_libraries(${test_name} m)
endforeach()

# ============================================================
# 学习线: lines/（端到端调用链，逐跳日志，非单元测试）
#   ./lines_all list        列出所有线
#   ./lines_all <名字>      跑一条线
#   ./lines_all all         全跑（LINE_QUIET=1 安静模式，可进 CI）
# ============================================================
set(LINE_SOURCES
    lines/_shared/line.c
    lines/lines_registry.c
    lines/line_embed.c
    lines/line_ln_attn.c
    lines/line_multi_head.c
    lines/line_causal_mask.c
    lines/line_tiny_train.c
    lines/line_trained_attn.c
    lines/line_generate.c
)

add_executable(lines_all lines/lines_main.c ${LINE_SOURCES} ${CORE_SOURCES})
target_include_directories(lines_all PRIVATE . lines/_shared)
target_link_libraries(lines_all m)
