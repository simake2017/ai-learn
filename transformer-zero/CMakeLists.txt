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
