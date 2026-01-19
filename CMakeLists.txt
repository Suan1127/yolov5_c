cmake_minimum_required(VERSION 3.12)
project(YOLOv5_C VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build options
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_TOOLS "Build C tools" ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blocks
    ${CMAKE_CURRENT_SOURCE_DIR}/src/models
    ${CMAKE_CURRENT_SOURCE_DIR}/src/postprocess
)

# Source files
set(CORE_SOURCES
    src/core/tensor.c
    src/core/memory.c
    src/core/weights_loader.c
)

set(OPS_SOURCES
    src/ops/conv2d.c
    src/ops/activation.c
    src/ops/batchnorm2d.c
    src/ops/pooling.c
    src/ops/upsample.c
    src/ops/concat.c
)

set(BLOCKS_SOURCES
    src/blocks/bottleneck.c
    src/blocks/c3.c
    src/blocks/sppf.c
)

set(MODELS_SOURCES
    src/models/yolov5s_graph.c
    src/models/yolov5s_build.c
    src/models/yolov5s_infer.c
)

set(POSTPROCESS_SOURCES
    src/postprocess/detect.c
    src/postprocess/nms.c
)

# Main library
add_library(yolov5_c STATIC
    ${CORE_SOURCES}
    ${OPS_SOURCES}
    ${BLOCKS_SOURCES}
    ${MODELS_SOURCES}
    ${POSTPROCESS_SOURCES}
)

# Main executable
add_executable(yolov5_infer src/main.c)
target_link_libraries(yolov5_infer yolov5_c)
# Math library (only needed on Unix/Linux, not Windows)
if(NOT MSVC)
    target_link_libraries(yolov5_infer m)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(/W4)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-O2 -march=native)  # Optimization
endif()
