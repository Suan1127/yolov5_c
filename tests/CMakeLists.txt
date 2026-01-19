cmake_minimum_required(VERSION 3.12)

# Test executables
add_executable(test_conv1x1 test_conv1x1.c)
target_link_libraries(test_conv1x1 yolov5_c)
# Math library (only needed on Unix/Linux, not Windows)
if(NOT MSVC)
    target_link_libraries(test_conv1x1 m)
endif()

add_executable(test_integration test_integration.c)
target_link_libraries(test_integration yolov5_c)
# Math library (only needed on Unix/Linux, not Windows)
if(NOT MSVC)
    target_link_libraries(test_integration m)
endif()

add_test(NAME Conv1x1 COMMAND test_conv1x1)
add_test(NAME Integration COMMAND test_integration)