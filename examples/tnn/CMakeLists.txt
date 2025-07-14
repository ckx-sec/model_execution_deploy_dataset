# TNN demo CMakeLists.txt
# 所有 demo 均为 TNN 官方 API 原生实现，无 lite.ai.toolkit 依赖
cmake_minimum_required(VERSION 3.10)
project(tnn_examples)

set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../cmake)
include(tnn)
find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB DEMOS "*.cpp")
foreach(demo ${DEMOS})
    get_filename_component(demo_name ${demo} NAME_WE)
    add_executable(${demo_name} ${demo})
    target_include_directories(${demo_name} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${demo_name} PRIVATE TNN::TNN ${OpenCV_LIBS})
endforeach() 