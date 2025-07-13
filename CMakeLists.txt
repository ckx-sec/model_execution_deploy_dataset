cmake_minimum_required(VERSION 3.10)
project(multi_engine_inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Building multi_engine_inference")

# --- Options to enable/disable backends ---
option(MEI_ENABLE_ONNXRUNTIME "Enable ONNXRuntime backend" ON)
option(MEI_ENABLE_MNN "Enable MNN backend" ON)
option(MEI_ENABLE_NCNN "Enable NCNN backend" ON)
option(MEI_ENABLE_TNN "Enable TNN backend" ON)
option(MEI_BUILD_EXAMPLES "Build example applications" ON)
option(MEI_LINK_STATIC "Link inference engines statically" OFF)

# --- Include dependency helpers ---
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find dependencies if they are enabled
if(MEI_ENABLE_ONNXRUNTIME)
    include(onnxruntime)
endif()
if(MEI_ENABLE_MNN)
    include(mnn)
endif()
if(MEI_ENABLE_NCNN)
    include(ncnn)
endif()
# We always need OpenCV for the examples for now
include(opencv)

# --- Add source directory ---
add_subdirectory(src)

# --- Add examples ---
if(MEI_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

message(STATUS "Configuration finished.") 