# --- Find ONNXRuntime ---
# This script will be responsible for finding the ONNXRuntime libraries and headers.
# For now, it's a placeholder.

message(STATUS "Looking for ONNXRuntime...")

set(ONNXRUNTIME_ROOT_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT_DIR}/include"
    PATH_SUFFIXES onnxruntime
    NO_DEFAULT_PATH
)

if(MEI_LINK_STATIC)
    message(STATUS "Attempting to link ONNXRuntime statically.")
    # Force find static library (.a)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    find_library(ONNXRUNTIME_LIBRARY 
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT_DIR}/lib"
        NO_DEFAULT_PATH
    )
    # Restore default find suffixes
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_DEFAULT_FIND_LIBRARY_SUFFIXES})
else()
    message(STATUS "Attempting to link ONNXRuntime dynamically.")
    # Find either .so on Linux or .dylib on macOS
    find_library(ONNXRUNTIME_LIBRARY 
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT_DIR}/lib"
        NO_DEFAULT_PATH
    )
endif()

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    set(ONNXRuntime_FOUND TRUE)
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY})
else()
    message(FATAL_ERROR "ONNXRuntime not found in third_party directory!")
endif()

if(ONNXRuntime_FOUND)
    if(MEI_LINK_STATIC)
        add_library(ONNXRuntime::onnxruntime STATIC IMPORTED)
    else()
        add_library(ONNXRuntime::onnxruntime SHARED IMPORTED)
    endif()

    set_target_properties(ONNXRuntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
    )
    message(STATUS "Found ONNXRuntime: ${ONNXRUNTIME_LIBRARIES}")
endif() 