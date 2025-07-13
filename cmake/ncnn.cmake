message(STATUS "Looking for NCNN...")

set(NCNN_ROOT_DIR "${CMAKE_SOURCE_DIR}/third_party/ncnn")
set(NCNN_INCLUDE_DIR "${NCNN_ROOT_DIR}/include")
set(NCNN_LIB_DIR "${NCNN_ROOT_DIR}/lib")

if(MEI_LINK_STATIC)
    message(STATUS "Attempting to link NCNN statically.")
    # Force find static library (.a)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    find_library(NCNN_LIBRARY 
        NAMES ncnn
        PATHS ${NCNN_LIB_DIR}
        NO_DEFAULT_PATH
    )
    # Restore default find suffixes
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_DEFAULT_FIND_LIBRARY_SUFFIXES})
else()
    message(STATUS "Attempting to link NCNN dynamically.")
    find_library(NCNN_LIBRARY 
        NAMES ncnn
        PATHS ${NCNN_LIB_DIR}
        NO_DEFAULT_PATH
    )
endif()

if(NOT NCNN_LIBRARY)
    message(FATAL_ERROR "NCNN library not found in ${NCNN_LIB_DIR}")
endif()

if(MEI_LINK_STATIC)
    add_library(NCNN::ncnn STATIC IMPORTED)
else()
    add_library(NCNN::ncnn SHARED IMPORTED)
endif()

set_target_properties(NCNN::ncnn PROPERTIES
    IMPORTED_LOCATION "${NCNN_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCNN_INCLUDE_DIR}/ncnn"
)

message(STATUS "Found NCNN: ${NCNN_LIBRARY}") 