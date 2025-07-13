# TNN cmake module
message(STATUS "Looking for TNN...")

set(TNN_ROOT_DIR "${CMAKE_SOURCE_DIR}/third_party/tnn")
set(TNN_INCLUDE_DIR "${TNN_ROOT_DIR}/include")
set(TNN_LIB_DIR "${TNN_ROOT_DIR}/lib")

if(MEI_LINK_STATIC)
    message(STATUS "Attempting to link TNN statically.")
    # Force find static library (.a)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    find_library(TNN_LIBRARY 
        NAMES TNN
        PATHS ${TNN_LIB_DIR}
        NO_DEFAULT_PATH
    )
    # Restore default find suffixes
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_DEFAULT_FIND_LIBRARY_SUFFIXES})
else()
    message(STATUS "Attempting to link TNN dynamically.")
    find_library(TNN_LIBRARY 
        NAMES TNN
        PATHS ${TNN_LIB_DIR}
        NO_DEFAULT_PATH
    )
endif()

if(NOT TNN_LIBRARY)
    message(FATAL_ERROR "TNN library not found in ${TNN_LIB_DIR}")
endif()

if(MEI_LINK_STATIC)
    add_library(TNN::TNN STATIC IMPORTED)
else()
    add_library(TNN::TNN SHARED IMPORTED)
endif()

set_target_properties(TNN::TNN PROPERTIES
    IMPORTED_LOCATION "${TNN_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TNN_INCLUDE_DIR}"
)

message(STATUS "Found TNN: ${TNN_LIBRARY}") 