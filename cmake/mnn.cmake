message(STATUS "Looking for MNN...")

set(MNN_ROOT_DIR "${CMAKE_SOURCE_DIR}/third_party/mnn")
set(MNN_INCLUDE_DIR "${MNN_ROOT_DIR}/include")
set(MNN_LIB_DIR "${MNN_ROOT_DIR}/lib")

if(MEI_LINK_STATIC)
    message(STATUS "Attempting to link MNN statically.")
    # Force find static library (.a)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    find_library(MNN_LIBRARY 
        NAMES MNN
        PATHS ${MNN_LIB_DIR}
        NO_DEFAULT_PATH
    )
    # Restore default find suffixes
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_DEFAULT_FIND_LIBRARY_SUFFIXES})
else()
    message(STATUS "Attempting to link MNN dynamically.")
    find_library(MNN_LIBRARY 
        NAMES MNN
        PATHS ${MNN_LIB_DIR}
        NO_DEFAULT_PATH
    )
endif()

if(NOT MNN_LIBRARY)
    message(FATAL_ERROR "MNN library not found in ${MNN_LIB_DIR}")
endif()

if(MEI_LINK_STATIC)
    add_library(MNN::MNN STATIC IMPORTED)
else()
    add_library(MNN::MNN SHARED IMPORTED)
endif()

set_target_properties(MNN::MNN PROPERTIES
    IMPORTED_LOCATION "${MNN_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MNN_INCLUDE_DIR}"
)

message(STATUS "Found MNN: ${MNN_LIBRARY}") 