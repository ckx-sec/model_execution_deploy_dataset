# Find TensorFlow Lite

message(STATUS "Looking for TensorFlow Lite...")

# --- Search for TFLite library and includes ---
# You can specify the root of your TFLite installation by setting TENSORFLOWLITE_ROOT_DIR
# We also hint at the default location where our own build script installs it.
find_path(TENSORFLOWLITE_INCLUDE_DIR
    NAMES "tensorflow/lite/interpreter.h"
    HINTS
        ENV TENSORFLOWLITE_ROOT_DIR
        "${CMAKE_SOURCE_DIR}/third_party/tflite"
    PATH_SUFFIXES include
)

find_library(TENSORFLOWLITE_LIBRARY
    NAMES tensorflow-lite
    HINTS
        ENV TENSORFLOWLITE_ROOT_DIR
        "${CMAKE_SOURCE_DIR}/third_party/tflite"
    PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TFLite
    DEFAULT_MSG
    TENSORFLOWLITE_LIBRARY TENSORFLOWLITE_INCLUDE_DIR
)

mark_as_advanced(TENSORFLOWLITE_INCLUDE_DIR TENSORFLOWLITE_LIBRARY)

if(TFLITE_FOUND)
    message(STATUS "Found TensorFlow Lite library: ${TENSORFLOWLITE_LIBRARY}")
    message(STATUS "Found TensorFlow Lite include path: ${TENSORFLOWLITE_INCLUDE_DIR}")
    
    add_library(TFLite::tflite UNKNOWN IMPORTED)
    set_target_properties(TFLite::tflite PROPERTIES
        IMPORTED_LOCATION "${TENSORFLOWLITE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TENSORFLOWLITE_INCLUDE_DIR}"
    )
else()
    message(FATAL_ERROR "Could not find TensorFlow Lite. Please ensure TENSORFLOWLITE_ROOT_DIR is set correctly.")
endif() 