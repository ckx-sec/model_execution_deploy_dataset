set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})

# --- Template for adding a new TFLite executable ---
# NOTE: To compile these targets, you must first add the TFLite dependency
# to the root CMakeLists.txt and ensure it provides the TFLite::tflite target.

macro(add_tflite_executable name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE
        model_deploy_dataset_lib
        ${OpenCV_LIBRARIES}
        TFLite::tflite
    )
    target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
    add_dependencies(${name} clean_assets)
endmacro()

add_tflite_executable(age_googlenet_tflite)
add_tflite_executable(emotion_ferplus_tflite)
add_tflite_executable(fsanet_headpose_tflite)
add_tflite_executable(gender_googlenet_tflite)
add_tflite_executable(mnist_tflite)
add_tflite_executable(pfld_landmarks_tflite)
add_tflite_executable(ssrnet_age_tflite)
add_tflite_executable(ultraface_detector_tflite)
add_tflite_executable(yolov5_detector_tflite) 