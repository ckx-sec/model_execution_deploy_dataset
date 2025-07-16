set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})

add_executable(example_onnxruntime example.cpp)
target_link_libraries(example_onnxruntime PRIVATE
    model_deploy_dataset_lib
    ${OpenCV_LIBRARIES}
)
target_include_directories(example_onnxruntime PRIVATE ${OpenCV_INCLUDE_DIRS})
if(MEI_ENABLE_ONNXRUNTIME)
    target_link_libraries(example_onnxruntime PRIVATE ONNXRuntime::onnxruntime)
endif()

add_executable(ultraface_detector_onnxruntime ultraface_detector.cpp)
target_link_libraries(ultraface_detector_onnxruntime PRIVATE
    model_deploy_dataset_lib
    ${OpenCV_LIBRARIES}
)
target_include_directories(ultraface_detector_onnxruntime PRIVATE ${OpenCV_INCLUDE_DIRS})
if(MEI_ENABLE_ONNXRUNTIME)
    target_link_libraries(ultraface_detector_onnxruntime PRIVATE ONNXRuntime::onnxruntime)
endif()

add_executable(yolov5_detector_onnxruntime yolov5_detector.cpp)
target_link_libraries(yolov5_detector_onnxruntime PRIVATE
    model_deploy_dataset_lib
    ${OpenCV_LIBRARIES}
)
target_include_directories(yolov5_detector_onnxruntime PRIVATE ${OpenCV_INCLUDE_DIRS})
if(MEI_ENABLE_ONNXRUNTIME)
    target_link_libraries(yolov5_detector_onnxruntime PRIVATE ONNXRuntime::onnxruntime)
endif()

add_executable(emotion_ferplus_onnxruntime emotion_ferplus.cpp)
target_link_libraries(emotion_ferplus_onnxruntime PRIVATE
    model_deploy_dataset_lib
    ${OpenCV_LIBRARIES}
)
target_include_directories(emotion_ferplus_onnxruntime PRIVATE ${OpenCV_INCLUDE_DIRS})
if(MEI_ENABLE_ONNXRUNTIME)
    target_link_libraries(emotion_ferplus_onnxruntime PRIVATE ONNXRuntime::onnxruntime)
endif()

add_executable(fsanet_headpose_onnxruntime fsanet_headpose.cpp)
target_link_libraries(fsanet_headpose_onnxruntime PRIVATE
    model_deploy_dataset_lib
    ${OpenCV_LIBRARIES}
)
target_include_directories(fsanet_headpose_onnxruntime PRIVATE ${OpenCV_INCLUDE_DIRS})
if(MEI_ENABLE_ONNXRUNTIME)
    target_link_libraries(fsanet_headpose_onnxruntime PRIVATE ONNXRuntime::onnxruntime)
endif()

add_executable(gender_googlenet_onnxruntime gender_googlenet.cpp)
target_link_libraries(gender_googlenet_onnxruntime PRIVATE
    model_deploy_dataset_lib
    ${OpenCV_LIBRARIES}
)
target_include_directories(gender_googlenet_onnxruntime PRIVATE ${OpenCV_INCLUDE_DIRS})
if(MEI_ENABLE_ONNXRUNTIME)
    target_link_libraries(gender_googlenet_onnxruntime PRIVATE ONNXRuntime::onnxruntime)
endif()

add_executable(ssrnet_age_onnxruntime ssrnet_age.cpp)
target_link_libraries(ssrnet_age_onnxruntime PRIVATE
    model_deploy_dataset_lib
    ${OpenCV_LIBRARIES}
)
target_include_directories(ssrnet_age_onnxruntime PRIVATE ${OpenCV_INCLUDE_DIRS})
if(MEI_ENABLE_ONNXRUNTIME)
    target_link_libraries(ssrnet_age_onnxruntime PRIVATE ONNXRuntime::onnxruntime)
endif()

add_executable(pfld_landmarks_onnxruntime pfld_landmarks.cpp)
target_link_libraries(pfld_landmarks_onnxruntime PRIVATE
    model_deploy_dataset_lib
    ${OpenCV_LIBRARIES}
)
target_include_directories(pfld_landmarks_onnxruntime PRIVATE ${OpenCV_INCLUDE_DIRS})
if(MEI_ENABLE_ONNXRUNTIME)
    target_link_libraries(pfld_landmarks_onnxruntime PRIVATE ONNXRuntime::onnxruntime)
endif()

add_executable(mnist_onnxruntime mnist.cpp)
target_link_libraries(mnist_onnxruntime PRIVATE
    model_deploy_dataset_lib
    ${OpenCV_LIBRARIES}
)
target_include_directories(mnist_onnxruntime PRIVATE ${OpenCV_INCLUDE_DIRS})
if(MEI_ENABLE_ONNXRUNTIME)
    target_link_libraries(mnist_onnxruntime PRIVATE ONNXRuntime::onnxruntime)
endif()

# add_executable(ssrnet_age ssrnet_age.cpp)
# target_link_libraries(ssrnet_age PRIVATE
#     multi_engine_lib
#     ${OpenCV_LIBRARIES}
# )
# target_include_directories(ssrnet_age PRIVATE ${OpenCV_INCLUDE_DIRS})
# if(MEI_ENABLE_ONNXRUNTIME)
#     target_link_libraries(ssrnet_age PRIVATE ONNXRuntime::onnxruntime)
# endif()

# Copy assets to the build directory
set(ASSETS_DIR ${CMAKE_SOURCE_DIR}/assets)
file(COPY ${ASSETS_DIR}/
    DESTINATION ${EXECUTABLE_OUTPUT_PATH}/assets/
) 