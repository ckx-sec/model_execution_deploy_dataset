set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})

add_executable(example_mnn example_mnn.cpp)
target_link_libraries(example_mnn PRIVATE
    model_deploy_dataset_lib
    ${OpenCV_LIBRARIES}
    MNN::MNN
)
target_include_directories(example_mnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(example_mnn clean_assets)

add_executable(ultraface_detector_mnn ultraface_detector_mnn.cpp)
target_link_libraries(ultraface_detector_mnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} MNN::MNN)
target_include_directories(ultraface_detector_mnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(ultraface_detector_mnn clean_assets)

add_executable(yolov5_detector_mnn yolov5_detector_mnn.cpp)
target_link_libraries(yolov5_detector_mnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} MNN::MNN)
target_include_directories(yolov5_detector_mnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(yolov5_detector_mnn clean_assets)

add_executable(pfld_landmarks_mnn pfld_landmarks_mnn.cpp)
target_link_libraries(pfld_landmarks_mnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} MNN::MNN)
target_include_directories(pfld_landmarks_mnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(pfld_landmarks_mnn clean_assets)

add_executable(fsanet_headpose_mnn fsanet_headpose_mnn.cpp)
target_link_libraries(fsanet_headpose_mnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} MNN::MNN)
target_include_directories(fsanet_headpose_mnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(fsanet_headpose_mnn clean_assets)

add_executable(gender_googlenet_mnn gender_googlenet_mnn.cpp)
target_link_libraries(gender_googlenet_mnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} MNN::MNN)
target_include_directories(gender_googlenet_mnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(gender_googlenet_mnn clean_assets)

add_executable(emotion_ferplus_mnn emotion_ferplus_mnn.cpp)
target_link_libraries(emotion_ferplus_mnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} MNN::MNN)
target_include_directories(emotion_ferplus_mnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(emotion_ferplus_mnn clean_assets)

add_executable(ssrnet_age_mnn ssrnet_age_mnn.cpp)
target_link_libraries(ssrnet_age_mnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} MNN::MNN)
target_include_directories(ssrnet_age_mnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(ssrnet_age_mnn clean_assets)

# 可继续添加更多 MNN demo
# add_executable(ultraface_detector_mnn ultraface_detector_mnn.cpp)
# target_link_libraries(ultraface_detector_mnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} MNN::MNN)
# target_include_directories(ultraface_detector_mnn PRIVATE ${OpenCV_INCLUDE_DIRS})

# 示例：后续可添加MNN相关的demo
# add_executable(mnn_example mnn_example.cpp)
# target_link_libraries(mnn_example PRIVATE
#     multi_engine_lib
#     ${OpenCV_LIBRARIES}
# )
# target_include_directories(mnn_example PRIVATE ${OpenCV_INCLUDE_DIRS})
# if(MEI_ENABLE_MNN)
#     target_link_libraries(mnn_example PRIVATE MNN::MNN)
# endif() 