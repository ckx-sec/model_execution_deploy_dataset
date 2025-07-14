set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})

add_executable(ultraface_detector_ncnn ultraface_detector_ncnn.cpp)
target_link_libraries(ultraface_detector_ncnn PRIVATE
    model_deploy_dataset_lib
    ${OpenCV_LIBRARIES}
    NCNN::ncnn
)
target_include_directories(ultraface_detector_ncnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(ultraface_detector_ncnn clean_assets)

add_executable(yolov5_detector_ncnn yolov5_detector_ncnn.cpp)
target_link_libraries(yolov5_detector_ncnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} NCNN::ncnn)
target_include_directories(yolov5_detector_ncnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(yolov5_detector_ncnn clean_assets)

add_executable(fsanet_headpose_ncnn fsanet_headpose_ncnn.cpp)
target_link_libraries(fsanet_headpose_ncnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} NCNN::ncnn)
target_include_directories(fsanet_headpose_ncnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(fsanet_headpose_ncnn clean_assets)

add_executable(gender_googlenet_ncnn gender_googlenet_ncnn.cpp)
target_link_libraries(gender_googlenet_ncnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} NCNN::ncnn)
target_include_directories(gender_googlenet_ncnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(gender_googlenet_ncnn clean_assets)

add_executable(emotion_ferplus_ncnn emotion_ferplus_ncnn.cpp)
target_link_libraries(emotion_ferplus_ncnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} NCNN::ncnn)
target_include_directories(emotion_ferplus_ncnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(emotion_ferplus_ncnn clean_assets)

add_executable(pfld_landmarks_ncnn pfld_landmarks_ncnn.cpp)
target_link_libraries(pfld_landmarks_ncnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} NCNN::ncnn)
target_include_directories(pfld_landmarks_ncnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(pfld_landmarks_ncnn clean_assets)

add_executable(ssrnet_age_ncnn ssrnet_age_ncnn.cpp)
target_link_libraries(ssrnet_age_ncnn PRIVATE model_deploy_dataset_lib ${OpenCV_LIBRARIES} NCNN::ncnn)
target_include_directories(ssrnet_age_ncnn PRIVATE ${OpenCV_INCLUDE_DIRS})
add_dependencies(ssrnet_age_ncnn clean_assets)

# 可继续添加更多 NCNN demo 