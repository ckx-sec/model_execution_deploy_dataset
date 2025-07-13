if(MEI_ENABLE_ONNXRUNTIME)
    add_subdirectory(onnxruntime)
endif()
if(MEI_ENABLE_MNN)
    add_subdirectory(mnn)
endif()
if(MEI_ENABLE_NCNN)
    add_subdirectory(ncnn)
endif()
if(MEI_ENABLE_TNN)
    add_subdirectory(tnn)
endif()

# Copy assets to the build directory
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(ASSETS_DIR ${CMAKE_SOURCE_DIR}/assets)
file(COPY ${ASSETS_DIR}/
    DESTINATION ${EXECUTABLE_OUTPUT_PATH}/assets/
)

# 添加自定义target用于删除拷贝的assets目录
add_custom_target(clean_assets
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${EXECUTABLE_OUTPUT_PATH}/assets
    COMMENT "Remove copied assets directory after build"
) 