#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// --- Main Inference Logic ---
int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    // --- Configuration ---
    const std::string onnx_path = argv[1];
    const std::string image_path = argv[2];
    const int input_width = 64;
    const int input_height = 64;

    // --- ONNXRuntime setup ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test-ssrnet");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, onnx_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // --- Get model input/output details ---
    std::vector<const char*> input_node_names = {"input"};
    std::vector<const char*> output_node_names = {"age"};
    
    std::cout << "--- Model Inputs ---" << std::endl;
    std::cout << "Input 0 : name=" << input_node_names[0] << std::endl;
    std::cout << "--- Model Outputs ---" << std::endl;
    std::cout << "Output 0 : name=" << output_node_names[0] << std::endl;
    std::cout << "--------------------" << std::endl;

    // --- Preprocessing ---
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_width, input_height));
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);
    
    // Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    float mean[] = {0.485f, 0.456f, 0.406f};
    float scale[] = {1/0.229f, 1/0.224f, 1/0.225f};
    
    std::vector<float> input_tensor_values(1 * 3 * input_height * input_width);
    
    // HWC to CHW and normalize
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_height; ++h) {
            for (int w = 0; w < input_width; ++w) {
                input_tensor_values[c * input_height * input_width + h * input_width + w] =
                    (resized_image.at<cv::Vec3f>(h, w)[c] - mean[c]) * scale[c];
            }
        }
    }

    // --- Create Tensor ---
    std::vector<int64_t> input_node_dims = {1, 3, input_height, input_width};
    size_t input_tensor_size = 1 * 3 * input_height * input_width;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size()
    );

    // --- Inference ---
    std::cout << "Running inference..." << std::endl;
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1
    );
    std::cout << "Inference finished." << std::endl;

    // --- Post-processing ---
    const float* raw_output = output_tensors[0].GetTensorData<float>();
    float predicted_age = raw_output[0];

    if (predicted_age >= 20.0f && predicted_age <= 40.0f) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    
    return 0;
} 