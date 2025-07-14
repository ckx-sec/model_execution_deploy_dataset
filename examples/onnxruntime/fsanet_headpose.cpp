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
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test-headpose");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, onnx_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // --- Get model input/output details ---
    std::vector<const char*> input_node_names = {"input"};
    std::vector<const char*> output_node_names = {"output"};

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

    // 1. Padding
    const float pad = 0.3f;
    const int h = image.rows;
    const int w = image.cols;
    const int nh = static_cast<int>(static_cast<float>(h) + pad * static_cast<float>(h));
    const int nw = static_cast<int>(static_cast<float>(w) + pad * static_cast<float>(w));
    const int nx1 = std::max(0, static_cast<int>((nw - w) / 2));
    const int ny1 = std::max(0, static_cast<int>((nh - h) / 2));

    cv::Mat padded_image = cv::Mat(nh, nw, CV_8UC3, cv::Scalar(0, 0, 0));
    image.copyTo(padded_image(cv::Rect(nx1, ny1, w, h)));
    
    // 2. Resize
    cv::Mat resized_image;
    cv::resize(padded_image, resized_image, cv::Size(input_width, input_height));

    // 3. Normalize
    std::vector<float> input_tensor_values(1 * 3 * input_height * input_width);
    resized_image.convertTo(resized_image, CV_32F);
    
    // HWC to CHW and normalize
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_height; ++h) {
            for (int w = 0; w < input_width; ++w) {
                input_tensor_values[c * input_height * input_width + h * input_width + w] =
                    (resized_image.at<cv::Vec3f>(h, w)[c] - 127.5f) / 127.5f;
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
    
    float yaw = raw_output[0];
    float pitch = raw_output[1];
    float roll = raw_output[2];

    const float angle_threshold = 15.0f;

    if (std::abs(yaw) < angle_threshold && std::abs(pitch) < angle_threshold && std::abs(roll) < angle_threshold) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    
    return 0;
} 