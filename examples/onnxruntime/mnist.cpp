#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// Softmax function
template <typename T>
static void softmax(T& input) {
  float rowmax = *std::max_element(input.begin(), input.end());
  std::vector<float> y(input.size());
  float sum = 0.0f;
  for (size_t i = 0; i != input.size(); ++i) {
    sum += y[i] = std::exp(input[i] - rowmax);
  }
  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = y[i] / sum;
  }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    // 1. Configuration
    const std::string onnx_path = argv[1];
    const std::string image_path = argv[2];
    const int input_width = 28;
    const int input_height = 28;

    // 2. ONNXRuntime setup
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mnist-onnx");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, onnx_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // 3. Preprocessing
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_width, input_height));
    
    std::vector<float> input_tensor_values(1 * 1 * input_height * input_width);
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);
    memcpy(input_tensor_values.data(), resized_image.data, input_tensor_values.size() * sizeof(float));

    // 4. Create Tensor
    std::vector<const char*> input_node_names = {"Input3"}; // Or your specific input name
    std::vector<const char*> output_node_names = {"Plus214_Output_0"}; // Or your specific output name
    std::vector<int64_t> input_node_dims = {1, 1, input_height, input_width};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_node_dims.data(), input_node_dims.size()
    );

    // 5. Inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1
    );

    // 6. Post-processing
    const float* raw_output = output_tensors[0].GetTensorData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = output_shape[1]; // Should be 10 for MNIST
    
    std::vector<float> results(raw_output, raw_output + output_size);
    softmax(results);
    
    auto max_it = std::max_element(results.begin(), results.end());
    int64_t predicted_digit = std::distance(results.begin(), max_it);
    float confidence = *max_it;

    const float prob_threshold = 0.8f;
    const int target_digit = 7;

    if (predicted_digit == target_digit && confidence > prob_threshold) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    
    return 0;
} 