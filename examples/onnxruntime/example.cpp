#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// --- Helper Functions ---

// A simple softmax implementation
template <typename T>
std::vector<T> softmax(const T* data, size_t size) {
    std::vector<T> result(size);
    const T max_val = *std::max_element(data, data + size);
    T sum = 0;
    for (size_t i = 0; i < size; ++i) {
        result[i] = std::exp(data[i] - max_val);
        sum += result[i];
    }
    for (size_t i = 0; i < size; ++i) {
        result[i] /= sum;
    }
    return result;
}

// --- Main Inference Logic ---
int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    // --- Configuration ---
    const std::string onnx_path = argv[1];
    const std::string image_path = argv[2];
    const int input_width = 224;
    const int input_height = 224;

    // --- ONNXRuntime setup ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, onnx_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // --- Print model input/output details ---
    size_t num_input_nodes = session.GetInputCount();
    std::vector<std::string> input_node_names_str;
    std::vector<const char*> input_node_names_char;
    input_node_names_str.reserve(num_input_nodes);
    input_node_names_char.reserve(num_input_nodes);

    std::cout << "--- Model Inputs ---" << std::endl;
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        std::cout << "Input " << i << " : name=" << input_name.get() << std::endl;
        input_node_names_str.push_back(input_name.get());
    }
    for(const auto& name : input_node_names_str) {
        input_node_names_char.push_back(name.c_str());
    }

    size_t num_output_nodes = session.GetOutputCount();
    std::vector<std::string> output_node_names_str;
    std::vector<const char*> output_node_names_char;
    output_node_names_str.reserve(num_output_nodes);
    output_node_names_char.reserve(num_output_nodes);
    
    std::cout << "--- Model Outputs ---" << std::endl;
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        std::cout << "Output " << i << " : name=" << output_name.get() << std::endl;
        output_node_names_str.push_back(output_name.get());
    }
    for(const auto& name : output_node_names_str) {
        output_node_names_char.push_back(name.c_str());
    }
    std::cout << "--------------------" << std::endl;

    // --- Preprocessing ---
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_width, input_height));
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    std::vector<float> input_tensor_values(1 * 3 * input_height * input_width);
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);
    
    // Normalize (assuming mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    float mean[] = {0.485f, 0.456f, 0.406f};
    float scale[] = {1/0.229f, 1/0.224f, 1/0.225f};
    
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
        Ort::RunOptions{nullptr}, input_node_names_char.data(), &input_tensor, num_input_nodes, output_node_names_char.data(), num_output_nodes
    );
    std::cout << "Inference finished." << std::endl;

    // --- Post-processing ---
    const float* raw_output = output_tensors[0].GetTensorData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
    
    auto probs = softmax(raw_output, output_size);
    
    // Find the interval with the highest probability
    auto max_it = std::max_element(probs.begin(), probs.end());
    int max_index = std::distance(probs.begin(), max_it);
    float confidence = *max_it;

    const float prob_threshold = 0.4f;
    const int target_age_index = 4; // 25-32 years
    
    if (max_index == target_age_index && confidence > prob_threshold) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    
    return 0;
} 