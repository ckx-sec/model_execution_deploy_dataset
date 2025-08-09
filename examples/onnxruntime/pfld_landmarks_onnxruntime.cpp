#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// --- Helper Functions ---

// Function to draw landmarks on the image
void draw_landmarks(cv::Mat& image, const std::vector<cv::Point2f>& landmarks) {
    for (const auto& point : landmarks) {
        cv::circle(image, point, 2, cv::Scalar(0, 255, 0), -1);
    }
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
    const int input_width = 112;
    const int input_height = 112;

    // --- ONNXRuntime setup ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test-landmark");
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
    float img_height_orig = static_cast<float>(image.rows);
    float img_width_orig = static_cast<float>(image.cols);

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_width, input_height));
    
    std::vector<float> input_tensor_values(1 * 3 * input_height * input_width);
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);
    
    // HWC to CHW
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_height; ++h) {
            for (int w = 0; w < input_width; ++w) {
                input_tensor_values[c * input_height * input_width + h * input_width + w] =
                    resized_image.at<cv::Vec3f>(h, w)[c];
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
        Ort::RunOptions{nullptr}, input_node_names_char.data(), &input_tensor, 1, output_node_names_char.data(), num_output_nodes
    );
    std::cout << "Inference finished." << std::endl;

    // --- Post-processing ---
    // PFLD has 2 outputs, the second one is landmarks
    const float* raw_output = output_tensors[1].GetTensorData<float>();
    auto output_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    size_t num_landmarks = output_shape[1]; // Should be 212 (106 * 2)

    bool valid = true;
    for (size_t i = 0; i < num_landmarks; i += 2) {
        float x = raw_output[i];
        float y = raw_output[i+1];
        if (x < 0.02f || x > 0.98f || y < 0.02f || y > 0.98f) {
            valid = false;
            break;
        }
    }
    if (valid) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 