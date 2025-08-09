#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// Helper: Softmax
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

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    // --- Configuration ---
    const std::string tflite_path = argv[1];
    const std::string image_path = argv[2];
    const int input_width = 224;
    const int input_height = 224;

    // --- TFLite setup ---
    auto model = tflite::FlatBufferModel::BuildFromFile(tflite_path.c_str());
    if (!model) {
        std::cerr << "Failed to load model: " << tflite_path << std::endl;
        return -1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter" << std::endl;
        return -1;
    }

    interpreter->SetNumThreads(1);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return -1;
    }

    // --- Preprocessing ---
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_width, input_height));
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    // --- Fill input tensor ---
    float* input_ptr = interpreter->typed_input_tensor<float>(0);
    // HWC layout
    for (int h = 0; h < input_height; ++h) {
        for (int w = 0; w < input_width; ++w) {
            for (int c = 0; c < 3; ++c) {
                // Normalize to [-1, 1]
                input_ptr[(h * input_width + w) * 3 + c] = (resized_image.at<cv::Vec3b>(h, w)[c] - 127.5f) / 127.5f;
            }
        }
    }

    // --- Inference ---
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return -1;
    }

    // --- Get output ---
    float* raw_output = interpreter->typed_output_tensor<float>(0);
    const auto* output_dims = interpreter->output_tensor(0)->dims;
    size_t output_size = output_dims->data[output_dims->size - 1]; // Get last dimension size

    auto probs = softmax(raw_output, output_size);
    auto max_it = std::max_element(probs.begin(), probs.end());
    int max_index = std::distance(probs.begin(), max_it);
    float confidence = *max_it;

    const float prob_threshold = 0.7f;
    const int target_age_index = 4; // 25-32 years

    if (max_index == target_age_index && confidence > prob_threshold) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 