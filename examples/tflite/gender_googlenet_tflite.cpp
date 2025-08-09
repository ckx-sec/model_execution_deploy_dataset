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
    const std::string tflite_path = argv[1];
    const std::string image_path = argv[2];
    const int input_size = 224;

    // --- TFLite setup ---
    auto model = tflite::FlatBufferModel::BuildFromFile(tflite_path.c_str());
    if (!model) {
        std::cerr << "Failed to load model: " << tflite_path << std::endl;
        return -1;
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    interpreter->AllocateTensors();

    // --- Preprocessing ---
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_size, input_size));
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    // --- Fill input tensor ---
    float* input_ptr = interpreter->typed_input_tensor<float>(0);
    for (int h = 0; h < input_size; ++h) {
        for (int w = 0; w < input_size; ++w) {
            for (int c = 0; c < 3; ++c) {
                // Normalize to [-1, 1]
                input_ptr[(h * input_size + w) * 3 + c] = (resized_image.at<cv::Vec3b>(h, w)[c] - 127.5f) / 128.0f;
            }
        }
    }

    // --- Inference ---
    interpreter->Invoke();

    // --- Get output ---
    float* raw_output = interpreter->typed_output_tensor<float>(0);
    const auto* output_dims = interpreter->output_tensor(0)->dims;
    size_t output_size = output_dims->data[output_dims->size - 1];

    auto probs = softmax(raw_output, output_size);
    auto max_it = std::max_element(probs.begin(), probs.end());
    int predicted_gender_index = std::distance(probs.begin(), max_it);
    const int target_gender_index = 1; // "Female"

    printf("DEBUG TFLITE: Predicted Gender Index: %d, Probs: [F: %.4f, M: %.4f]\n", 
           predicted_gender_index, probs.size() > 1 ? probs[1] : -1.0f, probs.size() > 0 ? probs[0] : -1.0f);

    if (predicted_gender_index == target_gender_index) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 