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
void softmax(std::vector<T>& input) {
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

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s [model_path] [image_path]\n", argv[0]);
        return 1;
    }
    const std::string tflite_path = argv[1];
    const std::string image_path = argv[2];
    const int input_size = 28;

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
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size, input_size));
    
    // --- Fill input tensor ---
    float* input_ptr = interpreter->typed_input_tensor<float>(0);
    for (int i = 0; i < input_size * input_size; ++i) {
        // Normalize to [0, 1]
        input_ptr[i] = resized.data[i] / 255.0f;
    }

    // --- Inference ---
    interpreter->Invoke();

    // --- Get output ---
    float* raw_output = interpreter->typed_output_tensor<float>(0);
    const auto* output_dims = interpreter->output_tensor(0)->dims;
    size_t output_size = output_dims->data[output_dims->size - 1];
    
    std::vector<float> results(raw_output, raw_output + output_size);
    softmax(results);
    
    auto max_it = std::max_element(results.begin(), results.end());
    int64_t predicted_digit = std::distance(results.begin(), max_it);
    float confidence = *max_it;

    printf("DEBUG TFLITE: Predicted Digit: %ld, Confidence: %.4f\n", predicted_digit, confidence);
    for(size_t i = 0; i < results.size(); ++i) {
        printf("  - Prob[%zu]: %.4f\n", i, results[i]);
    }

    const float prob_threshold = 0.7f;
    const int target_digit = 7;

    if (predicted_digit == target_digit && confidence > prob_threshold) {
        printf("true\n");
    } else {
        printf("false\n");
    }

    return 0;
} 