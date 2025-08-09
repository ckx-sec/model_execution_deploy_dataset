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
    const int input_size = 64;

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
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_size, input_size));

    // --- Fill input tensor ---
    // The model expects a float32 tensor.
    float* input_ptr = interpreter->typed_input_tensor<float>(0);
    resized_image.convertTo(resized_image, CV_32F);
    memcpy(input_ptr, resized_image.data, input_size * input_size * sizeof(float));

    // --- Inference ---
    interpreter->Invoke();

    // --- Get output ---
    float* raw_output = interpreter->typed_output_tensor<float>(0);
    const auto* output_dims = interpreter->output_tensor(0)->dims;
    size_t output_size = output_dims->data[output_dims->size - 1];

    auto probs = softmax(raw_output, output_size);

    printf("DEBUG TFLITE: Emotion Probs:\n");
    for(size_t i = 0; i < probs.size(); ++i) {
        printf("  - Prob[%zu]: %.4f\n", i, probs[i]);
    }

    const int happiness_index = 1; // "happiness"
    const int neutral_index = 0;   // "neutral"
    const float happiness_prob = probs[happiness_index];
    const float neutral_prob = probs[neutral_index];

    if (happiness_prob > 0.5f && neutral_prob < 0.4f) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 