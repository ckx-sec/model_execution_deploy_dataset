#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    const std::string tflite_path = argv[1];
    const std::string image_path = argv[2];
    const int input_size = 112;

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
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size, input_size));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // --- Fill input tensor ---
    float* input_ptr = interpreter->typed_input_tensor<float>(0);
    for (int h = 0; h < input_size; ++h) {
        for (int w = 0; w < input_size; ++w) {
            for (int c = 0; c < 3; ++c) {
                // Normalize to [-1, 1]
                input_ptr[(h * input_size + w) * 3 + c] = (resized.at<cv::Vec3b>(h, w)[c] - 127.5f) / 128.0f;
            }
        }
    }

    // --- Inference ---
    interpreter->Invoke();

    // --- Get output ---
    // PFLD has 2 outputs, the first one is landmarks
    float* raw_output = interpreter->typed_output_tensor<float>(0);
    const auto* output_dims = interpreter->output_tensor(0)->dims;
    size_t num_landmarks = output_dims->data[1] * output_dims->data[2]; // Should be 212 (106 * 2)

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