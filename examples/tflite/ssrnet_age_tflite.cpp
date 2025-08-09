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
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size, input_size));

    // --- Fill input tensor ---
    float* input_ptr = interpreter->typed_input_tensor<float>(0);
    // TFLite models usually expect float input in [0,1] or [-1,1] range.
    // Assuming normalization to [0,1] as a common case.
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    memcpy(input_ptr, resized.data, input_size * input_size * 3 * sizeof(float));

    // --- Inference ---
    interpreter->Invoke();

    // --- Get output ---
    float* predicted_age = interpreter->typed_output_tensor<float>(0);
    
    if (predicted_age[0] >= 18.0f && predicted_age[0] <= 35.0f) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 