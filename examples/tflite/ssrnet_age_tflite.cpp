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
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // --- Fill input tensor ---
    float* input_ptr = interpreter->typed_input_tensor<float>(0);
    resized.convertTo(resized, CV_32F);
    
    // Normalize with ImageNet mean and std
    float mean[] = {0.485f, 0.456f, 0.406f};
    float scale[] = {1/0.229f, 1/0.224f, 1/0.225f};

    for (int h = 0; h < input_size; ++h) {
        for (int w = 0; w < input_size; ++w) {
            for (int c = 0; c < 3; ++c) {
                input_ptr[(h * input_size + w) * 3 + c] = (resized.at<cv::Vec3f>(h, w)[c] / 255.0f - mean[c]) * scale[c];
            }
        }
    }

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