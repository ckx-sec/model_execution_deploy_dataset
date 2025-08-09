#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// Helper function to run inference on a single TFLite model
void run_fsanet_model(
    tflite::Interpreter* interpreter,
    const cv::Mat& img,
    float& yaw, float& pitch, float& roll)
{
    const int input_size = 64;

    // --- Preprocessing ---
    const float pad = 0.3f;
    const int h = img.rows;
    const int w = img.cols;
    const int nh = static_cast<int>(static_cast<float>(h) + pad * static_cast<float>(h));
    const int nw = static_cast<int>(static_cast<float>(w) + pad * static_cast<float>(w));
    const int nx1 = std::max(0, static_cast<int>((nw - w) / 2));
    const int ny1 = std::max(0, static_cast<int>((nh - h) / 2));

    cv::Mat padded_image = cv::Mat(nh, nw, CV_8UC3, cv::Scalar(0, 0, 0));
    img.copyTo(padded_image(cv::Rect(nx1, ny1, w, h)));
    
    cv::Mat resized;
    cv::resize(padded_image, resized, cv::Size(input_size, input_size));

    // --- Fill input tensor ---
    float* input_ptr = interpreter->typed_input_tensor<float>(0);
    for (int h_idx = 0; h_idx < input_size; ++h_idx) {
        for (int w_idx = 0; w_idx < input_size; ++w_idx) {
            for (int c = 0; c < 3; ++c) {
                // Normalize to [-1, 1]
                input_ptr[(h_idx * input_size + w_idx) * 3 + c] = (resized.at<cv::Vec3b>(h_idx, w_idx)[c] - 127.5f) / 127.5f;
            }
        }
    }

    // --- Inference ---
    interpreter->Invoke();

    // --- Post-processing ---
    float* raw_output = interpreter->typed_output_tensor<float>(0);
    // TFLite output is not scaled, so we scale it by 90.0f
    yaw = raw_output[0] * 90.0f;
    pitch = raw_output[1] * 90.0f;
    roll = raw_output[2] * 90.0f;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <var_model_path> <1x1_model_path> <image_path>" << std::endl;
        return -1;
    }
    std::string var_model_path = argv[1];
    std::string conv_model_path = argv[2];
    std::string image_path = argv[3];

    // --- Load Models ---
    auto var_model = tflite::FlatBufferModel::BuildFromFile(var_model_path.c_str());
    auto conv_model = tflite::FlatBufferModel::BuildFromFile(conv_model_path.c_str());
    if (!var_model || !conv_model) {
        std::cerr << "Failed to load TFLite models" << std::endl;
        return -1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> var_interpreter, conv_interpreter;
    tflite::InterpreterBuilder(*var_model, resolver)(&var_interpreter);
    tflite::InterpreterBuilder(*conv_model, resolver)(&conv_interpreter);
    var_interpreter->AllocateTensors();
    conv_interpreter->AllocateTensors();

    // --- Image Loading ---
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    float var_yaw, var_pitch, var_roll;
    float conv_yaw, conv_pitch, conv_roll;

    run_fsanet_model(var_interpreter.get(), img, var_yaw, var_pitch, var_roll);
    run_fsanet_model(conv_interpreter.get(), img, conv_yaw, conv_pitch, conv_roll);
    
    // Average the results
    float final_yaw = (var_yaw + conv_yaw) / 2.0f;
    float final_pitch = (var_pitch + conv_pitch) / 2.0f;
    float final_roll = (var_roll + conv_roll) / 2.0f;

    const float angle_threshold = 10.0f;

    if (fabs(final_yaw) < angle_threshold && fabs(final_pitch) < angle_threshold && fabs(final_roll) < angle_threshold) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 