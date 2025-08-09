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
    const int input_w = 320, input_h = 240;

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
    cv::resize(img, resized, cv::Size(input_w, input_h));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // --- Fill input tensor ---
    float* input_ptr = interpreter->typed_input_tensor<float>(0);
    for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
            for (int c = 0; c < 3; ++c) {
                // Normalize to [-1, 1]
                input_ptr[(h * input_w + w) * 3 + c] = (resized.at<cv::Vec3b>(h, w)[c] - 127.0f) / 128.0f;
            }
        }
    }

    // --- Inference ---
    interpreter->Invoke();

    // --- Get output ---
    // UltraFace TFLite model typically has two outputs: scores and boxes
    float* scores_ptr = interpreter->typed_output_tensor<float>(0);
    const auto* scores_dims = interpreter->output_tensor(0)->dims;
    int num_proposals = scores_dims->data[1];

    int num_faces = 0;
    float score_threshold = 0.7f;
    for (int i = 0; i < num_proposals; ++i) {
        if (scores_ptr[i * 2 + 1] > score_threshold) { // index 1 is face score
            num_faces++;
        }
    }

    if (num_faces == 1) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 