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

    // --- Print model input/output details ---
    std::cout << "--- TFLite Model ---" << std::endl;
    std::cout << "Input tensors: " << interpreter->inputs().size() << std::endl;
    for (size_t i = 0; i < interpreter->inputs().size(); ++i) {
        auto* tensor = interpreter->input_tensor(i);
        std::cout << "Input " << i << ": name=" << tensor->name;
        std::cout << ", dims=[";
        for (int d = 0; d < tensor->dims->size; ++d) {
            std::cout << tensor->dims->data[d] << (d < tensor->dims->size - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "Output tensors: " << interpreter->outputs().size() << std::endl;
    for (size_t i = 0; i < interpreter->outputs().size(); ++i) {
        auto* tensor = interpreter->output_tensor(i);
        std::cout << "Output " << i << ": name=" << tensor->name;
        std::cout << ", dims=[";
        for (int d = 0; d < tensor->dims->size; ++d) {
            std::cout << tensor->dims->data[d] << (d < tensor->dims->size - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "--------------------" << std::endl;

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

    // --- Post-processing ---
    float* raw_output = interpreter->typed_output_tensor<float>(1);
    const auto* output_dims = interpreter->output_tensor(1)->dims;
    // The landmark tensor shape is [1, 212], so we take the second dimension.
    size_t num_landmarks = output_dims->data[1]; // Total number of float values (106 landmarks * 2 coords)

    printf("DEBUG TFLITE: All landmarks (x, y):\n");
    for (size_t i = 0; i < num_landmarks; i += 2) {
        if (i < 10 || i > num_landmarks - 12) { // Print first 5 and last 5 landmarks
             printf("  - Landmark %zu: (%.4f, %.4f)\n", i/2, raw_output[i], raw_output[i+1]);
        }
    }

    bool valid = true;
    // Align with MNN version: scale coordinates to pixels before checking boundaries
    float margin_x = img.cols * 0.02f;
    float margin_y = img.rows * 0.02f;

    for (size_t i = 0; i < num_landmarks; i += 2) {
        // First, scale the normalized coordinates to pixel coordinates
        float x = raw_output[i] * img.cols;
        float y = raw_output[i+1] * img.rows;

        // Then, check boundaries on the pixel coordinates
        if (x < margin_x || x >= img.cols - margin_x || y < margin_y || y >= img.rows - margin_y) {
            printf("DEBUG: Landmark %zu failed: pixel coords (x=%.4f, y=%.4f)\n", i / 2, x, y);
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