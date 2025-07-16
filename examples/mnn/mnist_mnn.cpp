#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

// Softmax function
template <typename T>
static void softmax(T& input) {
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
    // 1. Configuration
    std::string model_path = argv[1];
    std::string image_path = argv[2]; 

    // 2. Load and Preprocess Image
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(28, 28));
    
    // 3. MNN Session Setup
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    auto session = net->createSession(config);

    // 4. Define and Fill Input Tensor
    auto input_tensor = net->getSessionInput(session, nullptr);
    std::vector<int> dims{1, 1, 28, 28};
    net->resizeTensor(input_tensor, dims);
    net->resizeSession(session);
    
    std::vector<float> input_tensor_values(28 * 28);
    for (int i = 0; i < resized.rows; i++) {
        for (int j = 0; j < resized.cols; j++) {
            // Normalize pixel values to [0, 1] and handle potential color inversion needed by model
            input_tensor_values[i * resized.cols + j] = resized.at<uchar>(i, j) / 255.0f;
        }
    }
    // Create a temporary host tensor and copy data to the device tensor
    auto tmp_input_tensor = MNN::Tensor(input_tensor, MNN::Tensor::CAFFE);
    memcpy(tmp_input_tensor.host<float>(), input_tensor_values.data(), input_tensor_values.size() * sizeof(float));
    input_tensor->copyFromHostTensor(&tmp_input_tensor);

    // 5. Run Inference
    net->runSession(session);

    // 6. Post-process and Print Results
    auto output_tensor = net->getSessionOutput(session, nullptr);
    MNN::Tensor output_host(output_tensor, output_tensor->getDimensionType());
    output_tensor->copyToHostTensor(&output_host);
    
    // The data is now on the host and can be accessed
    auto outptr = output_host.host<float>();
    std::vector<float> results(outptr, outptr + 10); // There are 10 classes for MNIST

    softmax(results);
    
    auto max_it = std::max_element(results.begin(), results.end());
    int64_t predicted_digit = std::distance(results.begin(), max_it);
    float confidence = *max_it;

    const float prob_threshold = 0.7f;
    const int target_digit = 7;

    // New business logic: return true if predicted digit is 7 with high confidence
    if (predicted_digit == target_digit && confidence > prob_threshold) {
        printf("true\n");
    } else {
        printf("false\n");
    }

    return 0;
} 