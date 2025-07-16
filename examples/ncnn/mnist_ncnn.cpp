#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <net.h>

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

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <param_path> <bin_path> <image_path>" << std::endl;
        return -1;
    }
    // 1. Configuration
    std::string model_param = argv[1];
    std::string model_bin = argv[2];
    std::string image_path = argv[3];
    
    // 2. Load and Preprocess Image
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(28, 28));

    // 3. NCNN Session Setup
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    if (net.load_param(model_param.c_str()) != 0 || net.load_model(model_bin.c_str()) != 0) {
        std::cerr << "Failed to load ncnn model." << std::endl;
        return -1;
    }
    
    // 4. Define and Fill Input Tensor
    ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_GRAY, 28, 28);
    const float norm_vals[1] = {1.0f / 255.0f};
    in.substract_mean_normalize(0, norm_vals);

    // 5. Run Inference
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in); // Assuming input blob name is "input"
    ncnn::Mat out;
    ex.extract("output", out); // Assuming output blob name is "output"
    
    // 6. Post-process and Print Results
    std::vector<float> results;
    results.resize(out.w);
    for (int i=0; i<out.w; i++) {
        results[i] = out[i];
    }
    
    softmax(results);
    
    auto max_it = std::max_element(results.begin(), results.end());
    int64_t predicted_digit = std::distance(results.begin(), max_it);
    float confidence = *max_it;

    const float prob_threshold = 0.7f;
    const int target_digit = 7;

    if (predicted_digit == target_digit && confidence > prob_threshold) {
        printf("true\n");
    } else {
        printf("false\n");
    }

    return 0;
} 