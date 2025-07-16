#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <net.h>

// A simple softmax implementation
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
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <param_path> <bin_path> <image_path>" << std::endl;
        return -1;
    }
    std::string model_param = argv[1];
    std::string model_bin = argv[2];
    std::string image_path = argv[3];
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    if (net.load_param(model_param.c_str()) != 0 || net.load_model(model_bin.c_str()) != 0) {
        std::cerr << "Failed to load ncnn model." << std::endl;
        return -1;
    }
    // 输入尺寸
    const int input_size = 224;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size, input_size));
    ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR2RGB, input_size, input_size);
    // Revert to original normalization: scale to [-1, 1]
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0f/128, 1.0f/128, 1.0f/128};
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("loss3/loss3_Y", out);
    // 输出 gender
    float* raw_output = (float*)out.data;
    size_t output_size = out.w;

    auto probs = softmax(raw_output, output_size);

    auto max_it = std::max_element(probs.begin(), probs.end());
    int predicted_gender_index = std::distance(probs.begin(), max_it);
    const int target_gender_index = 1; // "Female"

    if (predicted_gender_index == target_gender_index) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 