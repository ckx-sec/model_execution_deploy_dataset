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
    // --- Configuration ---
    const std::string model_param = argv[1];
    const std::string model_bin = argv[2];
    const std::string image_path = argv[3];
    const int input_width = 224;
    const int input_height = 224;

    // --- NCNN setup ---
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    if (net.load_param(model_param.c_str()) != 0 || net.load_model(model_bin.c_str()) != 0) {
        std::cerr << "Failed to load ncnn model." << std::endl;
        return -1;
    }

    // --- Preprocessing ---
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_width, input_height));
    
    ncnn::Mat in = ncnn::Mat::from_pixels(resized_image.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0f/128.0f, 1.0f/128.0f, 1.0f/128.0f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // --- Inference ---
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("loss3/loss3_Y", out);
    
    // --- Get output ---
    float* raw_output = (float*)out.data;
    size_t output_size = out.w;

    auto probs = softmax(raw_output, output_size);

    auto max_it = std::max_element(probs.begin(), probs.end());
    int max_index = std::distance(probs.begin(), max_it);
    float confidence = *max_it;

    printf("DEBUG NCNN: Age Bracket[%d], Confidence: %.4f\n", max_index, confidence);
    for(size_t i = 0; i < probs.size(); ++i) {
        printf("  - Prob[%zu]: %.4f\n", i, probs[i]);
    }

    const float prob_threshold = 0.7f;
    const int target_age_index = 4; // 25-32 years

    if (max_index == target_age_index && confidence > prob_threshold) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 