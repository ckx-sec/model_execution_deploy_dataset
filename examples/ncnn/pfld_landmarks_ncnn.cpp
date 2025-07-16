#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <net.h>

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
    const int input_size = 112;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size, input_size));
    ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR2RGB, input_size, input_size);
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0f/128, 1.0f/128, 1.0f/128};
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("output", out);
    // 输出 landmarks
    bool valid = true;
    float margin_x = img.cols * 0.02f;
    float margin_y = img.rows * 0.02f;
    for (int i = 0; i < 106; ++i) {
        float x = out[2*i] * img.cols;
        float y = out[2*i+1] * img.rows;
        if (x < margin_x || x >= img.cols - margin_x || y < margin_y || y >= img.rows - margin_y) {
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