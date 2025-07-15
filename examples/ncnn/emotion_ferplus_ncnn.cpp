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

    // Convert to grayscale to match other successful implementations
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    if (net.load_param(model_param.c_str()) != 0 || net.load_model(model_bin.c_str()) != 0) {
        std::cerr << "Failed to load ncnn model." << std::endl;
        return -1;
    }
    // 输入尺寸
    const int input_size = 64;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size, input_size));
    
    // Create ncnn::Mat from grayscale image, without normalization
    ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_GRAY, input_size, input_size);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("Input3", in);
    ncnn::Mat out;
    ex.extract("Plus692_Output_0", out);
    // 输出 emotion
    float* ptr = (float*)out.data;
    const float happiness_prob = ptr[1];
    const float neutral_prob = ptr[0];
    if (happiness_prob > 0.9f && neutral_prob < 0.05f) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 