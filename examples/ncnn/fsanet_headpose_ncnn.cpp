#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <net.h>

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <var_param_path> <var_bin_path> <conv_param_path> <conv_bin_path> <image_path>" << std::endl;
        return -1;
    }
    const std::string var_param_path = argv[1];
    const std::string var_bin_path = argv[2];
    const std::string conv_param_path = argv[3];
    const std::string conv_bin_path = argv[4];
    const std::string image_path = argv[5];

    // --- Load Image and Preprocess ---
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    const int input_size = 64;
    
    // 1. Padding
    const float pad = 0.3f;
    const int h = image.rows;
    const int w = image.cols;
    const int nh = static_cast<int>(static_cast<float>(h) + pad * static_cast<float>(h));
    const int nw = static_cast<int>(static_cast<float>(w) + pad * static_cast<float>(w));
    const int nx1 = std::max(0, static_cast<int>((nw - w) / 2));
    const int ny1 = std::max(0, static_cast<int>((nh - h) / 2));

    cv::Mat padded_image = cv::Mat(nh, nw, CV_8UC3, cv::Scalar(0, 0, 0));
    image.copyTo(padded_image(cv::Rect(nx1, ny1, w, h)));

    // 2. Resize
    cv::Mat resized;
    cv::resize(padded_image, resized, cv::Size(input_size, input_size));
    
    // 3. Normalize to ncnn::Mat
    ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR, input_size, input_size);
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0f/127.5f, 1.0f/127.5f, 1.0f/127.5f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // --- Var Model Inference ---
    ncnn::Net var_net;
    var_net.opt.use_vulkan_compute = false;
    if (var_net.load_param(var_param_path.c_str()) != 0 || var_net.load_model(var_bin_path.c_str()) != 0) {
        std::cerr << "Failed to load ncnn var model." << std::endl;
        return -1;
    }
    
    ncnn::Mat var_out;
    {
        ncnn::Extractor ex = var_net.create_extractor();
        ex.input("input", in);
        ex.extract("output", var_out);
    }
    float var_yaw = var_out[0];
    float var_pitch = var_out[1];
    float var_roll = var_out[2];

    // --- Conv Model Inference ---
    ncnn::Net conv_net;
    conv_net.opt.use_vulkan_compute = false;
    if (conv_net.load_param(conv_param_path.c_str()) != 0 || conv_net.load_model(conv_bin_path.c_str()) != 0) {
        std::cerr << "Failed to load ncnn conv model." << std::endl;
        return -1;
    }
    
    ncnn::Mat conv_out;
    {
        ncnn::Extractor ex = conv_net.create_extractor();
        ex.input("input", in);
        ex.extract("output", conv_out);
    }
    float conv_yaw = conv_out[0];
    float conv_pitch = conv_out[1];
    float conv_roll = conv_out[2];
    
    // Average the results
    float final_yaw = (var_yaw + conv_yaw) / 2.0f;
    float final_pitch = (var_pitch + conv_pitch) / 2.0f;
    float final_roll = (var_roll + conv_roll) / 2.0f;

    printf("DEBUG NCNN: var_yaw: %.4f, var_pitch: %.4f, var_roll: %.4f\n", var_yaw, var_pitch, var_roll);
    printf("DEBUG NCNN: conv_yaw: %.4f, conv_pitch: %.4f, conv_roll: %.4f\n", conv_yaw, conv_pitch, conv_roll);
    printf("DEBUG NCNN: final_yaw: %.4f, final_pitch: %.4f, final_roll: %.4f\n", final_yaw, final_pitch, final_roll);

    const float angle_threshold = 10.0f;
    if (fabs(final_yaw) < angle_threshold && fabs(final_pitch) < angle_threshold && fabs(final_roll) < angle_threshold) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 