#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <net.h>

struct FaceBox {
    float x1, y1, x2, y2, score;
};

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

    // NCNN 加载模型
    ncnn::Net net;
    if (net.load_param(model_param.c_str()) != 0 || net.load_model(model_bin.c_str()) != 0) {
        std::cerr << "Failed to load ncnn model." << std::endl;
        return -1;
    }

    // UltraFace 输入尺寸
    const int target_w = 320;
    const int target_h = 240;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(target_w, target_h));

    // 转 ncnn::Mat
    ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR2RGB, target_w, target_h);
    // 归一化
    const float mean_vals[3] = {127.0f, 127.0f, 127.0f};
    const float norm_vals[3] = {1.0f/128, 1.0f/128, 1.0f/128};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // 推理
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in);
    ncnn::Mat scores, boxes;
    ex.extract("scores", scores);
    ex.extract("boxes", boxes);


    // 后处理（假设输出格式为 N x 6: x1, y1, x2, y2, score, label）
    int num_faces = 0;
    float score_threshold = 0.95f;
    for (int i = 0; i < scores.h; ++i) {
        const float* values = scores.row(i);
        float conf = values[1];
        if (conf > score_threshold) {
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