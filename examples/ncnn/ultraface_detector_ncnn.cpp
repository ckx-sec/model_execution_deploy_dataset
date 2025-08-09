#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <net.h>
#include <algorithm>

struct FaceBox {
    float x1, y1, x2, y2, score;
};

// NMS (Non-Maximum Suppression)
static void nms(std::vector<FaceBox>& boxes, std::vector<FaceBox>& output, float iou_threshold) {
    std::sort(boxes.begin(), boxes.end(), [](const FaceBox& a, const FaceBox& b) {
        return a.score > b.score;
    });

    std::vector<bool> is_suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (is_suppressed[i]) {
            continue;
        }
        output.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (is_suppressed[j]) {
                continue;
            }

            float inter_x1 = std::max(boxes[i].x1, boxes[j].x1);
            float inter_y1 = std::max(boxes[i].y1, boxes[j].y1);
            float inter_x2 = std::min(boxes[i].x2, boxes[j].x2);
            float inter_y2 = std::min(boxes[i].y2, boxes[j].y2);

            float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
            float area1 = (boxes[i].x2 - boxes[i].x1) * (boxes[i].y2 - boxes[i].y1);
            float area2 = (boxes[j].x2 - boxes[j].x1) * (boxes[j].y2 - boxes[j].y1);
            float union_area = area1 + area2 - inter_area;

            if (union_area > 0 && inter_area / union_area > iou_threshold) {
                is_suppressed[j] = true;
            }
        }
    }
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
    const float img_h = img.rows;
    const float img_w = img.cols;

    // NCNN 加载模型
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
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


    // 后处理
    float score_threshold = 0.5f;
    float iou_threshold = 0.3f;
    std::vector<FaceBox> bbox_collection;
    for (int i = 0; i < scores.h; ++i) {
        const float* scores_ptr = scores.row(i);
        const float* boxes_ptr = boxes.row(i);
        if (scores_ptr[1] > score_threshold) {
            FaceBox box;
            box.x1 = boxes_ptr[0] * img_w;
            box.y1 = boxes_ptr[1] * img_h;
            box.x2 = boxes_ptr[2] * img_w;
            box.y2 = boxes_ptr[3] * img_h;
            box.score = scores_ptr[1];
            bbox_collection.push_back(box);
        }
    }

    std::vector<FaceBox> detected_boxes;
    nms(bbox_collection, detected_boxes, iou_threshold);
    
    if (detected_boxes.size() > 0) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 