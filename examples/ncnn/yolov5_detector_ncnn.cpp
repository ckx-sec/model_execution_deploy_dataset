#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <net.h>
#include <algorithm>

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// NMS implementation
static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const size_t n = objects.size();
    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++) {
        areas[i] = objects[i].rect.area();
    }
    for (size_t i = 0; i < n; i++) {
        const Object& a = objects[i];
        int keep = 1;
        for (int j : picked) {
            const Object& b = objects[j];
            float inter_area = (a.rect & b.rect).area();
            float union_area = areas[i] + areas[j] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep) picked.push_back(i);
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

    // NCNN 加载模型
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    if (net.load_param(model_param.c_str()) != 0 || net.load_model(model_bin.c_str()) != 0) {
        std::cerr << "Failed to load ncnn model." << std::endl;
        return -1;
    }

    // 输入尺寸（根据实际模型调整）
    const int target_size = 640;
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1.0f/255, 1.0f/255, 1.0f/255};

    // letterbox resize
    int w = img.cols;
    int h = img.rows;
    float scale = std::min(target_size / (w*1.f), target_size / (h*1.f));
    int new_w = w * scale;
    int new_h = h * scale;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    cv::Mat input = cv::Mat::zeros(target_size, target_size, CV_8UC3);
    resized.copyTo(input(cv::Rect(0, 0, new_w, new_h)));

    // 转 ncnn::Mat
    ncnn::Mat in = ncnn::Mat::from_pixels(input.data, ncnn::Mat::PIXEL_BGR2RGB, target_size, target_size);
    in.substract_mean_normalize(mean_vals, norm_vals);

    // 推理
    ncnn::Extractor ex = net.create_extractor();
    ex.input("images", in);
    ncnn::Mat out;
    ex.extract("pred", out);

    // 后处理
    std::vector<Object> proposals;
    const float conf_threshold = 0.25f;
    const int num_grid = out.h;
    const int num_class = out.w - 5;

    for (int i = 0; i < num_grid; i++)
    {
        const float* values = out.row(i);
        float box_confidence = values[4];
        if (box_confidence > conf_threshold)
        {
            // find max class
            int class_index = 0;
            float class_confidence = 0.f;
            for (int j = 0; j < num_class; j++)
            {
                if (values[5 + j] > class_confidence)
                {
                    class_confidence = values[5 + j];
                    class_index = j;
                }
            }

            float final_confidence = box_confidence * class_confidence;
            if (final_confidence > conf_threshold)
            {
                float cx = values[0];
                float cy = values[1];
                float box_w = values[2];
                float box_h = values[3];
                // 坐标还原
                float x0 = (cx - box_w * 0.5f) * target_size;
                float y0 = (cy - box_h * 0.5f) * target_size;
                float x1 = (cx + box_w * 0.5f) * target_size;
                float y1 = (cy + box_h * 0.5f) * target_size;
                // 反 letterbox
                x0 = (x0 - 0) / scale;
                y0 = (y0 - 0) / scale;
                x1 = (x1 - 0) / scale;
                y1 = (y1 - 0) / scale;

                x0 = std::max(std::min(x0, (float)(w - 1)), 0.f);
                y0 = std::max(std::min(y0, (float)(h - 1)), 0.f);
                x1 = std::max(std::min(x1, (float)(w - 1)), 0.f);
                y1 = std::max(std::min(y1, (float)(h - 1)), 0.f);

                Object obj;
                obj.rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
                obj.label = class_index;
                obj.prob = final_confidence;
                proposals.push_back(obj);
            }
        }
    }

    // NMS
    std::sort(proposals.begin(), proposals.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, 0.45f);

    if (!picked.empty()) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 