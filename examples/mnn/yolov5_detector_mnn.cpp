#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <algorithm>

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static void nms_sorted_bboxes(std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
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
        for (size_t j = 0; j < picked.size(); j++) {
            const Object& b = objects[picked[j]];
            if (!agnostic && a.label != b.label) {
                continue;
            }
            // intersection over union
            cv::Rect_<float> inter = a.rect & b.rect;
            float inter_area = inter.area();
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float iou = inter_area / union_area;
            if (iou > nms_threshold)
            {
                keep = 0;
                break;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }
}


int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    // MNN 加载模型
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    auto session = net->createSession(config);
    // 输入尺寸
    const int target_size = 640;
    int w = img.cols, h = img.rows;
    float scale = std::min(target_size / (w*1.f), target_size / (h*1.f));
    int new_w = w * scale;
    int new_h = h * scale;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    cv::Mat input_mat = cv::Mat::zeros(target_size, target_size, CV_8UC3);
    resized.copyTo(input_mat(cv::Rect(0, 0, new_w, new_h)));
    // 填充 MNN 输入
    auto input_tensor = net->getSessionInput(session, nullptr);
    net->resizeTensor(input_tensor, {1, 3, target_size, target_size});
    net->resizeSession(session);

    MNN::CV::ImageProcess::Config p_config;
    p_config.sourceFormat = MNN::CV::BGR;
    p_config.destFormat = MNN::CV::RGB;
    p_config.filterType = MNN::CV::BICUBIC;
    p_config.normal[0] = 1.0 / 255.0f;
    p_config.normal[1] = 1.0 / 255.0f;
    p_config.normal[2] = 1.0 / 255.0f;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(p_config));
    pretreat->convert(input_mat.data, target_size, target_size, input_mat.step[0], input_tensor);

    net->runSession(session);
    auto output_tensor = net->getSessionOutput(session, "pred");
    if (output_tensor == nullptr) {
        std::cerr << "Failed to get output tensor: pred" << std::endl;
        return -1;
    }
    MNN::Tensor output_host(output_tensor, output_tensor->getDimensionType());
    output_tensor->copyToHostTensor(&output_host);
    
    std::vector<Object> proposals;
    const float* outptr = output_host.host<float>();
    int num_proposal = output_host.shape()[1];
    int num_class = output_host.shape()[2] - 5;
    
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;

    for (int i = 0; i < num_proposal; i++)
    {
        const float* p = outptr + i * (num_class + 5);
        float box_score = p[4];
        if (box_score > conf_threshold)
        {
            int class_idx = std::max_element(p + 5, p + 5 + num_class) - (p + 5);
            float class_score = p[5 + class_idx];
            float confidence = box_score * class_score;

            if (confidence > conf_threshold)
            {
                float cx = p[0];
                float cy = p[1];
                float ww = p[2];
                float hh = p[3];

                float x0 = (cx - ww * 0.5f) / scale;
                float y0 = (cy - hh * 0.5f) / scale;
                float x1 = (cx + ww * 0.5f) / scale;
                float y1 = (cy + hh * 0.5f) / scale;
                
                Object obj;
                obj.rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
                obj.label = class_idx;
                obj.prob = confidence;
                proposals.push_back(obj);
            }
        }
    }
    std::sort(proposals.begin(), proposals.end(), [](const Object& a, const Object& b) {
        return a.prob > b.prob;
    });

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    
    if (picked.size() > 0) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 