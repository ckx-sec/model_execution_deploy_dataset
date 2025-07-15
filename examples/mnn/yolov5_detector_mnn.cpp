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
    // 假设输出格式为 N x 6: label, prob, x, y, w, h
    std::vector<Object> proposals;
    int num = output_host.shape()[1];
    const float* outptr = output_host.host<float>();
    for (int i = 0; i < num; ++i) {
        int label = static_cast<int>(outptr[i*6+0]);
        float prob = outptr[i*6+1];
        float x = outptr[i*6+2];
        float y = outptr[i*6+3];
        float w_box = outptr[i*6+4];
        float h_box = outptr[i*6+5];
        float x0 = (x - w_box/2.f) * target_size;
        float y0 = (y - h_box/2.f) * target_size;
        float x1 = (x + w_box/2.f) * target_size;
        float y1 = (y + h_box/2.f) * target_size;
        x0 = std::max((x0 - 0) / scale, 0.f);
        y0 = std::max((y0 - 0) / scale, 0.f);
        x1 = std::min((x1 - 0) / scale, (float)w);
        y1 = std::min((y1 - 0) / scale, (float)h);
        Object obj;
        obj.label = label;
        obj.prob = prob;
        obj.rect = cv::Rect_<float>(x0, y0, x1-x0, y1-y0);
        if (prob > 0.7f) proposals.push_back(obj); // Threshold increased from 0.25
    }
    std::sort(proposals.begin(), proposals.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, 0.45f);
    
    // Stricter condition: exactly one object must be detected after NMS
    if (picked.size() == 1) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 