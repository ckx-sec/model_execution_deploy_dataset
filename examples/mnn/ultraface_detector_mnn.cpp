#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
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
    const float img_h = img.rows;
    const float img_w = img.cols;

    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    auto session = net->createSession(config);
    const int input_w = 320, input_h = 240;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_w, input_h));
    auto input_tensor = net->getSessionInput(session, nullptr);
    net->resizeTensor(input_tensor, {1, 3, input_h, input_w});
    net->resizeSession(session);
    
    MNN::CV::ImageProcess::Config p_config;
    p_config.sourceFormat = MNN::CV::BGR;
    p_config.destFormat = MNN::CV::RGB;
    p_config.mean[0] = 127.0f;
    p_config.mean[1] = 127.0f;
    p_config.mean[2] = 127.0f;
    p_config.normal[0] = 1.0 / 128.0f;
    p_config.normal[1] = 1.0 / 128.0f;
    p_config.normal[2] = 1.0 / 128.0f;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(p_config));
    pretreat->convert(resized.data, input_w, input_h, resized.step[0], input_tensor);

    net->runSession(session);
    auto scores_tensor = net->getSessionOutput(session, "scores");
    auto boxes_tensor = net->getSessionOutput(session, "boxes");

    MNN::Tensor scores_host(scores_tensor, scores_tensor->getDimensionType());
    scores_tensor->copyToHostTensor(&scores_host);
    MNN::Tensor boxes_host(boxes_tensor, boxes_tensor->getDimensionType());
    boxes_tensor->copyToHostTensor(&boxes_host);
    
    int num_proposals = scores_host.shape()[1];
    const float* scores_ptr = scores_host.host<float>();
    const float* boxes_ptr = boxes_host.host<float>();

    float score_threshold = 0.5f;
    float iou_threshold = 0.3f;
    
    std::vector<FaceBox> bbox_collection;
    for (int i = 0; i < num_proposals; ++i) {
        if (scores_ptr[i*2+1] > score_threshold) {
            FaceBox box;
            box.x1 = boxes_ptr[i*4+0] * img_w;
            box.y1 = boxes_ptr[i*4+1] * img_h;
            box.x2 = boxes_ptr[i*4+2] * img_w;
            box.y2 = boxes_ptr[i*4+3] * img_h;
            box.score = scores_ptr[i*2+1];
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