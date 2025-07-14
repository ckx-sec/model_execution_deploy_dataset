#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>

struct FaceBox {
    float x1, y1, x2, y2, score;
};

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
    
    int num_faces = 0;
    int num_proposals = scores_host.shape()[1];
    const float* scores_ptr = scores_host.host<float>();
    float score_threshold = 0.7f;

    for (int i = 0; i < num_proposals; ++i) {
        if (scores_ptr[i*2+1] > score_threshold) { // index 1 is face score
            num_faces++;
        }
    }

    if (num_faces > 0) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 