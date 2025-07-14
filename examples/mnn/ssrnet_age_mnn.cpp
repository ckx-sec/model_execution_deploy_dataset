#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>

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
    const int input_size = 64;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size, input_size));
    auto input_tensor = net->getSessionInput(session, nullptr);
    net->resizeTensor(input_tensor, {1, 3, input_size, input_size});
    net->resizeSession(session);

    MNN::CV::ImageProcess::Config p_config;
    p_config.sourceFormat = MNN::CV::BGR;
    p_config.destFormat = MNN::CV::RGB;
    p_config.mean[0] = 127.5f;
    p_config.mean[1] = 127.5f;
    p_config.mean[2] = 127.5f;
    p_config.normal[0] = 1.0 / 128.0f;
    p_config.normal[1] = 1.0 / 128.0f;
    p_config.normal[2] = 1.0 / 128.0f;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(p_config));
    pretreat->convert(resized.data, input_size, input_size, resized.step[0], input_tensor);

    net->runSession(session);
    auto output_tensor = net->getSessionOutput(session, nullptr);
    MNN::Tensor output_host(output_tensor, output_tensor->getDimensionType());
    output_tensor->copyToHostTensor(&output_host);
    const float* outptr = output_host.host<float>();
    float predicted_age = outptr[0];

    if (predicted_age >= 20.0f && predicted_age <= 40.0f) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 