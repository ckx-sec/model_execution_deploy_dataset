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

    // Gray scale as the model expects
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    auto session = net->createSession(config);
    const int input_size = 64;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size, input_size));
    
    auto input_tensor = net->getSessionInput(session, nullptr);
    // NCHW
    net->resizeTensor(input_tensor, {1, 1, input_size, input_size});
    net->resizeSession(session);

    MNN::CV::ImageProcess::Config p_config;
    p_config.sourceFormat = MNN::CV::GRAY;
    p_config.destFormat = MNN::CV::GRAY;

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
        MNN::CV::ImageProcess::create(p_config)
    );
    pretreat->convert(resized.data, input_size, input_size, resized.step[0], input_tensor);

    net->runSession(session);
    auto output_tensor = net->getSessionOutput(session, nullptr);
    MNN::Tensor output_host(output_tensor, output_tensor->getDimensionType());
    output_tensor->copyToHostTensor(&output_host);
    const float* outptr = output_host.host<float>();
    int max_index = std::max_element(outptr, outptr + 8) - outptr;
    float max_prob = outptr[max_index];

    const float prob_threshold = 0.5f;
    const int target_emotion_index = 1; // "happiness"

    if (max_index == target_emotion_index && max_prob > prob_threshold) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 