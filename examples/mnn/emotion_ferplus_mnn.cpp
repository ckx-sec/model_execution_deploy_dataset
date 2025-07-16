#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>

// A simple softmax implementation
template <typename T>
std::vector<T> softmax(const T* data, size_t size) {
    std::vector<T> result(size);
    const T max_val = *std::max_element(data, data + size);
    T sum = 0;
    for (size_t i = 0; i < size; ++i) {
        result[i] = std::exp(data[i] - max_val);
        sum += result[i];
    }
    for (size_t i = 0; i < size; ++i) {
        result[i] /= sum;
    }
    return result;
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
    const float* raw_output = output_host.host<float>();
    size_t output_size = output_host.elementSize();

    auto probs = softmax(raw_output, output_size);

    const int happiness_index = 1; // "happiness"
    const int neutral_index = 0;   // "neutral"
    const float happiness_prob = probs[happiness_index];
    const float neutral_prob = probs[neutral_index];

    // Condition significantly narrowed: high happiness AND low neutral
    if (happiness_prob > 0.7f && neutral_prob < 0.3f) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 