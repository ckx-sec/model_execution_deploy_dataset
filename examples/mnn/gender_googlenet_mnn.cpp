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
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    auto session = net->createSession(config);
    const int input_size = 224;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size, input_size));
    auto input_tensor = net->getSessionInput(session, nullptr);
    net->resizeTensor(input_tensor, {1, 3, input_size, input_size});
    net->resizeSession(session);

    MNN::CV::ImageProcess::Config p_config;
    p_config.sourceFormat = MNN::CV::BGR;
    p_config.destFormat = MNN::CV::RGB;
    // Revert to original normalization: scale to [-1, 1]
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
    const float* raw_output = output_host.host<float>();
    size_t output_size = output_host.elementSize();

    auto probs = softmax(raw_output, output_size);

    auto max_it = std::max_element(probs.begin(), probs.end());
    int predicted_gender_index = std::distance(probs.begin(), max_it);
    const int target_gender_index = 1; // "Female"

    printf("DEBUG MNN: Predicted Gender Index: %d, Probs: [F: %.4f, M: %.4f]\n", 
           predicted_gender_index, probs[1], probs[0]);

    if (predicted_gender_index == target_gender_index) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 