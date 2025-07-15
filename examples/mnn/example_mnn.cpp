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

// --- Helper Functions ---
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
    // --- Configuration ---
    const std::string mnn_path = argv[1];
    const std::string image_path = argv[2];
    const int input_width = 224;
    const int input_height = 224;

    // --- MNN setup ---
    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 1;
    auto session = net->createSession(config);

    // --- Preprocessing ---
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_width, input_height));

    // --- Set input tensor ---
    auto input = net->getSessionInput(session, nullptr);
    net->resizeTensor(input, {1, 3, input_height, input_width});
    net->resizeSession(session);

    // Use MNN's ImageProcess to handle conversion and normalization
    MNN::CV::Matrix trans;
    trans.setScale(1.0f, 1.0f); // Already resized
    MNN::CV::ImageProcess::Config p_config;
    p_config.filterType = MNN::CV::BICUBIC;
    p_config.sourceFormat = MNN::CV::BGR;
    p_config.destFormat = MNN::CV::RGB;
    p_config.mean[0] = 127.5;
    p_config.mean[1] = 127.5;
    p_config.mean[2] = 127.5;
    p_config.normal[0] = 1.0 / 128.0;
    p_config.normal[1] = 1.0 / 128.0;
    p_config.normal[2] = 1.0 / 128.0;
    
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(p_config));
    pretreat->setMatrix(trans);
    pretreat->convert(resized_image.data, input_width, input_height, resized_image.step[0], input);

    // --- Inference ---
    net->runSession(session);

    // --- Get output ---
    auto output = net->getSessionOutput(session, nullptr);
    std::shared_ptr<MNN::Tensor> output_host(new MNN::Tensor(output, output->getDimensionType()));
    output->copyToHostTensor(output_host.get());
    const float* raw_output = output_host->host<float>();
    size_t output_size = output_host->elementSize();
    auto probs = softmax(raw_output, output_size);
    auto max_it = std::max_element(probs.begin(), probs.end());
    int max_index = std::distance(probs.begin(), max_it);
    float confidence = *max_it;

    const float prob_threshold = 0.95f; // Threshold significantly increased from 0.5
    const int target_age_index = 4; // 25-32 years

    if (max_index == target_age_index && confidence > prob_threshold) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 