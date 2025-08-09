// emotion_ferplus_tnn.cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tnn/core/tnn.h>
#include <tnn/core/instance.h>
#include <tnn/core/mat.h>
#include <tnn/utils/blob_converter.h>

using namespace TNN_NS;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " proto_path model_path image_path" << std::endl;
        return -1;
    }
    std::string proto_path = argv[1];
    std::string model_path = argv[2];
    std::string image_path = argv[3];

    ModelConfig config;
    config.model_type = MODEL_TYPE_TNN;
    config.params = {proto_path, model_path};
    TNN net;
    auto status = net.Init(config);
    if (status != TNN_OK) {
        std::cout << "TNN Init failed: " << status.description() << std::endl;
        return -1;
    }

    NetworkConfig net_config;
    net_config.device_type = DEVICE_ARM;
    auto instance = net.CreateInst(net_config, status);
    if (!instance || status != TNN_OK) {
        std::cout << "CreateInst failed: " << status.description() << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cout << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(64, 64));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    DimsVector input_shape = {1, 3, 64, 64};
    std::shared_ptr<Mat> input_mat = std::make_shared<Mat>(DEVICE_ARM, NCHW_FLOAT, input_shape, resized.data);

    MatConvertParam input_param;
    input_param.scale = {1/255.f, 1/255.f, 1/255.f, 0.f};
    input_param.bias = {0.f, 0.f, 0.f, 0.f};

    status = instance->SetInputMat(input_mat, input_param);
    if (status != TNN_OK) {
        std::cout << "SetInputMat failed: " << status.description() << std::endl;
        return -1;
    }

    status = instance->Forward();
    if (status != TNN_OK) {
        std::cout << "Forward failed: " << status.description() << std::endl;
        return -1;
    }

    std::shared_ptr<Mat> output_mat;
    status = instance->GetOutputMat(output_mat);
    if (status != TNN_OK) {
        std::cout << "GetOutputMat failed: " << status.description() << std::endl;
        return -1;
    }
    float* output_data = static_cast<float*>(output_mat->GetData());
    auto output_dims = output_mat->GetDims();
    size_t output_size = output_dims.empty() ? 0 : output_dims[1];

    // Softmax to get probabilities
    std::vector<float> probs(output_size);
    if (output_size > 0) {
        float max_val = *std::max_element(output_data, output_data + output_size);
        float sum_exp = 0.0f;
        for (size_t i = 0; i < output_size; i++) {
            probs[i] = std::exp(output_data[i] - max_val);
            sum_exp += probs[i];
        }
        for (size_t i = 0; i < output_size; i++) {
            probs[i] /= sum_exp;
        }
    }
    
    // Apply standard logic
    if (output_size >= 2) { // Ensure we have at least neutral and happiness probs
        const float happiness_prob = probs[1]; // "happiness"
        const float neutral_prob = probs[0];   // "neutral"
        if (happiness_prob > 0.5f && neutral_prob < 0.4f) {
            printf("true\n");
        } else {
            printf("false\n");
        }
    } else {
        printf("false\n"); // Not enough classes in output
    }
    
    return 0;
} 