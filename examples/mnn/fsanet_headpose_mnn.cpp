#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>

// Helper function to run inference on a single model
void run_fsanet_model(
    MNN::Interpreter* net,
    const cv::Mat& img,
    float& yaw, float& pitch, float& roll)
{
    // --- Session and tensor setup ---
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    auto session = net->createSession(config);
    auto input_tensor = net->getSessionInput(session, nullptr);
    const int input_size = 64;

    // --- Preprocessing ---
    // 1. Padding
    const float pad = 0.3f;
    const int h = img.rows;
    const int w = img.cols;
    const int nh = static_cast<int>(static_cast<float>(h) + pad * static_cast<float>(h));
    const int nw = static_cast<int>(static_cast<float>(w) + pad * static_cast<float>(w));
    const int nx1 = std::max(0, static_cast<int>((nw - w) / 2));
    const int ny1 = std::max(0, static_cast<int>((nh - h) / 2));

    cv::Mat padded_image = cv::Mat(nh, nw, CV_8UC3, cv::Scalar(0, 0, 0));
    img.copyTo(padded_image(cv::Rect(nx1, ny1, w, h)));
    
    // 2. Resize
    cv::Mat resized;
    cv::resize(padded_image, resized, cv::Size(input_size, input_size));

    net->resizeTensor(input_tensor, {1, 3, input_size, input_size});
    net->resizeSession(session);

    // 3. Normalize
    MNN::CV::ImageProcess::Config p_config;
    p_config.sourceFormat = MNN::CV::BGR;
    p_config.destFormat = MNN::CV::BGR;
    p_config.mean[0] = 127.5f;
    p_config.mean[1] = 127.5f;
    p_config.mean[2] = 127.5f;
    p_config.normal[0] = 1.0 / 127.5f;
    p_config.normal[1] = 1.0 / 127.5f;
    p_config.normal[2] = 1.0 / 127.5f;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(p_config));
    pretreat->convert(resized.data, input_size, input_size, resized.step[0], input_tensor);

    // --- Inference ---
    net->runSession(session);

    // --- Post-processing ---
    auto output_tensor = net->getSessionOutput(session, nullptr);
    MNN::Tensor output_host(output_tensor, output_tensor->getDimensionType());
    output_tensor->copyToHostTensor(&output_host);
    const float* outptr = output_host.host<float>();

    // The ONNX model seems to have scaling built-in. Assume MNN model is the same.
    yaw = outptr[0];
    pitch = outptr[1];
    roll = outptr[2];
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <var_model_path> <1x1_model_path> <image_path>" << std::endl;
        return -1;
    }
    std::string var_model_path = argv[1];
    std::string conv_model_path = argv[2];
    std::string image_path = argv[3];

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    auto var_net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(var_model_path.c_str()));
    auto conv_net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(conv_model_path.c_str()));

    float var_yaw, var_pitch, var_roll;
    float conv_yaw, conv_pitch, conv_roll;

    run_fsanet_model(var_net.get(), img, var_yaw, var_pitch, var_roll);
    run_fsanet_model(conv_net.get(), img, conv_yaw, conv_pitch, conv_roll);
    
    // Average the results
    float final_yaw = (var_yaw + conv_yaw) / 2.0f;
    float final_pitch = (var_pitch + conv_pitch) / 2.0f;
    float final_roll = (var_roll + conv_roll) / 2.0f;

    const float angle_threshold = 10.0f;

    if (fabs(final_yaw) < angle_threshold && fabs(final_pitch) < angle_threshold && fabs(final_roll) < angle_threshold) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 