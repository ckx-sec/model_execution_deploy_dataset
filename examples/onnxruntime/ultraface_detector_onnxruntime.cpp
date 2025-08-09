#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// --- Data Structures ---
struct Box {
    float x1, y1, x2, y2;
    float score;
    int label;
};

// --- Helper Functions ---

// NMS (Non-Maximum Suppression)
void nms(std::vector<Box>& boxes, std::vector<Box>& output, float iou_threshold) {
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
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


// --- Main Inference Logic ---
int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    // --- Configuration ---
    const std::string onnx_path = argv[1];
    const std::string image_path = argv[2];
    const int input_width = 320;
    const int input_height = 240;
    const float score_threshold = 0.5f;
    const float iou_threshold = 0.3f;

    // --- ONNXRuntime setup ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test-ultraface");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, onnx_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // --- Get model input/output details ---
    std::vector<const char*> input_node_names = {"input"};
    std::vector<const char*> output_node_names = {"scores", "boxes"};

    std::cout << "--- Model Inputs ---" << std::endl;
    std::cout << "Input 0 : name=" << input_node_names[0] << std::endl;
    std::cout << "--- Model Outputs ---" << std::endl;
    std::cout << "Output 0: name=" << output_node_names[0] << std::endl;
    std::cout << "Output 1: name=" << output_node_names[1] << std::endl;
    std::cout << "--------------------" << std::endl;

    // --- Preprocessing ---
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    const float img_height = static_cast<float>(image.rows);
    const float img_width = static_cast<float>(image.cols);

    cv::Mat resized_image;
    cv::cvtColor(image, resized_image, cv::COLOR_BGR2RGB);
    cv::resize(resized_image, resized_image, cv::Size(input_width, input_height));

    std::vector<float> input_tensor_values(1 * 3 * input_height * input_width);
    resized_image.convertTo(resized_image, CV_32F);

    // Normalize and HWC to CHW
    const float mean_val = 127.0f;
    const float scale_val = 1.0f / 128.0f;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_height; ++h) {
            for (int w = 0; w < input_width; ++w) {
                input_tensor_values[c * input_height * input_width + h * input_width + w] =
                    (resized_image.at<cv::Vec3f>(h, w)[c] - mean_val) * scale_val;
            }
        }
    }

    // --- Create Tensor ---
    std::vector<int64_t> input_node_dims = {1, 3, input_height, input_width};
    size_t input_tensor_size = 1 * 3 * input_height * input_width;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size()
    );

    // --- Inference ---
    std::cout << "Running inference..." << std::endl;
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2
    );
    std::cout << "Inference finished." << std::endl;

    // --- Post-processing ---
    const float* scores_data = output_tensors[0].GetTensorData<float>();
    const float* boxes_data = output_tensors[1].GetTensorData<float>();

    auto scores_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const int num_anchors = scores_shape[1];

    std::vector<Box> bbox_collection;
    for (int i = 0; i < num_anchors; ++i) {
        float confidence = scores_data[i * 2 + 1];
        if (confidence < score_threshold) continue;

        Box box;
        box.x1 = boxes_data[i * 4 + 0] * img_width;
        box.y1 = boxes_data[i * 4 + 1] * img_height;
        box.x2 = boxes_data[i * 4 + 2] * img_width;
        box.y2 = boxes_data[i * 4 + 3] * img_height;
        box.score = confidence;
        box.label = 1; // "face"
        bbox_collection.push_back(box);
    }
    
    // NMS
    std::vector<Box> detected_boxes;
    nms(bbox_collection, detected_boxes, iou_threshold);

    if (detected_boxes.size() > 0) {
        printf("DEBUG ONNX: Detected %zu faces. Top detection score: %.4f\n", detected_boxes.size(), detected_boxes[0].score);
    }

    if (detected_boxes.size() > 0 && detected_boxes[0].score > 0.5f) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 