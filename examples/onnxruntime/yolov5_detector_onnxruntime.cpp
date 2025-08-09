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

struct ScaleParams {
    float r;
    int dw;
    int dh;
};

// --- Helper Functions ---

void letterbox(const cv::Mat& image, cv::Mat& out_image, ScaleParams& params, int new_shape_w, int new_shape_h) {
    int width = image.cols;
    int height = image.rows;

    params.r = std::min((float)new_shape_h / height, (float)new_shape_w / width);
    int new_unpad_w = (int)round(width * params.r);
    int new_unpad_h = (int)round(height * params.r);
    
    params.dw = (new_shape_w - new_unpad_w) / 2;
    params.dh = (new_shape_h - new_unpad_h) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_unpad_w, new_unpad_h));

    out_image = cv::Mat(new_shape_h, new_shape_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out_image(cv::Rect(params.dw, params.dh, new_unpad_w, new_unpad_h)));
}

void nms(std::vector<Box>& boxes, std::vector<Box>& output, float iou_threshold) {
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.score > b.score;
    });

    std::vector<bool> is_suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (is_suppressed[i]) continue;
        output.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (is_suppressed[j]) continue;

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
    const int input_width = 640;
    const int input_height = 640;
    const float conf_threshold = 0.25f;
    const float iou_threshold = 0.45f;
    const int num_classes = 80;

    // --- ONNXRuntime setup ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test-yolov5");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, onnx_path.c_str(), session_options);

    // --- Preprocessing ---
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }

    ScaleParams scale_params;
    cv::Mat letterboxed_image;
    letterbox(image, letterboxed_image, scale_params, input_width, input_height);
    
    cv::Mat rgb_image;
    cv::cvtColor(letterboxed_image, rgb_image, cv::COLOR_BGR2RGB);
    
    std::vector<float> input_tensor_values(1 * 3 * input_height * input_width);
    rgb_image.convertTo(rgb_image, CV_32F, 1.0 / 255.0);
    
    // HWC to CHW
    for (int i = 0; i < input_height * input_width; i++) {
        input_tensor_values[i] = rgb_image.at<cv::Vec3f>(i)[0];
        input_tensor_values[i + input_height * input_width] = rgb_image.at<cv::Vec3f>(i)[1];
        input_tensor_values[i + 2 * input_height * input_width] = rgb_image.at<cv::Vec3f>(i)[2];
    }
    
    // --- Create Tensor ---
    std::vector<int64_t> input_node_dims = {1, 3, input_height, input_width};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_node_dims.data(), input_node_dims.size()
    );

    // --- Inference ---
    std::vector<const char*> input_node_names = {"images"};
    std::vector<const char*> output_node_names = {"pred"};
    std::cout << "Running inference..." << std::endl;
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1
    );

    // --- Post-processing ---
    const float* raw_output = output_tensors[0].GetTensorData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const int num_proposals = output_shape[1];
    const int proposal_length = output_shape[2]; // 85

    std::vector<Box> bbox_collection;
    for (int i = 0; i < num_proposals; ++i) {
        const float* proposal = raw_output + i * proposal_length;
        float conf = proposal[4];
        if (conf < conf_threshold) continue;

        const float* class_scores = proposal + 5;
        int class_id = std::distance(class_scores, std::max_element(class_scores, class_scores + num_classes));
        float class_score = class_scores[class_id];

        if (class_score * conf < conf_threshold) continue;
        
        float cx = proposal[0];
        float cy = proposal[1];
        float w = proposal[2];
        float h = proposal[3];

        Box box;
        box.x1 = (cx - 0.5f * w - scale_params.dw) / scale_params.r;
        box.y1 = (cy - 0.5f * h - scale_params.dh) / scale_params.r;
        box.x2 = (cx + 0.5f * w - scale_params.dw) / scale_params.r;
        box.y2 = (cy + 0.5f * h - scale_params.dh) / scale_params.r;
        box.score = conf * class_score;
        box.label = class_id;
        bbox_collection.push_back(box);
    }
    
    std::vector<Box> detected_boxes;
    nms(bbox_collection, detected_boxes, iou_threshold);

    if (detected_boxes.size() > 0) {
        printf("DEBUG ONNX: Detected %zu objects. Top detection: Label %d, Score %.4f\n", 
               detected_boxes.size(), detected_boxes[0].label, detected_boxes[0].score);
    }

    if (detected_boxes.size() > 0 && detected_boxes[0].score > 0.5f) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
}