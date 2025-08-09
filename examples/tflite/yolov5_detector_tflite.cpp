#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct ScaleParams {
    float r;
    int dw;
    int dh;
};

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

static void nms_sorted_bboxes(std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false) {
    picked.clear();
    const size_t n = objects.size();
    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++) {
        areas[i] = objects[i].rect.area();
    }

    for (size_t i = 0; i < n; i++) {
        const Object& a = objects[i];
        int keep = 1;
        for (size_t j = 0; j < picked.size(); j++) {
            const Object& b = objects[picked[j]];
            if (!agnostic && a.label != b.label) {
                continue;
            }
            // intersection over union
            cv::Rect_<float> inter = a.rect & b.rect;
            float inter_area = inter.area();
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float iou = inter_area / union_area;
            if (iou > nms_threshold)
            {
                keep = 0;
                break;
            }
        }

        if (keep) {
            picked.push_back(i);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }
    const std::string tflite_path = argv[1];
    const std::string image_path = argv[2];
    const int target_size = 640;
    const int num_classes = 80;

    auto model = tflite::FlatBufferModel::BuildFromFile(tflite_path.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    interpreter->AllocateTensors();

    cv::Mat img = cv::imread(image_path);
    ScaleParams scale_params;
    cv::Mat letterboxed_image;
    letterbox(img, letterboxed_image, scale_params, target_size, target_size);

    float* input_ptr = interpreter->typed_input_tensor<float>(0);
    letterboxed_image.convertTo(letterboxed_image, CV_32F, 1.0 / 255.0);
    memcpy(input_ptr, letterboxed_image.data, target_size * target_size * 3 * sizeof(float));
    
    interpreter->Invoke();

    const float* raw_output = interpreter->typed_output_tensor<float>(0);
    const auto* output_dims = interpreter->output_tensor(0)->dims;
    const int num_proposals = output_dims->data[1];
    const int proposal_length = output_dims->data[2];
    const float conf_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    std::vector<Object> proposals;
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
        float w_box = proposal[2];
        float h_box = proposal[3];

        float x1 = (cx - 0.5f * w_box - scale_params.dw) / scale_params.r;
        float y1 = (cy - 0.5f * h_box - scale_params.dh) / scale_params.r;
        float x2 = (cx + 0.5f * w_box - scale_params.dw) / scale_params.r;
        float y2 = (cy + 0.5f * h_box - scale_params.dh) / scale_params.r;
        
        Object obj;
        obj.rect = cv::Rect_<float>(x1, y1, x2 - x1, y2 - y1);
        obj.label = class_id;
        obj.prob = conf * class_score;
        proposals.push_back(obj);
    }
    
    std::sort(proposals.begin(), proposals.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    
    if (picked.size() > 0) {
        printf("DEBUG TFLITE: Detected %zu objects. Top detection: Label %d, Score %.4f\n", 
               picked.size(), proposals[picked[0]].label, proposals[picked[0]].prob);
    }

    if (picked.size() > 0 && proposals[picked[0]].prob > 0.5f) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    return 0;
} 