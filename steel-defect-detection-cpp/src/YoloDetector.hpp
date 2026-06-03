#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
    std::string class_name;
    std::string cn_name;
};

class YoloDetector {
public:
    YoloDetector();
    ~YoloDetector();

    bool init(const std::string& model_path, bool use_gpu = true);
    
    // Core detection method returning list of detections
    std::vector<Detection> detect(const cv::Mat& frame, float conf_threshold = 0.25f, float iou_threshold = 0.45f);

    const std::vector<std::string>& getClassNames() const { return class_names_; }

private:
    // Model parameters
    int input_width_ = 640;
    int input_height_ = 640;
    std::vector<std::string> class_names_;
    std::vector<std::string> cn_names_;

    // ONNX Runtime resources
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // Helper functions
    cv::Mat preprocess(const cv::Mat& frame, std::vector<float>& input_tensor_values, float& scale_factor_x, float& scale_factor_y);
    std::vector<Detection> postprocess(const std::vector<float>& output_tensor_values, const std::vector<int64_t>& output_shape, 
                                      float scale_factor_x, float scale_factor_y, float conf_threshold, float iou_threshold);
};
