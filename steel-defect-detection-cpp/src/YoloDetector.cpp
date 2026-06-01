#include "YoloDetector.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>

YoloDetector::YoloDetector() 
    : env_(ORT_LOGGING_LEVEL_WARNING, "YoloDetector") {
    
    // Class definitions matching Python/config specifications
    class_names_ = { "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches", "rust", "crack", "scratch", "scale", "indentation", "blister" };
    cn_names_ = { "裂纹", "夹杂", "斑块", "麻点", "轧制氧化皮", "划痕", "锈蚀", "裂纹", "划痕", "氧化皮", "压痕", "气泡" };

    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
}

bool YoloDetector::init(const std::string& model_path, bool use_gpu) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // GPU Acceleration Config
        if (use_gpu) {
            try {
                // If CUDA is available, link CUDA provider
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "[YoloDetector] CUDA Execution Provider successfully linked." << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[YoloDetector] WARNING: CUDA EP not available. Falling back to CPU. Info: " << e.what() << std::endl;
            }
        }

        // Initialize session
#ifdef _WIN32
        std::wstring model_path_w(model_path.begin(), model_path.end());
        session_ = Ort::Session(env_, model_path_w.c_str(), session_options);
#else
        session_ = Ort::Session(env_, model_path.c_str(), session_options);
#endif

        // Get Input/Output Names
        Ort::AllocatorWithDefaultOptions allocator;
        
        Ort::AllocatedStringPtr input_name = session_.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr output_name = session_.GetOutputNameAllocated(0, allocator);
        
        // Cache name strings
        char* in_name = new char[strlen(input_name.get()) + 1];
        strcpy(in_name, input_name.get());
        input_names_.push_back(in_name);

        char* out_name = new char[strlen(output_name.get()) + 1];
        strcpy(out_name, output_name.get());
        output_names_.push_back(out_name);

        std::cout << "[YoloDetector] Successfully loaded ONNX model: " << model_path << std::endl;
        std::cout << "[YoloDetector] Input Name: " << input_names_[0] << " | Output Name: " << output_names_[0] << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[YoloDetector] CRITICAL: Failed to load model! " << e.what() << std::endl;
        return false;
    }
}

cv::Mat YoloDetector::preprocess(const cv::Mat& frame, std::vector<float>& input_tensor_values, float& scale_factor_x, float& scale_factor_y) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_width_, input_height_));
    
    scale_factor_x = static_cast<float>(frame.cols) / input_width_;
    scale_factor_y = static_cast<float>(frame.rows) / input_height_;

    cv::Mat float_image;
    resized.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    // HWC to CHW planar transposition
    input_tensor_values.resize(3 * input_width_ * input_height_);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_height_; ++h) {
            for (int w = 0; w < input_width_; ++w) {
                // YOLO typically expects RGB, we can convert here
                // OpenCV loads BGR by default, so we read c in reverse order (2 - c)
                input_tensor_values[c * input_width_ * input_height_ + h * input_width_ + w] = 
                    float_image.at<cv::Vec3f>(h, w)[2 - c];
            }
        }
    }

    return resized;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame, float conf_threshold, float iou_threshold) {
    std::vector<Detection> detections;
    if (!session_) {
        std::cerr << "[YoloDetector] ERROR: Session is null. Init model first." << std::endl;
        return detections;
    }

    std::vector<float> input_tensor_values;
    float scale_factor_x = 1.0f;
    float scale_factor_y = 1.0f;
    preprocess(frame, input_tensor_values, scale_factor_x, scale_factor_y);

    // Create Input Tensor
    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, 
        input_tensor_values.data(), 
        input_tensor_values.size(), 
        input_shape.data(), 
        input_shape.size()
    );

    // Running Inference
    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr}, 
        input_names_.data(), 
        &input_tensor, 
        1, 
        output_names_.data(), 
        1
    );

    // Get Output Tensor
    float* float_raw_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto output_shape = output_type_info.GetShape(); // Typically [1, columns, 8400]

    int rows = static_cast<int>(output_shape[1]); // e.g. 4 coords + number of classes
    int cols = static_cast<int>(output_shape[2]); // e.g. 8400 candidate anchors

    std::vector<float> output_tensor_values(float_raw_data, float_raw_data + (rows * cols));

    return postprocess(output_tensor_values, output_shape, scale_factor_x, scale_factor_y, conf_threshold, iou_threshold);
}

std::vector<Detection> YoloDetector::postprocess(const std::vector<float>& output_tensor_values, const std::vector<int64_t>& output_shape, 
                                                float scale_factor_x, float scale_factor_y, float conf_threshold, float iou_threshold) {
    std::vector<Detection> detections;
    
    int rows = static_cast<int>(output_shape[1]); // e.g. 4 coordinates + classes (e.g. 6 or 12)
    int cols = static_cast<int>(output_shape[2]); // e.g. 8400

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // Outer loop through 8400 candidates
    for (int i = 0; i < cols; ++i) {
        float max_score = 0.0f;
        int class_id = -1;

        // Start reading class scores from row index 4 onwards
        for (int j = 4; j < rows; ++j) {
            float score = output_tensor_values[j * cols + i];
            if (score > max_score) {
                max_score = score;
                class_id = j - 4;
            }
        }

        // Apply confidence threshold
        if (max_score >= conf_threshold) {
            float cx = output_tensor_values[0 * cols + i];
            float cy = output_tensor_values[1 * cols + i];
            float w = output_tensor_values[2 * cols + i];
            float h = output_tensor_values[3 * cols + i];

            // Reconstruct coordinates
            int left = static_cast<int>((cx - 0.5f * w) * scale_factor_x);
            int top = static_cast<int>((cy - 0.5f * h) * scale_factor_y);
            int width = static_cast<int>(w * scale_factor_x);
            int height = static_cast<int>(h * scale_factor_y);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(max_score);
            class_ids.push_back(class_id);
        }
    }

    // High performance NMS using OpenCV
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, indices);

    for (int idx : indices) {
        Detection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.class_id = class_ids[idx];
        
        if (det.class_id >= 0 && det.class_id < static_cast<int>(class_names_.size())) {
            det.class_name = class_names_[det.class_id];
            det.cn_name = cn_names_[det.class_id];
        } else {
            det.class_name = "unknown";
            det.cn_name = "未知";
        }
        detections.push_back(det);
    }

    return detections;
}
