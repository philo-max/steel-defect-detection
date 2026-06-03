#include <drogon/drogon.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include "YoloDetector.hpp"
#include "DbManager.hpp"

using namespace drogon;

// Helper to escape strings for secure JSON construction
std::string escapeJsonString(const std::string& input) {
    std::ostringstream ss;
    for (char c : input) {
        switch (c) {
            case '\\': ss << "\\\\"; break;
            case '"': ss << "\\\""; break;
            case '/': ss << "\\/"; break;
            case '\b': ss << "\\b"; break;
            case '\f': ss << "\\f"; break;
            case '\n': ss << "\\n"; break;
            case '\r': ss << "\\r"; break;
            case '\t': ss << "\\t"; break;
            default:
                if (c >= 0 && c < 32) {
                    ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    ss << c;
                }
                break;
        }
    }
    return ss.str();
}

// Global state and thread safety
std::mutex g_ws_mutex;
std::vector<WebSocketConnectionPtr> g_ws_connections;
DbManager g_db;
YoloDetector g_detector;

// System Metrics struct
struct SystemStats {
    double cpu_usage = 10.0;
    double cpu_temp = 42.0;
    double gpu_usage = 25.0;
    double gpu_temp = 50.0;
    double inference_delay = 1.8;
} g_stats;

// Broadcast helper
void broadcastMessage(const std::string& message) {
    std::lock_guard<std::mutex> lock(g_ws_mutex);
    for (auto it = g_ws_connections.begin(); it != g_ws_connections.end();) {
        if ((*it)->connected()) {
            (*it)->send(message);
            ++it;
        } else {
            it = g_ws_connections.erase(it);
        }
    }
}

// Background Simulated Line-Scan Camera & Inference Pipeline
void cameraPipelineLoop() {
    std::cout << "[Pipeline] Camera scanning thread initialized." << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> defect_roll(0.0, 1.0);
    std::uniform_int_distribution<> record_id_gen(10000, 99999);
    
    // Set up dummy frames for YOLO processing
    cv::Mat mock_ok_frame = cv::Mat::zeros(640, 640, CV_8UC3); // Blank OK steel frame
    
    // Draw some typical industrial line scanning noise
    cv::line(mock_ok_frame, cv::Point(0, 320), cv::Point(640, 320), cv::Scalar(40, 40, 40), 2);
    
    cv::Mat mock_defect_frame = mock_ok_frame.clone();
    // Simulate a deep line scratch defect
    cv::line(mock_defect_frame, cv::Point(200, 150), cv::Point(450, 450), cv::Scalar(100, 100, 200), 3);

    while (true) {
        // Line scans every 4 seconds
        std::this_thread::sleep_for(std::chrono::milliseconds(4000));
        
        bool has_defect = defect_roll(gen) < 0.15; // 15% defect rate
        cv::Mat active_frame = has_defect ? mock_defect_frame : mock_ok_frame;
        
        // Execute ONNX C++ Inference
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> results = g_detector.detect(active_frame, 0.25f, 0.45f);
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        g_stats.inference_delay = duration;

        // Populate database object
        InspectionRecordCpp record;
        record.id = record_id_gen(gen);
        
        // Formulate ISO Timestamp
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        char time_str[64];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now_time));
        record.timestamp = time_str;
        
        record.image_path = "/data/images/sheet_" + std::to_string(record.id) + ".jpg";
        record.result_path = "";
        
        // YOLO output serialization
        std::stringstream yolo_json;
        yolo_json << "{\"defects\":[";
        
        std::stringstream defect_names;
        record.defect_count = 0;
        record.confidence = 0.99;

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& d = results[i];
            yolo_json << (i > 0 ? "," : "")
                      << "{\"type\":\"" << escapeJsonString(d.class_name) << "\",\"cn\":\"" << escapeJsonString(d.cn_name) 
                      << "\",\"confidence\":" << d.confidence 
                      << ",\"box\":[" << d.box.x << "," << d.box.y << "," << d.box.width << "," << d.box.height << "]}";
            
            if (i > 0) defect_names << ",";
            defect_names << d.cn_name;
            
            record.defect_count++;
            if (d.confidence < record.confidence) {
                record.confidence = d.confidence; // Track minimum confidence
            }
        }
        yolo_json << "],\"inference_time_ms\":" << duration << "}";
        
        record.yolo_result = yolo_json.str();
        record.vlm_result = "{}";
        record.defect_types = defect_names.str();
        
        // Final outcome mapping
        if (record.defect_count > 0) {
            record.final_result = "{\"status\":\"defect\",\"defects\":" + yolo_json.str() + "}";
            record.review_status = "pending";
        } else {
            record.final_result = "{\"status\":\"pass\"}";
            record.review_status = "confirmed";
        }
        
        record.reviewer = "";
        record.note = "";

        // Persist to database
        int db_id = g_db.saveRecord(record);
        if (db_id != -1) {
            record.id = db_id;
        }

        // Broadcast to WebSocket subscribers in JSON format
        std::stringstream ws_msg;
        ws_msg << "{\"type\":\"detection\",\"data\":{"
               << "\"id\":" << record.id << ","
               << "\"timestamp\":\"" << escapeJsonString(record.timestamp) << "\","
               << "\"image_path\":\"" << escapeJsonString(record.image_path) << "\","
               << "\"result_path\":\"" << escapeJsonString(record.result_path) << "\","
               << "\"yolo_result\":" << record.yolo_result << ","
               << "\"vlm_result\":" << record.vlm_result << ","
               << "\"final_result\":" << record.final_result << ","
               << "\"defect_types\":\"" << escapeJsonString(record.defect_types) << "\","
               << "\"defect_count\":" << record.defect_count << ","
               << "\"confidence\":" << record.confidence << ","
               << "\"reviewer\":\"" << escapeJsonString(record.reviewer) << "\","
               << "\"review_status\":\"" << escapeJsonString(record.review_status) << "\","
               << "\"note\":\"" << escapeJsonString(record.note) << "\""
               << "}}";

        broadcastMessage(ws_msg.str());
    }
}

// Background metrics broadcast thread
void metricsBroadcastLoop() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> offset_gen(-2.0, 2.0);

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        
        // Fluctuating hardware indicators
        g_stats.cpu_usage = std::max(5.0, std::min(45.0, g_stats.cpu_usage + offset_gen(gen)));
        g_stats.gpu_usage = std::max(10.0, std::min(90.0, g_stats.gpu_usage + offset_gen(gen) * 1.5));
        
        std::stringstream metrics_json;
        metrics_json << "{\"type\":\"metrics\",\"data\":{"
                     << "\"cpu_usage\":" << g_stats.cpu_usage << ","
                     << "\"cpu_temp\":" << g_stats.cpu_temp << ","
                     << "\"gpu_usage\":" << g_stats.gpu_usage << ","
                     << "\"gpu_temp\":" << g_stats.gpu_temp << ","
                     << "\"inference_delay\":" << g_stats.inference_delay << ","
                     << "\"camera_fps\":" << 30.0
                     << "}}";
                     
        broadcastMessage(metrics_json.str());
    }
}

int main() {
    std::cout << "[Server] Initializing V3.0 C++ industrial backend..." << std::endl;
    
    // Initialize Database
    if (!g_db.init("data/inspection.db")) {
        std::cerr << "[Server] Failed to initialize SQLite database! Exiting." << std::endl;
        return 1;
    }

    // Initialize YOLO Detector (loads the pre-exported ONNX model)
    if (!g_detector.init("models/weights/steel_defect.onnx", true)) {
        std::cerr << "[Server] Failed to initialize ONNX Runtime. Continuing in fallback mode." << std::endl;
    }

    // Launch Background Thread Pipelines
    std::thread pipeline_thread(cameraPipelineLoop);
    pipeline_thread.detach();

    std::thread metrics_thread(metricsBroadcastLoop);
    metrics_thread.detach();

    // Drogon Routes definitions
    
    // 1. GET /api/records -> Retrieve recent records
    app().registerHandler("/api/records", [](const HttpRequestPtr& req, std::function<void (const HttpResponsePtr&)>&& callback) {
        int limit = 20;
        int offset = 0;
        
        auto req_limit = req->getParameter("limit");
        auto req_offset = req->getParameter("offset");
        if (!req_limit.empty()) limit = std::stoi(req_limit);
        if (!req_offset.empty()) offset = std::stoi(req_offset);

        auto rows = g_db.getRecords(limit, offset);
        
        Json::Value json_arr(Json::arrayValue);
        for (const auto& r : rows) {
            Json::Value json_rec;
            json_rec["id"] = r.id;
            json_rec["timestamp"] = r.timestamp;
            json_rec["image_path"] = r.image_path;
            json_rec["result_path"] = r.result_path;
            
            // Parse JSON strings back to JSON values for neat responses
            Json::Reader reader;
            
            Json::Value yolo_val;
            reader.parse(r.yolo_result, yolo_val);
            json_rec["yolo_result"] = yolo_val;

            Json::Value vlm_val;
            reader.parse(r.vlm_result, vlm_val);
            json_rec["vlm_result"] = vlm_val;

            Json::Value final_val;
            reader.parse(r.final_result, final_val);
            json_rec["final_result"] = final_val;

            json_rec["defect_types"] = r.defect_types;
            json_rec["defect_count"] = r.defect_count;
            json_rec["confidence"] = r.confidence;
            json_rec["reviewer"] = r.reviewer;
            json_rec["review_status"] = r.review_status;
            json_rec["review_time"] = r.review_time;
            json_rec["note"] = r.note;
            
            json_arr.append(json_rec);
        }

        auto resp = HttpResponse::newHttpJsonResponse(json_arr);
        resp->addHeader("Access-Control-Allow-Origin", "*");
        callback(resp);
    });

    // 2. POST /api/audit -> Submit operator review judgment
    app().registerHandler("/api/audit", [](const HttpRequestPtr& req, std::function<void (const HttpResponsePtr&)>&& callback) {
        auto json_body = req->getJsonObject();
        Json::Value resp_json;
        
        if (json_body) {
            int id = (*json_body)["id"].asInt();
            std::string status = (*json_body)["review_status"].asString();
            std::string reviewer = (*json_body)["reviewer"].asString();
            std::string note = (*json_body)["note"].asString();

            bool success = g_db.updateAudit(id, status, reviewer, note);
            resp_json["success"] = success;
            resp_json["message"] = success ? "Audit saved successfully!" : "Failed to save audit.";
        } else {
            resp_json["success"] = false;
            resp_json["message"] = "Invalid request payload.";
        }

        auto resp = HttpResponse::newHttpJsonResponse(resp_json);
        resp->addHeader("Access-Control-Allow-Origin", "*");
        resp->addHeader("Access-Control-Allow-Headers", "Content-Type");
        callback(resp);
    }, {Options, Post});

    // 3. POST /api/consult -> Gemini VLM + SQLite GB/T RAG expert lookup
    app().registerHandler("/api/consult", [](const HttpRequestPtr& req, std::function<void (const HttpResponsePtr&)>&& callback) {
        auto req_id = req->getParameter("id");
        Json::Value resp_json;

        if (!req_id.empty()) {
            int record_id = std::stoi(req_id);
            InspectionRecordCpp record;
            
            if (g_db.getRecord(record_id, record)) {
                RagStandard standard;
                // Query standard via defect type
                bool has_standard = g_db.queryRagStandard(record.defect_types, standard);

                Json::Value rag_val;
                if (has_standard) {
                    rag_val["standard_code"] = standard.standard_code;
                    rag_val["title"] = standard.title;
                    rag_val["content"] = standard.content;
                } else {
                    rag_val["standard_code"] = "GB/T 3280-2015";
                    rag_val["title"] = "高精度不锈钢板外观合格判定通用规范";
                    rag_val["content"] = "成品钢板在交货状态下，表面应平整光洁。轻微缺陷允许在板公差厚度负偏差之半内存在。严重龟裂、断裂及氧化铁皮硬质压入为不合格品。";
                }
                resp_json["rag_standard"] = rag_val;

                // Simulated expert report leveraging prompt engineering
                std::stringstream diagnosis;
                diagnosis << "【大模型诊断结论】\n"
                          << "经由机器视觉大模型（VLM）与国家钢铁表面质量规范（" 
                          << (has_standard ? standard.standard_code : "GB/T 3280-2015") 
                          << "）联合校验判定。所指位置处的异常分类确属 " << record.defect_types 
                          << "（缺陷置信度: " << (record.confidence * 100) << "%）。\n"
                          << "根据 " << (has_standard ? standard.title : "国家通用标准") << " 限制，该类缺陷深度和纵深尺度对带钢疲劳抗剪切极限构成负向劣化阻碍，判定属 P0 级严重异常，应予裁切并判废。\n\n"
                          << "【工艺改进及整改路径】\n"
                          << "1. 彻底清空酸洗漂洗池铁泥颗粒，提高喷洗气刀冲洗冲击压强至 18MPa。\n"
                          << "2. 检测出口擦拭器的辊筒胶皮磨损状态，杜绝层间错移拉伤。";

                Json::Value vlm_val;
                vlm_val["status"] = "completed";
                vlm_val["confidence"] = 0.95;
                vlm_val["analysis"] = diagnosis.str();
                resp_json["vlm_result"] = vlm_val;

                // Sync update record in DB
                record.vlm_result = "{\"status\":\"completed\",\"confidence\":0.95,\"analysis\":\"" + diagnosis.str() + "\"}";
                g_db.saveRecord(record); // Overwrite record with expert consult outcomes
                
            } else {
                resp_json["error"] = "Record not found.";
            }
        } else {
            resp_json["error"] = "Missing parameter id.";
        }

        auto resp = HttpResponse::newHttpJsonResponse(resp_json);
        resp->addHeader("Access-Control-Allow-Origin", "*");
        callback(resp);
    }, {Options, Post});

    // 4. WebSocket stream handler
    app().registerWebSocketController("/camera/stream", [](const WebSocketConnectionPtr& conn, const std::string& message) {
        // Do nothing on message
        (void)message;
        (void)conn;
    }, [](const WebSocketConnectionPtr& conn) {
        // On connection open
        std::lock_guard<std::mutex> lock(g_ws_mutex);
        g_ws_connections.push_back(conn);
        std::cout << "[WebSocket] New operator terminal connected. Active sessions: " << g_ws_connections.size() << std::endl;
    }, [](const WebSocketConnectionPtr& conn) {
        // On connection close
        std::lock_guard<std::mutex> lock(g_ws_mutex);
        auto it = std::find(g_ws_connections.begin(), g_ws_connections.end(), conn);
        if (it != g_ws_connections.end()) {
            g_ws_connections.erase(it);
        }
        std::cout << "[WebSocket] Operator connection closed. Active sessions: " << g_ws_connections.size() << std::endl;
    });

    // Run Drogon Server on Port 8080
    std::cout << "[Server] Server successfully configured. Listening on http://0.0.0.0:8080" << std::endl;
    app().addListener("0.0.0.0", 8080);
    app().run();

    return 0;
}
