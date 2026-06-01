#pragma once

#include <string>
#include <vector>
#include <sqlite3.h>
#include <mutex>

struct RagStandard {
    std::string standard_code;
    std::string title;
    std::string content;
};

struct InspectionRecordCpp {
    int id;
    std::string timestamp;
    std::string image_path;
    std::string result_path;
    std::string yolo_result;
    std::string vlm_result;
    std::string final_result;
    std::string defect_types;
    int defect_count;
    double confidence;
    std::string reviewer;
    std::string review_status;
    std::string review_time;
    std::string note;
};

class DbManager {
public:
    DbManager() = default;
    ~DbManager();

    bool init(const std::string& db_path);
    
    // Save record to DB
    int saveRecord(const InspectionRecordCpp& record);
    
    // Retrieve record
    bool getRecord(int record_id, InspectionRecordCpp& record);
    
    // Query records lists
    std::vector<InspectionRecordCpp> getRecords(int limit = 20, int offset = 0);
    
    // Audit human control
    bool updateAudit(int record_id, const std::string& status, const std::string& reviewer, const std::string& note);
    
    // Retrieve matching RAG GB/T standards from knowledge_base
    bool queryRagStandard(const std::string& defect_type, RagStandard& standard);

private:
    sqlite3* db_ = nullptr;
    std::mutex db_mutex_;
    std::string db_path_;

    bool createTables();
};
