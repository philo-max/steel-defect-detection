#include "DbManager.hpp"
#include <iostream>
#include <sstream>
#include <chrono>

DbManager::~DbManager() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool DbManager::init(const std::string& db_path) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    db_path_ = db_path;

    int rc = sqlite3_open(db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::cerr << "[DbManager] CRITICAL: Can't open database: " << sqlite3_errmsg(db_) << std::endl;
        return false;
    }

    // Enable high performance WAL mode
    char* zErrMsg = nullptr;
    sqlite3_exec(db_, "PRAGMA journal_mode=WAL;", nullptr, nullptr, &zErrMsg);
    sqlite3_exec(db_, "PRAGMA synchronous=NORMAL;", nullptr, nullptr, &zErrMsg);
    
    // Set 5 seconds busy timeout
    sqlite3_busy_timeout(db_, 5000);

    std::cout << "[DbManager] Database opened successfully at: " << db_path_ << std::endl;
    return createTables();
}

bool DbManager::createTables() {
    char* errMsg = nullptr;
    
    const char* sql_records = 
        "CREATE TABLE IF NOT EXISTS inspection_records ("
        "  id              INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  timestamp       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  image_path      TEXT NOT NULL,"
        "  result_path     TEXT DEFAULT '',"
        "  yolo_result     TEXT DEFAULT '{}',"
        "  vlm_result      TEXT DEFAULT '{}',"
        "  final_result    TEXT DEFAULT '{}',"
        "  defect_types    TEXT DEFAULT '',"
        "  defect_count    INTEGER DEFAULT 0,"
        "  confidence      REAL DEFAULT 0.0,"
        "  reviewer        TEXT DEFAULT '',"
        "  review_status   TEXT DEFAULT 'pending',"
        "  review_time     DATETIME,"
        "  note            TEXT DEFAULT ''"
        ");";

    int rc = sqlite3_exec(db_, sql_records, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "[DbManager] SQL Error in records creation: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }

    // Create Indexes for high performance queries
    sqlite3_exec(db_, "CREATE INDEX IF NOT EXISTS idx_timestamp ON inspection_records(timestamp);", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "CREATE INDEX IF NOT EXISTS idx_defect_types ON inspection_records(defect_types);", nullptr, nullptr, nullptr);
    sqlite3_exec(db_, "CREATE INDEX IF NOT EXISTS idx_review_status ON inspection_records(review_status);", nullptr, nullptr, nullptr);

    return true;
}

int DbManager::saveRecord(const InspectionRecordCpp& record) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = 
        "INSERT INTO inspection_records ("
        "  image_path, result_path, yolo_result, vlm_result, final_result,"
        "  defect_types, defect_count, confidence, reviewer, review_status, note"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "[DbManager] Prepare statement failed: " << sqlite3_errmsg(db_) << std::endl;
        return -1;
    }

    // Bind parameters
    sqlite3_bind_text(stmt, 1, record.image_path.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, record.result_path.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, record.yolo_result.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, record.vlm_result.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, record.final_result.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 6, record.defect_types.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 7, record.defect_count);
    sqlite3_bind_double(stmt, 8, record.confidence);
    sqlite3_bind_text(stmt, 9, record.reviewer.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 10, record.review_status.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 11, record.note.c_str(), -1, SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    int last_id = -1;
    if (rc == SQLITE_DONE) {
        last_id = static_cast<int>(sqlite3_last_insert_rowid(db_));
    } else {
        std::cerr << "[DbManager] Insert step failed: " << sqlite3_errmsg(db_) << std::endl;
    }

    sqlite3_finalize(stmt);
    return last_id;
}

bool DbManager::getRecord(int record_id, InspectionRecordCpp& record) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = "SELECT * FROM inspection_records WHERE id = ?;";
    sqlite3_stmt* stmt = nullptr;
    
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return false;

    sqlite3_bind_int(stmt, 1, record_id);
    
    bool found = false;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        found = true;
        record.id = sqlite3_column_int(stmt, 0);
        record.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        record.image_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        record.result_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        record.yolo_result = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        record.vlm_result = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        record.final_result = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        record.defect_types = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        record.defect_count = sqlite3_column_int(stmt, 8);
        record.confidence = sqlite3_column_double(stmt, 9);
        record.reviewer = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 10));
        record.review_status = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 11));
        
        const unsigned char* rev_time = sqlite3_column_text(stmt, 12);
        record.review_time = rev_time ? reinterpret_cast<const char*>(rev_time) : "";
        record.note = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 13));
    }

    sqlite3_finalize(stmt);
    return found;
}

std::vector<InspectionRecordCpp> DbManager::getRecords(int limit, int offset) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    std::vector<InspectionRecordCpp> results;
    
    const char* sql = "SELECT * FROM inspection_records ORDER BY id DESC LIMIT ? OFFSET ?;";
    sqlite3_stmt* stmt = nullptr;
    
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return results;

    sqlite3_bind_int(stmt, 1, limit);
    sqlite3_bind_int(stmt, 2, offset);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        InspectionRecordCpp record;
        record.id = sqlite3_column_int(stmt, 0);
        record.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        record.image_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        record.result_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        record.yolo_result = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        record.vlm_result = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        record.final_result = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        record.defect_types = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        record.defect_count = sqlite3_column_int(stmt, 8);
        record.confidence = sqlite3_column_double(stmt, 9);
        record.reviewer = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 10));
        record.review_status = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 11));
        
        const unsigned char* rev_time = sqlite3_column_text(stmt, 12);
        record.review_time = rev_time ? reinterpret_cast<const char*>(rev_time) : "";
        record.note = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 13));
        
        results.push_back(record);
    }

    sqlite3_finalize(stmt);
    return results;
}

bool DbManager::updateAudit(int record_id, const std::string& status, const std::string& reviewer, const std::string& note) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    const char* sql = 
        "UPDATE inspection_records "
        "SET review_status = ?, reviewer = ?, note = ?, review_time = CURRENT_TIMESTAMP "
        "WHERE id = ?;";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return false;

    sqlite3_bind_text(stmt, 1, status.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, reviewer.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, note.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 4, record_id);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

bool DbManager::queryRagStandard(const std::string& defect_type, RagStandard& standard) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    // Fuzzy matching by defect type, map standard classes exactly
    const char* sql = "SELECT standard_code, title, content FROM knowledge_base WHERE defect_type = ? OR title LIKE ? LIMIT 1;";
    sqlite3_stmt* stmt = nullptr;
    
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) return false;

    std::string like_query = "%" + defect_type + "%";
    sqlite3_bind_text(stmt, 1, defect_type.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, like_query.c_str(), -1, SQLITE_TRANSIENT);

    bool found = false;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        found = true;
        standard.standard_code = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        standard.title = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        standard.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
    }

    sqlite3_finalize(stmt);
    return found;
}
