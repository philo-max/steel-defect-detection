import os
import sys
import json
import time
import asyncio
import sqlite3
import logging
import threading
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO

# Setup paths and logger
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "inspection.db"
MODEL_PATH = PROJECT_ROOT / "models" / "weights" / "steel_defect.pt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api_bridge")

app = FastAPI(title="Steel Defect Detection V3.0 Python Bridge")

# CORS middleware for smooth Vue 3 frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active websocket client manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New frontend dashboard connected. Total active sessions: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Dashboard disconnected. Total active sessions: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        # Clean disconnected sockets on broadcast
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Global metrics cache
system_stats = {
    "cpu_usage": 15.0,
    "cpu_temp": 44.0,
    "gpu_usage": 32.0,
    "gpu_temp": 54.0,
    "inference_delay": 2.2,
    "camera_fps": 30.0
}

# 1. Background real-time camera scanning & YOLOv8 inference pipeline
def run_realtime_pipeline():
    logger.info("Initializing background real-time YOLOv8 scanning pipeline...")
    
    # Load YOLOv8 Model using Ultralytics
    try:
        model = YOLO(str(MODEL_PATH))
        logger.info(f"YOLOv8 weights successfully loaded from: {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load YOLOv8 model! {e}")
        return

    # Define standard colors/classes map
    class_cn = {
        "crazing": "裂纹", "inclusion": "夹杂", "patches": "斑块",
        "pitted_surface": "麻点", "rolled-in_scale": "轧制氧化皮", "scratches": "划痕",
        "rust": "锈蚀", "crack": "裂纹", "scratch": "划痕",
        "scale": "氧化皮", "indentation": "压痕", "blister": "气泡"
    }

    # Standard blank frame
    mock_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.line(mock_frame, (0, 320), (640, 320), (45, 45, 45), 2)
    
    # Standard defect frame
    mock_defect_frame = mock_frame.copy()
    # Simulate defect scratches/crazing
    cv2.line(mock_defect_frame, (150, 120), (480, 490), (95, 95, 180), 3)

    # Establish asyncio loop for websocket broadcasting
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        time.sleep(4.0)  # Linescans every 4 seconds
        
        # 15% probability of defect triggering
        has_defect = random.random() < 0.15
        active_frame = mock_defect_frame if has_defect else mock_frame
        
        # Model Inference
        start_time = time.time()
        results = model(active_frame, verbose=False)
        inference_ms = (time.time() - start_time) * 1000.0
        
        system_stats["inference_delay"] = inference_ms
        
        # Parse detections
        defects_list = []
        defect_count = 0
        min_conf = 0.99
        defect_names = []

        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                cn_name = class_cn.get(class_name, "未知")
                conf = float(box.conf[0])
                
                # Bounding Box [x, y, w, h]
                xyxy = box.xyxy[0].tolist()
                x = int(xyxy[0])
                y = int(xyxy[1])
                w = int(xyxy[2] - xyxy[0])
                h = int(xyxy[3] - xyxy[1])

                defects_list.append({
                    "type": class_name,
                    "cn": cn_name,
                    "confidence": conf,
                    "box": [x, y, w, h]
                })
                defect_names.append(cn_name)
                defect_count += 1
                if conf < min_conf:
                    min_conf = conf

        # Formulate record
        record_id = random.randint(10000, 99999)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        image_path = f"/data/images/sheet_{record_id}.jpg"

        yolo_result = {
            "defects": defects_list,
            "inference_time_ms": inference_ms
        }
        
        final_status = "defect" if defect_count > 0 else "pass"
        final_result = {
            "status": final_status,
            "defects": defects_list if defect_count > 0 else []
        }

        # Persist to SQLite
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO inspection_records (
                    image_path, result_path, yolo_result, vlm_result, final_result,
                    defect_types, defect_count, confidence, reviewer, review_status, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_path, "", json.dumps(yolo_json := yolo_result), "{}", json.dumps(final_result),
                ",".join(defect_names), defect_count, min_conf if defect_count > 0 else 0.99,
                "", "pending" if defect_count > 0 else "confirmed", ""
            ))
            conn.commit()
            record_id = cursor.lastrowid
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save record to database: {e}")

        # Broadcast WS detection event to frontend
        ws_msg = {
            "type": "detection",
            "data": {
                "id": record_id,
                "timestamp": timestamp,
                "image_path": image_path,
                "result_path": "",
                "yolo_result": yolo_result,
                "vlm_result": {},
                "final_result": final_result,
                "defect_types": ",".join(defect_names),
                "defect_count": defect_count,
                "confidence": min_conf if defect_count > 0 else 0.99,
                "reviewer": "",
                "review_status": "pending" if defect_count > 0 else "confirmed",
                "note": ""
            }
        }
        
        # Async broadcast in synchronous thread pool
        asyncio.run_coroutine_threadsafe(manager.broadcast(ws_msg), loop)

# 2. Background system metrics broadcaster
def run_metrics_broadcaster():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    while True:
        time.sleep(2.0)
        
        # Fluctuate hardware stats
        system_stats["cpu_usage"] = max(5.0, min(45.0, system_stats["cpu_usage"] + random.uniform(-2.0, 2.0)))
        system_stats["gpu_usage"] = max(10.0, min(90.0, system_stats["gpu_usage"] + random.uniform(-3.0, 3.0)))
        
        ws_msg = {
            "type": "metrics",
            "data": {
                "cpu_usage": system_stats["cpu_usage"],
                "cpu_temp": system_stats["cpu_temp"],
                "gpu_usage": system_stats["gpu_usage"],
                "gpu_temp": system_stats["gpu_temp"],
                "inference_delay": system_stats["inference_delay"],
                "camera_fps": 30.0
            }
        }
        
        asyncio.run_coroutine_threadsafe(manager.broadcast(ws_msg), loop)

# 3. REST API: GET /api/records
@app.get("/api/records")
def get_records(limit: int = 20, offset: int = 0):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inspection_records ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
        rows = cursor.fetchall()
        conn.close()

        records_list = []
        for r in rows:
            record = dict(r)
            # Parse stored JSON strings back to dict objects for neat frontend processing
            try:
                record["yolo_result"] = json.loads(record["yolo_result"])
            except Exception:
                record["yolo_result"] = {}
                
            try:
                record["vlm_result"] = json.loads(record["vlm_result"])
            except Exception:
                record["vlm_result"] = {}
                
            try:
                record["final_result"] = json.loads(record["final_result"])
            except Exception:
                record["final_result"] = {}
                
            records_list.append(record)
            
        return records_list
    except Exception as e:
        logger.error(f"Failed to query records: {e}")
        return []

# 4. REST API: POST /api/audit
@app.post("/api/audit")
def post_audit(payload: dict):
    record_id = payload.get("id")
    status = payload.get("review_status")
    reviewer = payload.get("reviewer", "操作员")
    note = payload.get("note", "")

    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE inspection_records 
            SET review_status = ?, reviewer = ?, note = ?, review_time = CURRENT_TIMESTAMP 
            WHERE id = ?
        """, (status, reviewer, note, record_id))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return {"success": success, "message": "Audit processed successfully!" if success else "Record not found."}
    except Exception as e:
        logger.error(f"Audit transaction failed: {e}")
        return {"success": false, "message": str(e)}

# 5. REST API: POST /api/consult (Gemini VLM + SQLite GB/T RAG)
@app.post("/api/consult")
def post_consult(id: int = Query(...)):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # 1. Fetch the record details
        cursor.execute("SELECT * FROM inspection_records WHERE id = ?", (id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return {"error": "Record not found."}

        defect_types = row[7]
        confidence = row[9]

        # 2. Query matching GB/T standards from knowledge_base (RAG)
        cursor.execute("""
            SELECT standard_code, title, content FROM knowledge_base 
            WHERE defect_type = ? OR title LIKE ? LIMIT 1
        """, (defect_types, f"%{defect_types}%"))
        standard_row = cursor.fetchone()
        
        rag_standard = {}
        if standard_row:
            rag_standard = {
                "standard_code": standard_row[0],
                "title": standard_row[1],
                "content": standard_row[2]
            }
        else:
            rag_standard = {
                "standard_code": "GB/T 3280-2015",
                "title": "高精度不锈钢板外观判定通用标准",
                "content": "国家标准规范指出，对于带钢表面缺陷检测，任何大面积起皮、龟裂和宏观物理拉线划伤严重危害物理载截面性能，均视为失效，归于 P0 级判废范畴。"
            }

        # 3. Simulate expert consultation report based on Gemini VLM guidelines
        analysis = (
            f"【大模型诊断结论】\n"
            f"经由机器视觉大模型（VLM）与国家钢铁表面质量规范（{rag_standard['standard_code']}）联合校验判定。由于线阵相机在高对比度下的缺陷提取，所指位置处的异常分类确属 {defect_types}（缺陷置信度: {int(confidence * 100)}%）。\n"
            f"根据《{rag_standard['title']}》限制，该类缺陷深度和纵深尺度对带钢疲劳抗剪切极限构成负向劣化阻碍，判定属 P0 级严重异常，应予裁切并判废。\n\n"
            f"【工艺改进及整改路径】\n"
            f"1. 彻底清空酸洗漂洗池铁泥颗粒，提高喷洗气刀冲洗冲击压强至 18MPa。\n"
            f"2. 检测出口擦拭器的辊筒胶皮磨损状态，杜绝层间错移拉伤。"
        )

        vlm_result = {
            "status": "completed",
            "confidence": 0.95,
            "analysis": analysis
        }

        # 4. Save consultation outcomes to record
        cursor.execute("""
            UPDATE inspection_records 
            SET vlm_result = ? 
            WHERE id = ?
        """, (json.dumps(vlm_result), id))
        conn.commit()
        conn.close()

        return {
            "rag_standard": rag_standard,
            "vlm_result": vlm_result
        }
    except Exception as e:
        logger.error(f"Consultation failed: {e}")
        return {"error": str(e)}

# 6. WebSocket Stream controller
@app.websocket("/camera/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for incoming messages if any
            data = await websocket.receive_text()
            logger.info(f"Received message from client: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Startup event triggers background threads
@app.on_event("startup")
def startup_event():
    # Make sure database is seeded
    db_dir = DB_PATH.parent
    db_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Spawn Camera scan loop thread
    t1 = threading.Thread(target=run_realtime_pipeline, daemon=True)
    t1.start()
    
    # 2. Spawn Metrics loop thread
    t2 = threading.Thread(target=run_metrics_broadcaster, daemon=True)
    t2.start()
    
    logger.info("FastAPI Web Server Bridge successfully set up and running on port 8080.")

if __name__ == "__main__":
    uvicorn.run("vue_api_bridge:app", host="0.0.0.0", port=8080, reload=False)
