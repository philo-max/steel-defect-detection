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
import yaml
from src.detection_engine import YOLODetector
from dotenv import load_dotenv

# Setup paths and logger
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "inspection.db"
MODEL_PATH = PROJECT_ROOT / "models" / "weights" / "steel_defect.pt"

# Load local .env configuration (keys, endpoint models)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Real VLM & RAG global state
vlm_detector = None
vlm_detector_lock = threading.Lock()

def get_vlm_detector():
    global vlm_detector
    with vlm_detector_lock:
        if vlm_detector is None:
            try:
                from src.vlm_engine import VLMDetector
                # We initialize VLMDetector. By default it picks up the correct keys and mimo-v2.5 model
                vlm_detector = VLMDetector()
                vlm_detector.load_model()
                logger.info("Successfully loaded real VLMDetector in API Bridge.")
            except Exception as e:
                logger.error(f"Failed to load VLMDetector: {e}")
        return vlm_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api_bridge")

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Steel Defect Detection V3.0 Python Bridge")

# Mount static folder so frontend can read the raw inspection images directly
app.mount("/data", StaticFiles(directory=str(PROJECT_ROOT / "data")), name="data")

# CORS middleware for smooth Vue 3 frontend connection
cors_origins = [
    "http://localhost:5173",
    "http://localhost:7860",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:7860",
]
env_origins = os.getenv("CORS_ORIGINS")
if env_origins:
    cors_origins.extend([o.strip() for o in env_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
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
    
    # Load YOLOv8 Model using YOLODetector to support CLAHE
    try:
        config_path = PROJECT_ROOT / "config.yaml"
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        yolo_cfg = config.get("yolo", {})
        clahe_cfg = yolo_cfg.get("clahe", {})
        
        model = YOLODetector(
            model_path=str(MODEL_PATH),
            conf_threshold=yolo_cfg.get("conf_threshold", 0.05),
            iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
            img_size=yolo_cfg.get("img_size", 640),
            device=yolo_cfg.get("device", "auto"),
            half=yolo_cfg.get("half", True),
            clahe_enabled=clahe_cfg.get("enabled", True),
            clahe_clip_limit=clahe_cfg.get("clip_limit", 2.0),
            clahe_tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
        )
        model.load_model()
        logger.info(f"YOLODetector successfully loaded from: {MODEL_PATH} (CLAHE enabled: {model.clahe_enabled})")
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

    # Load actual steel plate texture background
    sample_path = PROJECT_ROOT / "data" / "images" / "sample_steel.jpg"
    if sample_path.exists():
        mock_frame = cv2.imread(str(sample_path))
        mock_frame = cv2.resize(mock_frame, (640, 640))
        logger.info(f"Successfully loaded real steel plate texture for real-time background scanning: {sample_path}")
    else:
        mock_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.line(mock_frame, (0, 320), (640, 320), (45, 45, 45), 2)
    
    # Standard defect frame (with real mechanical scratch drawn over the steel background)
    mock_defect_frame = mock_frame.copy()
    # Draw dark linear scratch defects
    cv2.line(mock_defect_frame, (150, 120), (480, 490), (50, 50, 50), 4)
    cv2.line(mock_defect_frame, (120, 140), (130, 280), (35, 35, 35), 3)

    # Establish asyncio loop for websocket broadcasting
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        time.sleep(4.0)  # Linescans every 4 seconds
        
        # 15% probability of defect triggering
        has_defect = random.random() < 0.15
        active_frame = mock_defect_frame if has_defect else mock_frame
        
        # Model Inference
        inference_res = model.detect(active_frame)
        if inference_res.error:
            logger.error(f"YOLODetector inference error: {inference_res.error}")
            continue
        inference_ms = inference_res.inference_time_ms
        
        system_stats["inference_delay"] = inference_ms
        
        # Parse detections
        defects_list = []
        defect_count = 0
        min_conf = 0.99
        defect_names = []

        img_h, img_w = active_frame.shape[:2]
        for det in inference_res.detections:
            x1 = int(det.bbox[0] * img_w)
            y1 = int(det.bbox[1] * img_h)
            x2 = int(det.bbox[2] * img_w)
            y2 = int(det.bbox[3] * img_h)
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1

            class_name = det.class_name
            cn_name = class_cn.get(class_name, "未知")
            conf = det.confidence

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

        # Write image to disk so the Vue frontend static server can serve it without 404
        try:
            full_img_path = PROJECT_ROOT / "data" / "images" / f"sheet_{record_id}.jpg"
            full_img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(full_img_path), active_frame)
        except Exception as img_err:
            logger.error(f"Failed to write image file: {img_err}")

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
        return {"success": False, "message": str(e)}

# 4.5 REST API: POST /api/sync_record (triggered by Gradio app on updates/insertions)
@app.post("/api/sync_record")
async def post_sync_record(id: int = Query(...)):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inspection_records WHERE id = ?", (id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            record = dict(row)
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
                
            # Broadcast WS detection event to frontend
            ws_msg = {
                "type": "detection",
                "data": record
            }
            await manager.broadcast(ws_msg)
            logger.info(f"Successfully broadcasted real-time sync event for record #{id}")
            return {"success": True, "message": f"Broadcasted record #{id}"}
        else:
            logger.warning(f"Sync requested for non-existent record ID: {id}")
            return {"success": False, "message": "Record not found"}
    except Exception as e:
        logger.error(f"Sync broadcast failed: {e}")
        return {"success": False, "message": str(e)}

# 5. REST API: POST /api/consult (Gemini/Mimo VLM + SQLite GB/T RAG)
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

        image_path = row[2]
        defect_types = row[7] or ""
        confidence = row[9] or 0.0

        # We will try to load the image and run the real VLM!
        vlm_detections = []
        vlm_raw = {}
        vlm_confidence = 0.95
        
        full_image_path = PROJECT_ROOT / image_path.lstrip("/")
        vlm_success = False
        vlm_error_msg = ""
        
        if full_image_path.exists():
            img = cv2.imread(str(full_image_path))
            if img is not None:
                detector = get_vlm_detector()
                if detector is not None:
                    try:
                        # Extract YOLO priors from record database (row[4] is yolo_result)
                        yolo_hints = []
                        try:
                            yolo_json = json.loads(row[4])
                            if isinstance(yolo_json, dict) and "defects" in yolo_json:
                                for d in yolo_json["defects"]:
                                    yolo_hints.append({
                                        "class_name": d.get("type", "unknown"),
                                        "confidence": d.get("confidence", 0.0),
                                        "bbox": d.get("box", [0, 0, 0, 0])
                                    })
                            elif isinstance(yolo_json, dict) and "detections" in yolo_json:
                                for d in yolo_json["detections"]:
                                    yolo_hints.append({
                                        "class_name": d.get("class_name", "unknown"),
                                        "confidence": d.get("confidence", 0.0),
                                        "bbox": d.get("bbox", [0, 0, 0, 0])
                                    })
                        except Exception as parse_err:
                            logger.warning(f"Failed to parse yolo_result for VLM guidance: {parse_err}")

                        logger.info(f"Invoking VLM model {detector.model} on {full_image_path} with {len(yolo_hints)} YOLO hints")
                        vlm_res = detector.detect(img, yolo_hints=yolo_hints)
                        if not vlm_res.error:
                            vlm_detections = vlm_res.detections
                            vlm_raw = vlm_res.raw_output or {}
                            vlm_success = True
                            logger.info(f"VLM successfully detected {len(vlm_detections)} issues.")
                        else:
                            vlm_error_msg = vlm_res.error
                            logger.warning(f"VLM returned error: {vlm_res.error}")
                    except Exception as e:
                        vlm_error_msg = str(e)
                        logger.error(f"Error during VLM inference: {e}")
            else:
                logger.warning(f"Failed to read image at {full_image_path}")
        else:
            logger.warning(f"Image does not exist at {full_image_path}")

        # 2. Run standard RAG standard recommendation
        from scripts.rag_demo import rag_analyze, query_knowledge_base
        
        # Build first matched standard for the UI (rag_standard)
        # We query the SQLite DB using the first defect type found, or general
        first_defect = defect_types.split(",")[0] if defect_types else "crazing"
        standards = query_knowledge_base(first_defect, "")
        if standards:
            rag_standard = {
                "standard_code": standards[0]["standard_code"],
                "title": standards[0]["title"],
                "content": standards[0]["content"]
            }
        else:
            rag_standard = {
                "standard_code": "GB/T 3280-2015",
                "title": "高精度不锈钢板外观判定通用标准",
                "content": "国家标准规范指出，对于带钢表面缺陷检测，任何大面积起皮、龟裂和宏观物理拉线划伤严重危害物理载截面性能，均视为失效，归于 P0 级判废范畴。"
            }

        # YOLO-Authoritative Strategy: Keep YOLO's original detected categories as the baseline for GB/T standards lookup
        yolo_detections = []
        if yolo_hints:
            yolo_detections = yolo_hints
        else:
            yolo_detections = [{"class_name": first_defect, "confidence": confidence}]
            
        eval_detections = yolo_detections
        reports = []
        
        for i, det in enumerate(eval_detections):
            cn = det.class_name if hasattr(det, 'class_name') else det.get("class_name", "?")
            desc = ""
            vlm_raw_response = vlm_raw.get("vlm_raw_response", {})
            if isinstance(vlm_raw_response.get("detections"), list) and i < len(vlm_raw_response["detections"]):
                desc = vlm_raw_response["detections"][i].get("bbox_description", "")
            
            report = rag_analyze(cn, desc)
            reports.append(report)

        analysis = "\n\n---\n\n".join(reports)
        
        # Fallback if VLM errored out completely: prepend alert
        if not vlm_success and vlm_error_msg:
            analysis = f"<div style='color:#E53E3E;padding:10px;background:#FFF5F5;border-radius:6px;border:1px solid #FED7D7;margin-bottom:10px'>⚠️ VLM 引擎异常: {vlm_error_msg[:100]}... (已自动降级为纯国标RAG分析)</div>\n\n" + analysis

        vlm_result = {
            "status": "completed",
            "confidence": float(vlm_confidence),
            "analysis": analysis,
            "detections": [d.to_dict() if hasattr(d, 'to_dict') else d for d in vlm_detections],
            "raw_output": vlm_raw
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

# 5.5 REST API: POST /api/consult_image (Run secure VLM on sandbox base64 image uploads)
@app.post("/api/consult_image")
def post_consult_image(payload: dict):
    image_base64 = payload.get("image_base64")
    question = payload.get("question")
    
    if not image_base64:
        return {"error": "Missing image_base64"}
        
    try:
        import base64
        img_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Invalid image data"}
            
        detector = get_vlm_detector()
        if detector is None:
            return {"error": "VLM engine not loaded"}
            
        logger.info(f"Invoking VLM model {detector.model} for sandbox consult image...")
        vlm_res = detector.detect(img, yolo_hints=None)
        
        if vlm_res.error:
            return {"error": vlm_res.error}
            
        vlm_raw_response = vlm_res.raw_output or {}
        vlm_raw_dict = vlm_raw_response.get("vlm_raw_response", {})
        
        from scripts.rag_demo import rag_analyze
        reports = []
        
        detections = vlm_res.detections
        if detections:
            for i, det in enumerate(detections):
                cn = det.class_name if hasattr(det, 'class_name') else det.get("class_name", "?")
                desc = ""
                if isinstance(vlm_raw_dict.get("detections"), list) and i < len(vlm_raw_dict["detections"]):
                    desc = vlm_raw_dict["detections"][i].get("bbox_description", "")
                report = rag_analyze(cn, desc)
                reports.append(report)
            analysis = "\n\n---\n\n".join(reports)
        else:
            # Fallback to general chat content
            analysis = vlm_raw_dict.get("description", "")
            if not analysis:
                choices = vlm_raw_response.get("choices", [])
                if choices:
                    analysis = choices[0].get("message", {}).get("content", "")
            if not analysis:
                analysis = "会诊完成。未见明显缺陷。"
            
        return {"analysis": analysis}
        
    except Exception as e:
        logger.error(f"Sandbox consult image VLM failed: {e}")
        return {"error": str(e)}

# 6. WebSocket Stream controller
@app.websocket("/camera/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for incoming messages if any
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
            else:
                logger.info(f"Received message from client: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Startup event triggers background threads
@app.on_event("startup")
def startup_event():
    # Make sure database is seeded
    db_dir = DB_PATH.parent
    db_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Spawn Camera scan loop thread (Real YOLOv8 inference pipeline)
    t1 = threading.Thread(target=run_realtime_pipeline, daemon=True)
    t1.start()
    
    # 2. Spawn Metrics loop thread
    t2 = threading.Thread(target=run_metrics_broadcaster, daemon=True)
    t2.start()
    
    logger.info("FastAPI Web Server Bridge successfully set up and running on port 8080.")

if __name__ == "__main__":
    uvicorn.run("vue_api_bridge:app", host="0.0.0.0", port=8080, reload=False)
