"""
训练接口单元测试。
"""

import pytest
from fastapi.testclient import TestClient
import os
import tempfile
import json
from pathlib import Path

from server import app, state, train_state, check_training_status
from src.db_manager import DBManager, InspectionRecord

@pytest.fixture
def client():
    # Setup temporary db
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    # Backup original state
    old_db = state.db
    state.db = DBManager(path)
    
    with TestClient(app) as test_client:
        yield test_client
        
    state.db.close()
    try:
        os.unlink(path)
    except Exception:
        pass
    # Restore original state
    state.db = old_db

def test_train_status_endpoint(client):
    response = client.get("/api/train/status")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "status" in data
    assert data["status"] in ("idle", "training", "completed", "failed")
    assert "correctedCount" in data

def test_train_start_no_data(client):
    # Try starting training when correctedCount is 0
    response = client.post("/api/train/start")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "没有找到可用于重训" in data["error"]

def test_train_start_with_data(client):
    # Insert a corrected bad case record
    record = InspectionRecord(
        image_path="test_badcase.jpg",
        review_status="corrected",
        final_result=json.dumps({
            "defects": [
                {
                    "type": "Scratches",
                    "bbox": [10, 20, 30, 40],
                    "confidence": 1.0
                }
            ]
        })
    )
    state.db.insert(record)
    
    # Mock subprocess.Popen and open files to avoid running real training in unit tests
    # Wait, we can test start but it might spawn the subprocess, which is fine since we enforce cpu and quick
    # But to prevent file not found on images, let's create a dummy image test_badcase.jpg
    with open("test_badcase.jpg", "w") as f:
        f.write("dummy image content")
        
    try:
        response = client.post("/api/train/start")
        assert response.status_code == 200
        data = response.json()
        
        # If train_state.process is started, it will be True
        assert data["success"] is True
        assert "已启动后台训练任务" in data["message"]
        
        # Poll status
        status_resp = client.get("/api/train/status")
        status_data = status_resp.json()
        assert status_data["success"] is True
        assert status_data["status"] in ("training", "completed", "failed")
    finally:
        if os.path.exists("test_badcase.jpg"):
            os.remove("test_badcase.jpg")
        # Clean up any badcase files exported in data/datasets/neu_det
        import shutil
        from server import PROJECT_ROOT
        train_img = PROJECT_ROOT / "data" / "datasets" / "neu_det" / "images" / "train" / "badcase_1.jpg"
        train_lbl = PROJECT_ROOT / "data" / "datasets" / "neu_det" / "labels" / "train" / "badcase_1.txt"
        if train_img.exists():
            os.remove(train_img)
        if train_lbl.exists():
            os.remove(train_lbl)
        if train_state.process:
            train_state.process.kill()
            train_state.process = None
            train_state.status = "idle"
