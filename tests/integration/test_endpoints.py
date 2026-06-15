import os
# Force default password for testing to match test credentials
os.environ["APP_DEFAULT_PASSWORD"] = "123456"

import pytest
from fastapi.testclient import TestClient
import json
import base64

from server import app, state

# Initialize app state for testing if not done
@pytest.fixture(scope="module", autouse=True)
def init_state():
    import tempfile
    from unittest.mock import patch
    import yaml

    # Create temporary database path
    fd, temp_db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    orig_safe_load = yaml.safe_load
    def mock_safe_load(stream):
        cfg = orig_safe_load(stream)
        if cfg and "database" in cfg:
            cfg["database"]["path"] = temp_db_path
        return cfg

    with patch("yaml.safe_load", side_effect=mock_safe_load):
        if not state.db:
            state.init()

    yield

    # Clean up
    if state.db:
        state.db.close()
    try:
        os.unlink(temp_db_path)
    except Exception:
        pass


@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def auth_headers(client):
    # Log in to get session token
    res = client.post("/api/login", json={"username": "admin", "password": "123456"})
    data = res.json()
    assert data["success"] is True
    token = data["token"]
    return {"Authorization": f"Bearer {token}"}

def test_login(client):
    res = client.post("/api/login", json={"username": "admin", "password": "123456"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["role"] == "admin"

    # Wrong credentials
    res = client.post("/api/login", json={"username": "admin", "password": "wrong"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False

def test_kpi_endpoint(client):
    res = client.get("/api/kpi")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "leakageRate" in data
    assert "overkillRate" in data
    assert "avgDelayMs" in data
    assert "hardware" in data

def test_export_endpoints(client):
    # Test CSV export
    res = client.get("/api/export/csv")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]

    # Test HTML report export
    res = client.get("/api/export/html")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]

    # Test Bad Case ZIP export
    res = client.get("/api/export/badcase")
    assert res.status_code == 200
    assert "application/zip" in res.headers["content-type"]

def test_detect_routing_preset(client, auth_headers):
    # Test preset image detect routing
    # Load a real sample image from data/images
    import os
    import base64
    import numpy as np
    import cv2
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img_path = os.path.join(base_dir, "data", "images", "img_20260608_141256_318476.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(base_dir, "data", "images", "sample_steel.jpg")
    
    if not os.path.exists(img_path):
        # Create a small test image if no sample exists
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:] = (128, 128, 128)
        _, buf = cv2.imencode(".jpg", test_img)
        img_base64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
    else:
        with open(img_path, "rb") as f:
            img_base64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
    
    # Send a request with selectedSampleId = "scratch" to hit the preset
    res = client.post("/api/detect", json={
        "image": img_base64,
        "selectedSampleId": "scratch",
        "filename": "scratch.jpg"
    }, headers=auth_headers)
    
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "engine" in data["data"]
    assert len(data["data"]["defects"]) > 0
    assert data["data"]["defects"][0]["severity"] in ("Low", "Medium", "High")
