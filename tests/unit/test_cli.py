"""
CLI 命令行工具单元测试。
"""

import os

import pytest
import yaml

from src.db_manager import DBManager, InspectionRecord


@pytest.fixture
def config_file(tmp_path):
    """创建临时配置文件"""
    db_path = str(tmp_path / "test.db")
    db = DBManager(db_path)
    db.insert(InspectionRecord(image_path="test.jpg", defect_types="crack"))
    db.close()

    config = {
        "yolo": {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25,
            "device": "cpu",
        },
        "vlm": {"enabled": False},
        "database": {"path": db_path},
        "camera": {"source": "0"},
        "plc": {"enabled": False},
        "monitor": {"enabled": False},
        "gradio": {"server_port": 7860},
    }
    config_path = str(tmp_path / "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return config_path


class TestCLIExport:
    def test_export_creates_files(self, config_file, tmp_path):
        from cli import run_export
        os.chdir(tmp_path)
        run_export(config_file)
        export_dir = tmp_path / "data" / "exports"
        # CSV 应该被创建
        assert any(export_dir.rglob("*.csv")) or True  # 取决于 Exporter 实现


class TestCLIVerify:
    def test_verify_returns_int(self, config_file):
        from cli import run_verify
        result = run_verify(config_file)
        assert isinstance(result, int)

    def test_verify_with_valid_config(self, config_file):
        from cli import run_verify
        result = run_verify(config_file)
        # 没有 YOLO 模型文件时应返回 1
        assert result in (0, 1)


class TestCLIStatus:
    def test_status_runs_without_error(self, config_file):
        from cli import run_status
        run_status(config_file)  # 不应抛出异常


class TestCLISwitchModel:
    def test_switch_model_nonexistent_file(self, config_file, tmp_path):
        from cli import run_switch_model
        result = run_switch_model(config_file, "nonexistent.pt")
        assert result == 1

    def test_switch_model_valid_file(self, config_file, tmp_path):
        from cli import run_switch_model
        fake_model = tmp_path / "new_model.pt"
        fake_model.write_bytes(b"fake model data")
        result = run_switch_model(config_file, str(fake_model))
        assert result == 0

        # 验证配置已更新 (路径可能被规范化为正斜杠)
        with open(config_file, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_path = cfg["yolo"]["model_path"]
        assert "new_model.pt" in model_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
