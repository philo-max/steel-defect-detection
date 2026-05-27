"""
核心模块单元测试。
"""

import numpy as np
import pytest

from src.base_detector import DetectionResult, InferenceResult


class TestDetectionResult:
    def test_create_detection(self):
        det = DetectionResult(
            bbox=[0.1, 0.2, 0.5, 0.6],
            class_name="crack",
            confidence=0.95,
            class_id=0,
        )
        assert det.class_name == "crack"
        assert det.confidence == 0.95
        assert det.bbox == [0.1, 0.2, 0.5, 0.6]

    def test_to_dict(self):
        det = DetectionResult(
            bbox=[0.1, 0.2, 0.3, 0.4],
            class_name="scratch",
            confidence=0.88,
        )
        d = det.to_dict()
        assert d["class_name"] == "scratch"
        assert d["confidence"] == 0.88
        assert "bbox" in d


class TestInferenceResult:
    def test_empty_result(self):
        result = InferenceResult()
        assert result.defect_count == 0
        assert not result.has_defect

    def test_with_detections(self):
        det = DetectionResult([0.1]*4, "crack", 0.9)
        result = InferenceResult(detections=[det])
        assert result.defect_count == 1
        assert result.has_defect

    def test_with_error(self):
        result = InferenceResult(error="模型加载失败")
        assert result.error == "模型加载失败"
        assert result.defect_count == 0

    def test_inference_time(self):
        result = InferenceResult(inference_time_ms=50.5)
        assert result.inference_time_ms == 50.5


# ========== YOLO 模型加载与推理测试 ==========

@pytest.mark.slow
class TestYOLODetectorModel:
    """YOLO 模型实际加载测试（需模型文件）"""

    @pytest.fixture(autouse=True)
    def _check_model(self):
        from pathlib import Path
        model_path = Path("models/weights/yolov8n.pt")
        if not model_path.exists():
            model_path = Path("yolov8n.pt")
        if not model_path.exists():
            pytest.skip("YOLO model not found for testing")
        self.model_path = str(model_path)

    def test_load_model(self):
        """加载 YOLO 模型不崩溃"""
        from src.detection_engine import YOLODetector
        detector = YOLODetector(
            model_path=self.model_path,
            conf_threshold=0.25,
            device="cpu",  # 测试用 CPU 避免 GPU 占用
        )
        detector.load_model()
        assert detector._model is not None

    def test_detect_with_image(self):
        """对合成图像执行推理"""
        from src.detection_engine import YOLODetector
        detector = YOLODetector(
            model_path=self.model_path,
            conf_threshold=0.95,  # 高阈值确保合成图像无检出
            device="cpu",
        )
        detector.load_model()
        img = np.ones((320, 320, 3), dtype=np.uint8) * 128
        result = detector.detect(img)
        assert result is not None
        assert hasattr(result, 'detections')

    def test_detect_with_high_conf(self):
        """高置信度阈值应减少检出"""
        from src.detection_engine import YOLODetector
        detector = YOLODetector(
            model_path=self.model_path,
            conf_threshold=0.99,
            device="cpu",
        )
        detector.load_model()
        img = np.ones((320, 320, 3), dtype=np.uint8) * 128
        result = detector.detect(img)
        assert result.defect_count == 0, "Synthetic gray image should have no defects"

    def test_confidence_threshold_setter(self):
        """动态修改置信度阈值生效"""
        from src.detection_engine import YOLODetector
        detector = YOLODetector(
            model_path=self.model_path,
            conf_threshold=0.25,
            device="cpu",
        )
        detector.load_model()
        detector.conf_threshold = 0.88
        assert detector.conf_threshold == 0.88


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
