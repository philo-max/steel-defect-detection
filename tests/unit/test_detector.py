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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
