"""
VLM 引擎单元测试。
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.base_detector import InferenceResult


class TestVLMDetectorInit:
    def test_default_system_prompt(self):
        from src.vlm_engine import VLMDetector
        detector = VLMDetector()
        assert "钢铁" in detector._system_prompt
        assert "defect" in detector._system_prompt.lower() or "缺陷" in detector._system_prompt

    def test_custom_system_prompt(self):
        from src.vlm_engine import VLMDetector
        custom = "自定义提示词"
        detector = VLMDetector(system_prompt=custom)
        assert detector._system_prompt == custom

    def test_provider_detection_no_keys(self):
        from src.vlm_engine import VLMDetector
        detector = VLMDetector()
        assert detector._provider == "none"


class TestVLMDetectorDetect:
    def test_detect_without_load(self):
        from src.vlm_engine import VLMDetector
        detector = VLMDetector()
        result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert isinstance(result, InferenceResult)
        assert result.error is not None
        assert "未初始化" in result.error

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"})
    def test_detect_with_mock_client(self):
        from src.vlm_engine import VLMDetector
        detector = VLMDetector()
        detector.load_model()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"detections": []}'
        detector._client = MagicMock()
        detector._client.chat.completions.create.return_value = mock_response

        result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert isinstance(result, InferenceResult)
        assert result.error is None


class TestVLMDetectorParsing:
    def test_parse_valid_json(self):
        from src.vlm_engine import VLMDetector
        detector = VLMDetector()
        json_str = '{"detections": [{"class_name": "crack", "confidence": 0.9, "bbox_description": "左侧"}]}'
        result = detector._parse_response(json_str)
        assert isinstance(result, dict)
        assert len(result["detections"]) == 1
        assert result["detections"][0]["class_name"] == "crack"

    def test_parse_empty_detections(self):
        from src.vlm_engine import VLMDetector
        detector = VLMDetector()
        json_str = '{"detections": []}'
        result = detector._parse_response(json_str)
        assert result["detections"] == []

    def test_parse_invalid_json(self):
        from src.vlm_engine import VLMDetector
        detector = VLMDetector()
        result = detector._parse_response("not json")
        # 无效 JSON 应返回 dict (降级结果或原始数据)
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
