"""
VLM 检测 Skill 实现。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2

from src.vlm_engine import VLMDetector

_detector: VLMDetector | None = None


def _get_detector() -> VLMDetector:
    global _detector
    if _detector is None:
        _detector = VLMDetector()
        _detector.load_model()
    return _detector


def analyze(image_path: str) -> dict:
    """对图像执行 VLM 缺陷分析"""
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"无法读取图像: {image_path}"}
    else:
        image = image_path

    detector = _get_detector()
    result = detector.detect(image)
    return result.to_dict()
