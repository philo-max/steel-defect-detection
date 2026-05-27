"""
YOLO 检测 Skill 实现。
"""

import sys
from pathlib import Path

# 将项目根添加到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np

from src.detection_engine import YOLODetector

# 全局检测器实例 (延迟加载)
_detector: YOLODetector | None = None


def _get_detector() -> YOLODetector:
    global _detector
    if _detector is None:
        _detector = YOLODetector()
        _detector.load_model()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _detector.warmup(dummy)
    return _detector


def detect(image_path: str) -> dict:
    """对图像执行 YOLO 缺陷检测"""
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"无法读取图像: {image_path}"}
    else:
        image = image_path

    detector = _get_detector()
    result = detector.detect(image)
    return result.to_dict()
