"""
检测器基类 - 定义统一的检测器接口。

所有检测引擎 (YOLO, VLM, ...) 必须继承此基类并实现 detect 方法。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np


@dataclass
class DetectionResult:
    """单条检测结果"""

    bbox: list[float]          # [x1, y1, x2, y2] 归一化坐标 (0-1)
    class_name: str            # 缺陷类别名称
    confidence: float          # 置信度 (0-1)
    class_id: int = 0          # 类别 ID

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
        }


@dataclass
class InferenceResult:
    """单次推理完整结果"""

    detections: list[DetectionResult] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_shape: tuple[int, int] = (0, 0)
    raw_output: Optional[dict] = None   # 引擎原始输出
    error: Optional[str] = None

    @property
    def defect_count(self) -> int:
        return len(self.detections)

    @property
    def has_defect(self) -> bool:
        return self.defect_count > 0

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "inference_time_ms": self.inference_time_ms,
            "image_shape": list(self.image_shape),
            "defect_count": self.defect_count,
            "error": self.error,
        }


class BaseDetector(ABC):
    """检测器抽象基类"""

    def __init__(self, name: str = "base"):
        self.name = name
        self._warm = False

    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> None:
        """加载模型权重"""
        ...

    @abstractmethod
    def detect(self, image: np.ndarray) -> InferenceResult:
        """对单张图像执行检测"""
        ...

    def warmup(self, image: np.ndarray) -> None:
        """预热模型 (首次推理前调用)"""
        _ = self.detect(image)
        self._warm = True

    @property
    def is_ready(self) -> bool:
        return self._warm

    @staticmethod
    def _measure_time(start: float) -> float:
        """计算耗时 (毫秒)"""
        return (time.perf_counter() - start) * 1000.0
