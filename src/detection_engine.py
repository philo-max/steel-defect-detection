"""
YOLO 检测引擎 - 基于 Ultralytics YOLO 的钢铁表面缺陷检测。

支持 YOLOv8/v10/v12 模型，可切换 ONNX/TensorRT 推理后端。
"""

from pathlib import Path
from typing import Optional
import time

import numpy as np
from ultralytics import YOLO

from .base_detector import BaseDetector, InferenceResult, DetectionResult


class YOLODetector(BaseDetector):
    """YOLO 目标检测器"""

    def __init__(
        self,
        model_path: str = "models/weights/yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: str = "auto",
        half: bool = False,
        augment: bool = False,
    ):
        super().__init__(name="yolo")
        # auto 模式: 自动检测最佳设备
        if device == "auto":
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device
        self.half = half
        self.augment = augment
        self._model: Optional[YOLO] = None

    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """加载 YOLO 模型，自动回退到 yolov8n.pt"""
        if model_path:
            self.model_path = model_path

        path = Path(self.model_path)
        if not path.exists():
            fallback = Path("models/weights/yolov8n.pt")
            print(f"[WARN] 自定义模型不存在: {self.model_path}")
            if fallback.exists():
                print(f"[INFO] 回退到预训练模型: {fallback}")
                path = fallback
            else:
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self._model = YOLO(str(path))
        # 仅在 CUDA 可用时迁移到 GPU
        if self.device != "cpu" and "cuda" in self.device:
            import torch
            if torch.cuda.is_available():
                self._model.to(self.device)
            else:
                print(f"[WARN] CUDA 不可用，YOLO 将使用 CPU 推理")
                self.device = "cpu"

        self._warm = False

    def detect(self, image: np.ndarray) -> InferenceResult:
        """对单张图像执行 YOLO 检测"""
        if self._model is None:
            return InferenceResult(error="模型未加载，请先调用 load_model()")

        start = time.perf_counter()

        try:
            results = self._model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                half=self.half,
                augment=self.augment,
                verbose=False,
            )
        except Exception as e:
            return InferenceResult(
                inference_time_ms=self._measure_time(start),
                error=f"YOLO 推理异常: {e}",
            )

        elapsed = self._measure_time(start)
        result = results[0]

        # 获取类别名称映射
        names = self._model.names if self._model.names else {}

        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # 绝对坐标
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            h, w = result.orig_shape
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                # 归一化到 [0, 1]
                norm_box = [
                    float(box[0]) / w,
                    float(box[1]) / h,
                    float(box[2]) / w,
                    float(box[3]) / h,
                ]
                detections.append(DetectionResult(
                    bbox=norm_box,
                    class_name=names.get(cls_id, f"class_{cls_id}"),
                    confidence=float(conf),
                    class_id=int(cls_id),
                ))

        return InferenceResult(
            detections=detections,
            inference_time_ms=elapsed,
            image_shape=result.orig_shape,
            raw_output={"boxes": result.boxes.data.cpu().numpy().tolist() if result.boxes else []},
        )

    @property
    def class_names(self) -> dict[int, str]:
        if self._model and self._model.names:
            return self._model.names
        return {}
