"""
YOLO 检测引擎 - 基于 Ultralytics YOLO 的钢铁表面缺陷检测。
支持 YOLOv8/v10/v12 模型。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO
from loguru import logger

from .base_detector import BaseDetector, DetectionResult, InferenceResult
from .utils.bbox import bbox_iou

# NEU-DET 标准类别
NEU_DET_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

# 别名映射: 任何输入 -> NEU-DET 标准名
CLASS_NAME_ALIASES = {
    # crazing
    "crazing": "crazing", "crack": "crazing", "cracks": "crazing",
    "裂纹": "crazing", "裂缝": "crazing", "龟裂": "crazing",
    # inclusion
    "inclusion": "inclusion", "inclusions": "inclusion",
    "夹杂": "inclusion", "夹杂物": "inclusion",
    # patches
    "patches": "patches", "patch": "patches",
    "斑块": "patches", "斑": "patches",
    # pitted_surface
    "pitted_surface": "pitted_surface", "pitted surface": "pitted_surface",
    "pitted": "pitted_surface", "pit": "pitted_surface", "pits": "pitted_surface",
    "麻点": "pitted_surface", "凹坑": "pitted_surface", "坑": "pitted_surface",
    # rolled-in_scale
    "rolled-in_scale": "rolled-in_scale", "rolled_in_scale": "rolled-in_scale",
    "rolled in scale": "rolled-in_scale", "scale": "rolled-in_scale",
    "rolled-in": "rolled-in_scale", "scales": "rolled-in_scale",
    "氧化皮": "rolled-in_scale", "氧化": "rolled-in_scale", "鳞片": "rolled-in_scale",
    # scratches
    "scratches": "scratches", "scratch": "scratches",
    "划痕": "scratches", "划伤": "scratches", "擦痕": "scratches",
}


def _normalize_class_name(name: str) -> str:
    key = name.strip().lower()
    # 直接匹配
    if key in CLASS_NAME_ALIASES:
        return CLASS_NAME_ALIASES[key]
    # 尝试包含匹配
    for alias, canonical in CLASS_NAME_ALIASES.items():
        if alias in key or key in alias:
            return canonical
    # 保留原始名称（让app.py的模糊匹配处理）
    return key


class YOLODetector(BaseDetector):
    def __init__(
        self,
        model_path: str = "models/weights/yolov8n.pt",
        conf_threshold: float = 0.01,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: str = "auto",
        half: bool = False,
        augment: bool = True,
    ):
        super().__init__(name="yolo")
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device
        self.half = half
        self.augment = augment
        self._class_name_map: dict[int, str] = {}
        self._models: list[YOLO] = []
        self._model_paths: list[str] = []

    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        if model_path:
            self.model_path = model_path

        raw_paths = [p.strip() for p in self.model_path.replace(";", ",").split(",")]
        self._model_paths = []
        self._models = []
        used_fallbacks = set()

        for rp in raw_paths:
            if not rp:
                continue
            path = Path(rp)
            if not path.exists():
                fallback = Path("models/weights/yolov8n.pt")
                logger.warning(f"模型不存在: {rp}")
                if fallback.exists() and str(fallback) not in used_fallbacks:
                    logger.info(f"回退到: {fallback}")
                    path = fallback
                    used_fallbacks.add(str(fallback))
                else:
                    raise FileNotFoundError(f"模型文件不存在: {rp}")
            self._model_paths.append(str(path))

        for mp in self._model_paths:
            logger.info(f"加载模型: {mp}")
            model = YOLO(mp)

            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda:0"
                    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                else:
                    self.device = "cpu"
                    logger.info("CPU 推理")

            if self.device != "cpu" and "cuda" in self.device:
                if torch.cuda.is_available():
                    try:
                        if hasattr(model, "to") and not mp.endswith(".onnx") and not mp.endswith(".engine"):
                            model.to(self.device)
                    except Exception as e:
                        logger.warning(f"无法将模型移动到 {self.device}: {e}")
                else:
                    self.device = "cpu"

            self._models.append(model)
            names = model.names if model.names else {}
            logger.info(f"模型类别: {names}")

        if self._models:
            self._build_class_map()

    def _build_class_map(self) -> None:
        primary = self._models[0]
        names = primary.names if primary.names else {}
        for idx, name in names.items():
            norm = _normalize_class_name(name)
            self._class_name_map[int(idx)] = norm
        logger.info(f"类别映射: {self._class_name_map}")

    def detect(self, image: np.ndarray) -> InferenceResult:
        if not self._models:
            return InferenceResult(error="模型未加载")

        start = time.perf_counter()

        if len(self._models) == 1:
            return self._detect_single(self._models[0], image, start)
        else:
            return self._detect_ensemble(image, start)

    def _detect_single(self, model: YOLO, image: np.ndarray, start_time: float) -> InferenceResult:
        mp = self._model_paths[0] if self._model_paths else ""
        is_onnx = mp.endswith(".onnx") or mp.endswith(".engine")
        try:
            results = model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                half=self.half if not is_onnx else False,
                augment=self.augment if not is_onnx else False,
                verbose=False,
            )
        except Exception as e:
            return InferenceResult(
                inference_time_ms=self._measure_time(start_time),
                error=f"YOLO 推理异常: {e}",
            )

        elapsed = self._measure_time(start_time)
        result = results[0]
        h, w = result.orig_shape

        detections: list[DetectionResult] = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                norm_box = [
                    float(box[0]) / w,
                    float(box[1]) / h,
                    float(box[2]) / w,
                    float(box[3]) / h,
                ]
                class_name = self._class_name_map.get(cls_id, f"class_{cls_id}")
                detections.append(
                    DetectionResult(
                        bbox=norm_box,
                        class_name=class_name,
                        confidence=float(conf),
                        class_id=int(cls_id),
                    )
                )

        return InferenceResult(
            detections=detections,
            inference_time_ms=elapsed,
            image_shape=result.orig_shape,
            raw_output={
                "boxes": result.boxes.data.cpu().numpy().tolist() if result.boxes else [],
                "model_path": self._model_paths[0] if self._model_paths else "",
                "class_map": self._class_name_map,
            },
        )

    def _detect_ensemble(self, image: np.ndarray, start_time: float) -> InferenceResult:
        all_detections: list[DetectionResult] = []
        total_infer_ms = 0.0
        orig_shape: tuple = (0, 0)

        for model, mp in zip(self._models, self._model_paths):
            is_onnx = mp.endswith(".onnx") or mp.endswith(".engine")
            t0 = time.perf_counter()
            try:
                results = model.predict(
                    source=image,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.img_size,
                    device=self.device,
                    half=self.half if not is_onnx else False,
                    augment=self.augment if not is_onnx else False,
                    verbose=False,
                )
            except Exception as e:
                logger.warning(f"模型推理失败: {e}")
                continue
            total_infer_ms += (time.perf_counter() - t0) * 1000.0
            result = results[0]
            orig_shape = result.orig_shape
            h, w = orig_shape

            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                norm_box = [
                    float(box[0]) / w,
                    float(box[1]) / h,
                    float(box[2]) / w,
                    float(box[3]) / h,
                ]
                class_name = self._class_name_map.get(int(cls_id), f"class_{cls_id}")
                all_detections.append(
                    DetectionResult(
                        bbox=norm_box,
                        class_name=class_name,
                        confidence=float(conf),
                        class_id=int(cls_id),
                    )
                )

        # 应用多模型 NMS 融合去重
        detections = self._nms(all_detections, self.iou_threshold)

        return InferenceResult(
            detections=detections,
            inference_time_ms=total_infer_ms,
            image_shape=orig_shape,
        )

    def _nms(self, detections: list[DetectionResult], iou_threshold: float) -> list[DetectionResult]:
        if not detections:
            return []

        from collections import defaultdict
        class_groups = defaultdict(list)
        for det in detections:
            class_groups[det.class_name].append(det)

        keep_all = []
        for class_name, group in class_groups.items():
            # 依置信度从高到低排序
            group.sort(key=lambda x: x.confidence, reverse=True)
            keep = []
            while group:
                best = group.pop(0)
                keep.append(best)
                # 剔除与当前框 IoU 过大的重叠预测
                group = [d for d in group if bbox_iou(best.bbox, d.bbox) < iou_threshold]
            keep_all.extend(keep)

        # 最终仍按置信度排序
        keep_all.sort(key=lambda x: x.confidence, reverse=True)
        return keep_all

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_name_map.copy()

    @property
    def is_ensemble(self) -> bool:
        return len(self._models) > 1

    @property
    def model_count(self) -> int:
        return len(self._models)
