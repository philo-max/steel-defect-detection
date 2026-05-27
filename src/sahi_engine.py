"""
SAHI 滑窗推理模块 — 大图小目标检测增强。

功能：
1. 将高分辨率图像切分为重叠的滑窗切片
2. 对每个切片独立执行 YOLO 推理
3. 合并所有切片的检测结果（坐标映射回原图 + 跨切片 NMS）
4. 支持自适应切片策略（根据图像尺寸和目标大小）

原理：
    在 8K 线阵相机图像中，微小裂纹/孔洞可能只占几十个像素。
    直接缩放到 640×640 会让这些小目标消失。
    SAHI 保持原始分辨率，通过滑窗切图让每个小目标在切片中足够大。

使用方式:
    from src.sahi_engine import SAHIDetector
    sahi = SAHIDetector(yolo_detector, slice_size=640, overlap=0.2)
    result = sahi.detect(large_image)
"""

import time
from typing import Optional

import cv2
import numpy as np

from .base_detector import BaseDetector, InferenceResult, DetectionResult


class SAHIDetector(BaseDetector):
    """
    SAHI 滑窗检测器 — 包装任意 BaseDetector 实现大图小目标检测。

    工作流:
    1. 将输入图像切分为 slice_size × slice_size 的重叠滑窗
    2. 对每个滑窗调用底层检测器
    3. 将切片坐标映射回原图坐标
    4. 跨切片 NMS 去重
    """

    def __init__(
        self,
        detector: BaseDetector,
        slice_size: int = 640,
        overlap_ratio: float = 0.2,
        min_slice_area_ratio: float = 0.1,
        postprocess_match_metric: str = "ios",  # ios (intersection over smaller)
        postprocess_match_threshold: float = 0.5,
        postprocess_class_agnostic: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            detector: 底层检测器（YOLODetector / TensorRTDetector）
            slice_size: 切片尺寸（宽=高）
            overlap_ratio: 切片间重叠比例 (0.0-0.5，默认 0.2)
            min_slice_area_ratio: 最后一行/列切片的最小面积比例，低于此则丢弃
            postprocess_match_metric: 合并时匹配度量 "ios" 或 "iou"
            postprocess_match_threshold: 合并 NMS 阈值
            postprocess_class_agnostic: 合并时是否跨类别抑制
            verbose: 是否打印详细日志
        """
        super().__init__(name="sahi")
        self._detector = detector
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.min_slice_area_ratio = min_slice_area_ratio
        self.postprocess_match_metric = postprocess_match_metric
        self.postprocess_match_threshold = postprocess_match_threshold
        self.postprocess_class_agnostic = postprocess_class_agnostic
        self.verbose = verbose

        self._model_loaded = False

    # ==================== 检测器接口 ====================

    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """委托给底层检测器"""
        self._detector.load_model(model_path, **kwargs)
        self._model_loaded = True

    def detect(self, image: np.ndarray) -> InferenceResult:
        """
        对单张图像执行 SAHI 滑窗检测。

        如果图像尺寸 <= slice_size，则直接委托给底层检测器（无开销）。
        """
        if not self._model_loaded:
            return InferenceResult(error="模型未加载，请先调用 load_model()")

        start = time.perf_counter()
        h, w = image.shape[:2]

        # 小图直接推理，无需滑窗
        if h <= self.slice_size and w <= self.slice_size:
            return self._detector.detect(image)

        # ===== SAHI 滑窗推理 =====
        slice_bboxes = self._compute_slice_bboxes(h, w)

        if self.verbose:
            print(f"[SAHI] 图像 {w}×{h}, 切片 {len(slice_bboxes)} 个 "
                  f"(slice={self.slice_size}, overlap={self.overlap_ratio:.0%})")

        all_detections: list[tuple[DetectionResult, int, int, int, int]] = []
        total_slice_time = 0.0

        for x1, y1, x2, y2 in slice_bboxes:
            # 裁剪切片
            slice_img = image[y1:y2, x1:x2]

            # 底层推理
            result = self._detector.detect(slice_img)
            total_slice_time += result.inference_time_ms

            if result.error:
                if self.verbose:
                    print(f"[SAHI] 切片 ({x1},{y1})-({x2},{y2}) 推理失败: {result.error}")
                continue

            # 将切片坐标映射回原图
            for det in result.detections:
                # 归一化坐标 → 切片像素坐标 → 原图像素坐标
                bx1 = det.bbox[0] * (x2 - x1) + x1
                by1 = det.bbox[1] * (y2 - y1) + y1
                bx2 = det.bbox[2] * (x2 - x1) + x1
                by2 = det.bbox[3] * (y2 - y1) + y1

                all_detections.append((det, int(bx1), int(by1), int(bx2), int(by2)))

        # 跨切片 NMS 合并
        merged = self._merge_across_slices(all_detections, h, w)

        elapsed = self._measure_time(start)

        return InferenceResult(
            detections=merged,
            inference_time_ms=elapsed,
            image_shape=(h, w),
            raw_output={
                "num_slices": len(slice_bboxes),
                "total_slice_time_ms": total_slice_time,
            },
        )

    # ==================== 切片计算 ====================

    def _compute_slice_bboxes(self, image_h: int, image_w: int) -> list[tuple[int, int, int, int]]:
        """
        计算滑窗切片坐标。

        返回: [(x1, y1, x2, y2), ...] 每个切片的像素坐标
        """
        stride = int(self.slice_size * (1 - self.overlap_ratio))

        bboxes = []
        y = 0
        while y < image_h:
            y2 = min(y + self.slice_size, image_h)

            # 跳过过小的最后一行
            if image_h - y < self.slice_size * self.min_slice_area_ratio:
                break

            x = 0
            while x < image_w:
                x2 = min(x + self.slice_size, image_w)

                # 跳过过小的最后一列
                if image_w - x < self.slice_size * self.min_slice_area_ratio:
                    break

                bboxes.append((x, y, x2, y2))
                x += stride

            y += stride

        return bboxes

    # ==================== 跨切片合并 ====================

    def _merge_across_slices(
        self,
        detections: list[tuple[DetectionResult, int, int, int, int]],
        image_h: int,
        image_w: int,
    ) -> list[DetectionResult]:
        """
        跨切片 NMS 合并。

        策略：按置信度排序 → 逐个保留 → 抑制重叠度高的后续框
        """
        if not detections:
            return []

        # 按置信度降序
        items = sorted(detections, key=lambda x: x[0].confidence, reverse=True)

        kept: list[DetectionResult] = []
        kept_boxes: list[list[int]] = []  # 像素坐标

        for det, bx1, by1, bx2, by2 in items:
            keep = True
            box_a = [bx1, by1, bx2, by2]

            for box_b in kept_boxes:
                if self._should_suppress(box_a, box_b, det.class_id,
                                         kept[kept_boxes.index(box_b)].class_id):
                    keep = False
                    break

            if keep:
                # 归一化到原图
                norm_box = [
                    bx1 / image_w,
                    by1 / image_h,
                    bx2 / image_w,
                    by2 / image_h,
                ]
                kept.append(DetectionResult(
                    bbox=norm_box,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    class_id=det.class_id,
                ))
                kept_boxes.append(box_a)

        return kept

    def _should_suppress(
        self,
        box_a: list[int],
        box_b: list[int],
        class_id_a: int,
        class_id_b: int,
    ) -> bool:
        """判断 box_a 是否应被 box_b 抑制"""
        if not self.postprocess_class_agnostic and class_id_a != class_id_b:
            return False

        if self.postprocess_match_metric == "ios":
            return self._compute_ios(box_a, box_b) > self.postprocess_match_threshold
        else:
            return self._compute_iou(box_a, box_b) > self.postprocess_match_threshold

    @staticmethod
    def _compute_iou(box_a: list[int], box_b: list[int]) -> float:
        """计算 IoU"""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        return inter / (area_a + area_b - inter + 1e-6)

    @staticmethod
    def _compute_ios(box_a: list[int], box_b: list[int]) -> float:
        """计算 IOS (Intersection over Smaller area) — 对小目标更友好"""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        return inter / (min(area_a, area_b) + 1e-6)

    # ==================== 自适应切片策略 ====================

    @classmethod
    def auto_sahi(
        cls,
        detector: BaseDetector,
        image_size: tuple[int, int],
        min_defect_size_px: int = 20,
    ) -> "SAHIDetector":
        """
        自适应创建 SAHI 检测器。

        根据图像尺寸和期望检测的最小缺陷尺寸，自动选择最优 slice_size。

        Args:
            detector: 底层检测器
            image_size: 图像 (h, w)
            min_defect_size_px: 最小缺陷像素数（期望检测的最小目标）

        Returns:
            配置好的 SAHIDetector
        """
        h, w = image_size
        max_dim = max(h, w)

        # 策略：让 slice_size 使最小缺陷在切片中至少占 8% 的尺寸
        # slice_size = min_defect_px / 0.08
        ideal_slice = int(min_defect_size_px / 0.08)

        # 限制在 [480, 1280] 之间
        slice_size = max(480, min(1280, ideal_slice))

        # 重叠率：大图用更多重叠
        if max_dim > 4000:
            overlap = 0.25
        elif max_dim > 2000:
            overlap = 0.20
        else:
            overlap = 0.15

        return cls(
            detector=detector,
            slice_size=slice_size,
            overlap_ratio=overlap,
            verbose=True,
        )

    @property
    def is_ready(self) -> bool:
        return self._detector.is_ready


# ==================== 自检 ====================

if __name__ == "__main__":
    import sys

    print("SAHI 滑窗检测器自检\n" + "=" * 50)

    # 创建模拟大图 (3000×2000)
    large_img = np.ones((2000, 3000, 3), dtype=np.uint8) * 128
    # 在右下角画一个小方块模拟缺陷（原图 3000×2000 中只占 30×30）
    cv2.rectangle(large_img, (2800, 1850), (2830, 1880), (0, 0, 0), -1)

    # 计算切片
    from src.fast_screener import FastScreener
    class MockDetector(BaseDetector):
        def load_model(self, model_path=None, **kwargs): pass
        def detect(self, image):
            return InferenceResult(detections=[], inference_time_ms=1.0)

    sahi = SAHIDetector(MockDetector(), slice_size=640, overlap_ratio=0.2)
    sahi._model_loaded = True

    slices = sahi._compute_slice_bboxes(2000, 3000)
    print(f"图像 3000×2000 → {len(slices)} 个切片 (640×640, overlap=20%)")

    # 验证切片覆盖
    covered = np.zeros((2000, 3000), dtype=np.uint8)
    for x1, y1, x2, y2 in slices:
        covered[y1:y2, x1:x2] = 1
    coverage = covered.mean()
    print(f"覆盖率: {coverage:.1%}")

    print(f"\n小目标 30×30 在 3000×2000 原图中占比: {30*30/(3000*2000)*100:.3f}%")
    print(f"同目标在 640×640 切片中占比: {30*30/(640*640)*100:.2f}%")
    print(f"放大倍数: {(640*640)/(30*30):.0f}×")

    sys.exit(0)
