"""
SAHI 推理验证 — 测试滑窗切片对弱项 inclusion/rolled-in-scale 的提升效果。

用法: python scripts/validate_sahi.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.sahi_engine import SAHIDetector
from src.detection_engine import YOLODetector


def main():
    # 加载模型
    model = YOLO("models/weights/steel_defect.pt")
    detector = YOLODetector(
        model_path="models/weights/steel_defect.pt",
        conf_threshold=0.05,
        iou_threshold=0.45,
        device="0",
    )
    detector._model = model

    # 初始化 SAHI (slice=640, overlap=0.2)
    sahi = SAHIDetector(
        detector=detector,
        slice_size=640,
        overlap_ratio=0.2,
        verbose=False,
    )
    sahi._model_loaded = True  # 手动注入已加载的底层模型

    # 加载验证集
    val_dir = Path("data/datasets/neu_det/images/val")
    label_dir = Path("data/datasets/neu_det/labels/val")
    images = sorted(val_dir.glob("*.jpg"))[:30]  # 前30张快速测试

    print(f"测试 {len(images)} 张图像")
    print("=" * 60)

    class_names = ["crazing", "inclusion", "patches",
                   "pitted_surface", "rolled-in_scale", "scratches"]

    yolo_total, sahi_total = 0, 0
    yolo_by_class = {c: 0 for c in class_names}
    sahi_by_class = {c: 0 for c in class_names}
    total_gt = {c: 0 for c in class_names}
    sahi_triggered = False

    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # 读取 ground truth
        lbl_path = label_dir / f"{img_path.stem}.txt"
        with open(lbl_path) as f:
            for line in f:
                cls_id = int(line.split()[0])
                total_gt[class_names[cls_id]] += 1

        # SAHI 检测（仅大图 > 1000px 才启用）
        if max(w, h) > 1000:
            sahi_triggered = True
            result = sahi.detect(img)
            for d in result.detections:
                if d.confidence > 0.05:
                    sahi_total += 1
                    if d.class_name in sahi_by_class:
                        sahi_by_class[d.class_name] += 1
                    else:
                        print(f"[WARN] SAHI 返回未知类别: {d.class_name}")

        # YOLO 直接检测（对照组）
        result = detector.detect(img)
        for d in result.detections:
            if d.confidence > 0.05:
                yolo_total += 1
                if d.class_name in yolo_by_class:
                    yolo_by_class[d.class_name] += 1
                else:
                    print(f"[WARN] YOLO 返回未知类别: {d.class_name}")

    if not sahi_triggered:
        print("[INFO] 所有图像均小于 1000px，SAHI 滑窗未触发（NEU-DET 小图不适用 SAHI）\n")

    print(f"\n{'类别':20s} {'GT':>5s} {'YOLO':>5s} {'SAHI':>5s} {'提升':>6s}")
    print("-" * 50)
    for cn in class_names:
        y = yolo_by_class[cn]
        s = sahi_by_class[cn]
        gt = total_gt[cn]
        delta = f"+{s-y:+d}" if s > y else f"{s-y:+d}"
        print(f"{cn:20s} {gt:5d} {y:5d} {s:5d} {delta:>6s}")

    print("-" * 50)
    print(f"{'总计':20s} {sum(total_gt.values()):5d} {yolo_total:5d} {sahi_total:5d}")

    # 弱项专项
    print(f"\n弱项汇总:")
    for cn in ["inclusion", "rolled-in_scale"]:
        y = yolo_by_class[cn]
        s = sahi_by_class[cn]
        print(f"  {cn}: YOLO={y} SAHI={s} (提升 {s-y:+d})")


if __name__ == "__main__":
    main()
