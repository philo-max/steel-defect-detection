"""
命令行工具 - 支持单张检测、批量导出和交互式 Shell。
"""

import sys
from pathlib import Path

import cv2
import yaml

from src.detection_engine import YOLODetector
from src.vlm_engine import VLMDetector
from src.db_manager import DBManager, InspectionRecord
from src.exporter import Exporter


def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_detect(config_path: str, image_path: str | None):
    """单张图像检测"""
    if image_path is None:
        print("请使用 --image 参数指定图像路径")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    cfg = _load_config(config_path)
    yolo_cfg = cfg.get("yolo", {})

    detector = YOLODetector(
        model_path=yolo_cfg.get("model_path", "models/weights/yolov8n.pt"),
        conf_threshold=yolo_cfg.get("conf_threshold", 0.25),
        device=yolo_cfg.get("device", "auto"),
    )

    try:
        detector.load_model()
    except FileNotFoundError:
        print("⚠️ YOLO 模型文件未找到，请先放置模型权重到 models/weights/")
        return

    result = detector.detect(image)

    print(f"\n{'='*50}")
    print(f"检测完成 | 耗时: {result.inference_time_ms:.1f}ms")
    print(f"缺陷数量: {result.defect_count}")

    if result.detections:
        print("\n检测结果:")
        for i, det in enumerate(result.detections, 1):
            print(f"  {i}. {det.class_name} (置信度: {det.confidence:.2f})")
            print(f"     位置: [{det.bbox[0]:.3f}, {det.bbox[1]:.3f}, {det.bbox[2]:.3f}, {det.bbox[3]:.3f}]")
    else:
        print("未检测到缺陷")

    if result.error:
        print(f"\n错误: {result.error}")
    print(f"{'='*50}\n")


def run_export(config_path: str):
    """导出检测数据"""
    cfg = _load_config(config_path)
    db = DBManager(cfg.get("database", {}).get("path", "data/inspection.db"))
    exporter = Exporter(db)

    csv_path = exporter.export_csv()
    html_path = exporter.export_html_report()
    badcase_dir = exporter.export_badcase()

    print(f"✅ CSV 导出: {csv_path}")
    print(f"✅ HTML 报告: {html_path}")
    print(f"✅ Bad Case: {badcase_dir}")


def run_cli(config_path: str):
    """交互式命令行 (预留)"""
    print("钢铁表面缺陷检测系统 CLI v1.0")
    print("目前请使用 --mode detect 或 --mode export")
    print("交互式 Shell 功能开发中...")
