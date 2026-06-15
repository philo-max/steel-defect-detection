"""
YOLO 模型训练脚本 (GPU 优化版)。

支持:
- 从 YAML 数据集配置训练
- CPU/GPU 自适应
- 快速验证模式 (--quick)
- 训练后导出 ONNX
"""

import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

# NEU-DET 标准类别
CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


def detect_device(device_arg: str) -> str:
    """自动检测最佳训练设备"""
    if device_arg.lower() == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[INFO] GPU 可用: {gpu_name}")
            return device_arg  # 可能是 "0" 或 "cuda:0"
        else:
            print("[INFO] CUDA 不可用，使用 CPU 训练")
            return "cpu"
    except ImportError:
        return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="训练 YOLO 钢铁缺陷检测模型")
    parser.add_argument("--data", default="data/datasets/neu_det/dataset.yaml",
                        help="数据集配置文件")
    parser.add_argument("--model", default="yolov8n.pt", help="预训练模型")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸 (GPU 建议 640)")
    parser.add_argument("--batch", type=int, default=16, help="批次大小 (GPU 建议 8-16)")
    parser.add_argument("--lr", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--device", default="0", help="训练设备")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")
    parser.add_argument("--quick", action="store_true",
                        help="快速验证: 仅训练 5 epoch 验证流水线")
    parser.add_argument("--export", action="store_true", help="训练后导出 ONNX")
    parser.add_argument("--output", default="models/weights/steel_defect.pt",
                        help="最终模型输出路径")
    return parser.parse_args()


def main():
    args = parse_args()

    # 快速验证模式
    if args.quick:
        args.epochs = 5
        args.imgsz = 160
        args.batch = 2
        print("[QUICK] 快速验证模式: epochs=5 imgsz=160 batch=2")

    device = detect_device(args.device)

    # 检查数据集
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] 数据集配置不存在: {args.data}")
        print("  请先运行: python scripts/download_neu_det.py")
        return

    print(f"[INFO] 数据集: {args.data}")
    print(f"[INFO] 预训练模型: {args.model}")
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] Epochs: {args.epochs}  ImgSz: {args.imgsz}  Batch: {args.batch}")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project="runs/train",
        name="steel_defect",
        exist_ok=True,
        # 数据增强 (适中，避免过拟合小数据集)
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=2.0,
        flipud=0.3,
        fliplr=0.5,
        mosaic=0.0,       # 小图禁用 mosaic
        erasing=0.1,
    )

    # 复制最佳模型到指定路径
    best_pt = Path("runs/train/steel_defect/weights/best.pt")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if best_pt.exists():
        shutil.copy2(best_pt, output_path)
        print(f"\n✓ 最佳模型已保存: {output_path}")

    # 验证
    if not args.quick:
        print("\n[VAL] 验证集评估...")
        model.val()

    # 导出 ONNX
    if args.export:
        onnx_path = output_path.with_suffix(".onnx")
        model.export(format="onnx")
        onnx_src = best_pt.with_suffix(".onnx")
        if onnx_src.exists():
            shutil.copy2(onnx_src, onnx_path)
            print(f"✓ ONNX 模型: {onnx_path}")

    print("\n✓ 训练完成!")


if __name__ == "__main__":
    main()
