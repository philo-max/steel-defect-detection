"""
YOLO 模型训练脚本 (RTX 5060 GPU 优化版)。

支持:
- yolov8n / yolov8s / yolov8m 多模型
- GPU 自适应 (RTX 5060 sm_120 兼容)
- 快速验证模式 (--quick)
- 训练后验证 + 导出 ONNX
- 断点续训 (--resume)
"""

import argparse
import shutil
from datetime import datetime
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
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[INFO] GPU: {gpu_name} ({vram:.1f} GB VRAM)")
            return device_arg
        else:
            print("[INFO] CUDA 不可用，使用 CPU 训练")
            return "cpu"
    except ImportError:
        return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="训练 YOLO 钢铁缺陷检测模型")
    parser.add_argument("--data", default="data/datasets/neu_det/dataset.yaml")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="预训练模型 (yolov8n/s/m/l)")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="初始学习率 (AdamW 建议 0.001-0.003)")
    parser.add_argument("--device", default="0", help="训练设备")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值")
    parser.add_argument("--quick", action="store_true",
                        help="快速验证: 仅训练 3 epoch")
    parser.add_argument("--resume", action="store_true", help="从上次中断续训")
    parser.add_argument("--export", action="store_true", help="训练后导出 ONNX")
    parser.add_argument("--output", default="models/weights/steel_defect.pt",
                        help="最终模型输出路径")
    return parser.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    if args.quick:
        args.epochs = 3
        args.imgsz = 320
        args.batch = 8
        print("[QUICK] 快速验证: epochs=3 imgsz=320 batch=8")

    device = detect_device(args.device)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] 数据集配置不存在: {args.data}")
        return

    # 项目根目录（脚本所在目录的父目录）
    project_root = Path(__file__).resolve().parent.parent
    project_dir = str(project_root / "runs" / "train")

    model_name = Path(args.model).stem  # e.g. "yolov8s"
    run_name = f"{model_name}_neu_det_{ts}" if not args.quick else "quick_test"

    print(f"[INFO] 模型: {args.model}")
    print(f"[INFO] 数据集: {args.data}")
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] Epochs: {args.epochs}  ImgSz: {args.imgsz}  Batch: {args.batch}  LR: {args.lr}")

    # 加载模型
    if args.resume:
        resume_path = Path(project_dir) / run_name / "weights" / "last.pt"
        if resume_path.exists():
            print(f"[INFO] 从 {resume_path} 续训")
            model = YOLO(str(resume_path))
        else:
            print(f"[WARN] 未找到续训权重，从头训练")
            model = YOLO(args.model)
    else:
        model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        lrf=0.01,            # cos 退火最终 LR 因子
        optimizer="AdamW",   # 小数据集用 AdamW
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,         # 余弦退火学习率
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        # 数据增强 (NEU-DET 缺陷检测专用)
        hsv_h=0.015,         # 色调变化 (钢表面颜色统一，小幅度)
        hsv_s=0.3,           # 饱和度
        hsv_v=0.2,           # 亮度
        degrees=5.0,         # 旋转角度
        translate=0.1,       # 平移
        scale=0.3,           # 缩放
        shear=2.0,           # 剪切
        flipud=0.3,          # 上下翻转 (缺陷可能出现在任意位置)
        fliplr=0.5,          # 左右翻转
        mosaic=0.8,          # mosaic 增强
        close_mosaic=10,     # 最后 10 epoch 关闭 mosaic
        erasing=0.1,         # 随机擦除
        # RTX 5060 sm_120 兼容 (FP16/AMP 不可用)
        half=False,
        amp=False,
    )

    # 复制最佳模型
    best_pt = Path(project_dir) / run_name / "weights" / "best.pt"
    if best_pt.exists():
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, output_path)
        print(f"\n[OK] 最佳模型: {output_path}")

        # 同时复制到模型目录带模型名
        named_copy = Path("models/weights") / f"{model_name}_steel.pt"
        shutil.copy2(best_pt, named_copy)
        print(f"[OK] 命名副本: {named_copy}")

    # 验证
    if not args.quick:
        print("\n[VAL] 验证集评估...")
        val_results = model.val()
        if val_results:
            print(f"  mAP50: {val_results.box.map50:.4f}")
            print(f"  mAP50-95: {val_results.box.map:.4f}")

    # 导出 ONNX
    if args.export:
        onnx_path = Path(args.output).with_suffix(".onnx")
        model.export(format="onnx", imgsz=args.imgsz, half=False)
        onnx_src = best_pt.with_suffix(".onnx")
        if onnx_src.exists():
            shutil.copy2(onnx_src, onnx_path)
            print(f"[OK] ONNX: {onnx_path}")

    print("\n[OK] 训练完成!")


if __name__ == "__main__":
    main()
