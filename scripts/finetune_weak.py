"""
针对性微调 — 强化 inclusion（窄目标）和 rolled-in_scale（纹理混淆）。

策略:
- 从 best.pt 加载 (mAP50=0.886)
- 提升分辨率 640→960，帮助窄目标 inclusion
- 降低增强强度，保留纹理细节（帮助 rolled-in_scale）
- 低学习率 (lr=0.0005)，避免破坏已学好类别
- 50 epoch，patience=20
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="弱项专项微调")
    parser.add_argument("--weights", default="runs/train/yolov8s_neu_det_20260527_1013/weights/best.pt")
    parser.add_argument("--data", default="data/datasets/neu_det/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="models/weights/steel_defect.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"finetune_weak_{ts}"

    print(f"[INFO] 基础模型: {args.weights}")
    print(f"[INFO] 分辨率: {args.imgsz} (↑ 提升窄目标 inclusion 检测)")
    print(f"[INFO] 学习率: {args.lr} (↓ 精细微调)")
    print(f"[INFO] Epochs: {args.epochs}  Batch: {args.batch}")

    model = YOLO(args.weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        lrf=0.01,
        optimizer="AdamW",
        weight_decay=0.0005,
        warmup_epochs=2,
        cos_lr=True,
        device=args.device,
        workers=4,
        patience=args.patience,
        project="runs/train",
        name=run_name,
        exist_ok=True,
        # 弱增强 — 保留纹理细节（关键：rolled-in_scale 需要纹理信息）
        hsv_h=0.005,         # ↓ 色调几乎不变
        hsv_s=0.1,           # ↓ 饱和度变化小
        hsv_v=0.1,           # ↓ 亮度变化小
        degrees=3.0,         # ↓ 旋转幅度小
        translate=0.05,      # ↓ 平移小
        scale=0.2,           # ↓ 缩放小
        shear=1.0,           # ↓ 剪切小
        flipud=0.3,
        fliplr=0.5,
        mosaic=0.3,          # ↓ mosaic 低概率（保留原始纹理）
        close_mosaic=5,      # 早关闭
        erasing=0.05,        # ↓ 少擦除
        # RTX 5060 兼容
        half=False,
        amp=False,
    )

    # 保存模型
    best_pt = Path("runs/train") / run_name / "weights" / "best.pt"
    if best_pt.exists():
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 备份当前模型
        if output_path.exists():
            backup = output_path.with_suffix(".pt.bak")
            shutil.copy2(output_path, backup)
            print(f"[OK] 旧模型备份: {backup}")
        shutil.copy2(best_pt, output_path)
        print(f"[OK] 新模型: {output_path}")

    # 验证
    print("\n[VAL] 验证集评估...")
    val_results = model.val(data=args.data, imgsz=args.imgsz)
    if val_results:
        print(f"  mAP50: {val_results.box.map50:.4f}")
        print(f"  mAP50-95: {val_results.box.map:.4f}")
        # 逐类
        if hasattr(val_results, 'ap_class_index'):
            names = val_results.names
            for i, ap in enumerate(val_results.box.ap):
                print(f"  {names[i]:20s}: {ap:.4f}")

    print("\n[OK] 微调完成!")


if __name__ == "__main__":
    main()
