"""
Copy-Paste 缺陷数据增强脚本 — 缓解工业缺陷样本极度不均衡问题。

原理：
    工业生产中缺陷是小概率事件，正常样本远多于缺陷样本。
    从有标注的缺陷图像中裁剪出缺陷区域，随机拼贴到正常图像上，
    生成大量合成缺陷样本，显著提升模型对小样本缺陷类型的泛化能力。

引用：
    Ghiasi et al. "Simple Copy-Paste is a Strong Data Augmentation
    for Instance Segmentation" (CVPR 2021)

功能：
1. 从 YOLO 标注中提取缺陷区域（按 bbox 裁剪）
2. 随机粘贴到正常（无缺陷）图像上
3. 几何变换（旋转 ±30°、缩放 0.7-1.5×、水平翻转）
4. 泊松融合/Alpha 混合避免硬边界
5. 自动更新 YOLO 标签文件
6. 粘贴数量可控、重叠检查避免覆盖

用法:
    # 基础用法
    python scripts/augment_defects.py \
        --images data/datasets/neu_det/images/train \
        --labels data/datasets/neu_det/labels/train \
        --output data/datasets/neu_det_augmented \
        --num-aug 3

    # 仅增强特定类别
    python scripts/augment_defects.py ... --target-classes scratches,crazing

    # 调整粘贴参数
    python scripts/augment_defects.py ... --paste-count 5 --blend-mode poisson
"""

import argparse
import os
import random
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


# NEU-DET 类别映射
CLASS_NAMES = [
    "crazing",          # 0
    "inclusion",        # 1
    "patches",          # 2
    "pitted_surface",   # 3
    "rolled-in_scale",  # 4
    "scratches",        # 5
]


# ==================== YOLO 标签解析 ====================

def parse_yolo_label(label_path: str) -> list[dict]:
    """
    解析 YOLO 格式标签文件。

    YOLO 格式: class_id cx cy w h (归一化)
    返回: [{"class_id": int, "bbox": [cx, cy, w, h] (归一化)}, ...]
    """
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append({
                    "class_id": int(parts[0]),
                    "bbox": [float(x) for x in parts[1:5]],
                })
    return labels


def yolo_to_pixel(bbox_norm: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """
    YOLO (cx, cy, w, h) 归一化 → 像素坐标 (x1, y1, x2, y2)
    """
    cx, cy, bw, bh = bbox_norm
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))


def pixel_to_yolo(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> list[float]:
    """像素坐标 → YOLO (cx, cy, w, h) 归一化"""
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    return [cx, cy, bw, bh]


# ==================== 缺陷区域提取 ====================

def extract_defects(
    images_dir: str,
    labels_dir: str,
    target_classes: Optional[set[int]] = None,
    min_area_px: int = 100,
) -> list[dict]:
    """
    从标注数据中提取所有缺陷区域。

    返回: [
        {"image": np.ndarray (BGR), "class_id": int, "bbox_pixel": (x1,y1,x2,y2)},
        ...
    ]
    """
    defects = []
    image_files = list(Path(images_dir).glob("*.jpg")) + \
                   list(Path(images_dir).glob("*.png")) + \
                   list(Path(images_dir).glob("*.bmp"))

    print(f"[CopyPaste] 扫描 {len(image_files)} 张标注图像...")

    for img_path in tqdm(image_files, desc="提取缺陷"):
        # 对应标签文件
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        labels = parse_yolo_label(str(label_path))

        if not labels:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        for lbl in labels:
            cls_id = lbl["class_id"]
            if target_classes is not None and cls_id not in target_classes:
                continue

            x1, y1, x2, y2 = yolo_to_pixel(lbl["bbox"], w, h)

            # 跳过过小的缺陷
            area = (x2 - x1) * (y2 - y1)
            if area < min_area_px:
                continue

            # 裁剪缺陷区域
            defect_crop = img[y1:y2, x1:x2].copy()
            if defect_crop.size == 0:
                continue

            defects.append({
                "image": defect_crop,
                "class_id": cls_id,
                "source_bbox": (x1, y1, x2, y2),
                "source_size": (w, h),
                "source_name": img_path.stem,
            })

    print(f"[CopyPaste] 提取 {len(defects)} 个缺陷区域")
    return defects


# ==================== 缺陷粘贴 ====================

def random_transform(defect_img: np.ndarray) -> np.ndarray:
    """
    对缺陷区域随机几何变换。

    - 旋转: -30° ~ +30°
    - 缩放: 0.7× ~ 1.5×
    - 水平翻转: 50% 概率
    """
    dh, dw = defect_img.shape[:2]

    # 随机旋转
    angle = random.uniform(-30, 30)
    center = (dw // 2, dh // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 随机缩放
    scale = random.uniform(0.7, 1.5)
    rot_mat[:, :2] *= scale

    # 计算旋转后尺寸
    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    new_w = int(dh * sin + dw * cos)
    new_h = int(dh * cos + dw * sin)

    # 调整平移量使图像居中
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    transformed = cv2.warpAffine(
        defect_img, rot_mat, (new_w, new_h),
        borderMode=cv2.BORDER_REFLECT,
    )

    # 随机水平翻转
    if random.random() < 0.5:
        transformed = cv2.flip(transformed, 1)

    return transformed


def paste_defect(
    bg_image: np.ndarray,
    defect_img: np.ndarray,
    x: int,
    y: int,
    blend_mode: str = "alpha",
    alpha: float = 0.85,
) -> np.ndarray:
    """
    将缺陷区域粘贴到背景图像上。

    Args:
        bg_image: 背景图像
        defect_img: 缺陷区域
        x, y: 粘贴位置（左上角）
        blend_mode: "alpha" (Alpha混合) | "poisson" (泊松融合) | "direct" (直接覆盖)
        alpha: Alpha 混合系数

    Returns:
        粘贴后的图像
    """
    dh, dw = defect_img.shape[:2]
    bh, bw = bg_image.shape[:2]

    # 裁剪到背景范围内
    if x < 0:
        defect_img = defect_img[:, -x:]
        dw = defect_img.shape[1]
        x = 0
    if y < 0:
        defect_img = defect_img[-y:, :]
        dh = defect_img.shape[0]
        y = 0
    if x + dw > bw:
        defect_img = defect_img[:, :bw - x]
        dw = defect_img.shape[1]
    if y + dh > bh:
        defect_img = defect_img[:bh - y, :]
        dh = defect_img.shape[0]

    if defect_img.size == 0:
        return bg_image

    roi = bg_image[y:y + dh, x:x + dw]

    if blend_mode == "direct":
        roi[:] = defect_img

    elif blend_mode == "alpha":
        # 创建软边缘掩码（中心不透明，边缘渐变透明）
        mask = np.ones((dh, dw), dtype=np.float32)
        margin_w = max(1, dw // 8)
        margin_h = max(1, dh // 8)

        # 水平渐变
        if dw > margin_w * 2:
            mask[:, :margin_w] = np.linspace(0, 1, margin_w)
            mask[:, -margin_w:] = np.linspace(1, 0, margin_w)
        # 垂直渐变
        if dh > margin_h * 2:
            mask[:margin_h, :] *= np.linspace(0, 1, margin_h)[:, np.newaxis]
            mask[-margin_h:, :] *= np.linspace(1, 0, margin_h)[:, np.newaxis]

        mask = np.clip(mask * alpha, 0, 1)
        mask_3ch = np.stack([mask] * 3, axis=2)

        roi[:] = (defect_img * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)

    elif blend_mode == "poisson":
        try:
            center = (x + dw // 2, y + dh // 2)
            # 创建缺陷的灰度掩码
            gray = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            # 腐蚀掩码边缘避免伪影
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
            roi[:] = cv2.seamlessClone(defect_img, roi, mask, center, cv2.NORMAL_CLONE)
        except Exception:
            # 泊松融合失败时回退到 Alpha 混合
            roi[:] = cv2.addWeighted(defect_img, alpha, roi, 1 - alpha, 0)

    bg_image[y:y + dh, x:x + dw] = roi
    return bg_image


def check_overlap(
    new_box: tuple[int, int, int, int],
    existing_boxes: list[tuple[int, int, int, int]],
    max_iou: float = 0.15,
) -> bool:
    """
    检查新粘贴区域是否与已有缺陷重叠过多。

    Returns: True 表示可以粘贴（无严重重叠）
    """
    nx1, ny1, nx2, ny2 = new_box
    for ex1, ey1, ex2, ey2 in existing_boxes:
        inter_x1 = max(nx1, ex1)
        inter_y1 = max(ny1, ey1)
        inter_x2 = min(nx2, ex2)
        inter_y2 = min(ny2, ey2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        new_area = (nx2 - nx1) * (ny2 - ny1)
        exist_area = (ex2 - ex1) * (ey2 - ey1)

        iou = inter_area / (min(new_area, exist_area) + 1e-6)
        if iou > max_iou:
            return False

    return True


# ==================== 主流程 ====================

def augment_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    num_augment: int = 3,
    paste_count: int = 3,
    target_classes: Optional[set[int]] = None,
    blend_mode: str = "alpha",
    max_iou: float = 0.15,
    skip_empty_images: bool = True,
) -> dict:
    """
    执行 Copy-Paste 数据增强。

    Args:
        images_dir: 原始图像目录
        labels_dir: 原始标签目录
        output_dir: 输出目录
        num_augment: 每张图像生成的增强版本数
        paste_count: 每张增强图粘贴的缺陷数
        target_classes: 要增强的类别 ID 集合（None=全部）
        blend_mode: 粘贴融合方式
        max_iou: 最大允许重叠 IoU
        skip_empty_images: 是否跳过无缺陷图像（True=仅增强有缺陷图）

    Returns:
        统计信息字典
    """
    random.seed(42)
    np.random.seed(42)

    # 创建输出目录
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # 第一步：提取所有缺陷区域
    defects = extract_defects(images_dir, labels_dir, target_classes)
    if not defects:
        print("[CopyPaste] 错误: 未找到任何缺陷区域")
        return {"error": "no_defects_found"}

    # 按 class_id 分组
    defects_by_class: dict[int, list[dict]] = {}
    for d in defects:
        defects_by_class.setdefault(d["class_id"], []).append(d)

    # 第二步：获取图像列表
    image_files = list(Path(images_dir).glob("*.jpg")) + \
                   list(Path(images_dir).glob("*.png")) + \
                   list(Path(images_dir).glob("*.bmp"))

    # 第三步：逐图增强
    stats = {
        "total_images": len(image_files),
        "total_augmented": 0,
        "total_pastes": 0,
        "defects_used": len(defects),
        "defects_by_class": {k: len(v) for k, v in defects_by_class.items()},
    }

    print(f"\n[CopyPaste] 开始增强 (模式={blend_mode}, 每图{paste_count}个粘贴)...")

    for img_path in tqdm(image_files, desc="Copy-Paste 增强"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 读取原始标签（用于判断是否已有缺陷）
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        orig_labels = parse_yolo_label(str(label_path))

        if skip_empty_images and not orig_labels:
            continue

        h, w = img.shape[:2]

        # 生成 num_augment 个增强版本
        for aug_idx in range(num_augment):
            aug_img = img.copy()
            new_labels = list(orig_labels)  # 保留原始缺陷
            placed_boxes = []

            # 还原已有缺陷的像素坐标（用于重叠检查）
            for lbl in orig_labels:
                x1, y1, x2, y2 = yolo_to_pixel(lbl["bbox"], w, h)
                placed_boxes.append((x1, y1, x2, y2))

            # 随机粘贴 paste_count 个缺陷
            for _ in range(paste_count * 3):  # 尝试次数 = paste_count × 3
                if len(placed_boxes) - len(orig_labels) >= paste_count:
                    break

                # 随机选择一个缺陷
                cls_id = random.choice(list(defects_by_class.keys()))
                defect = random.choice(defects_by_class[cls_id])

                # 随机变换
                transformed = random_transform(defect["image"])
                td_h, td_w = transformed.shape[:2]

                if td_w <= 5 or td_h <= 5:
                    continue

                # 随机粘贴位置（避免边缘）
                margin = 10
                max_x = max(margin, w - td_w - margin)
                max_y = max(margin, h - td_h - margin)
                px = random.randint(margin, max_x) if max_x > margin else margin
                py = random.randint(margin, max_y) if max_y > margin else margin

                new_box = (px, py, px + td_w, py + td_h)

                # 重叠检查
                if not check_overlap(new_box, placed_boxes, max_iou):
                    continue

                # 执行粘贴
                aug_img = paste_defect(aug_img, transformed, px, py, blend_mode)

                # 记录标签
                yolo_bbox = pixel_to_yolo(px, py, px + td_w, py + td_h, w, h)
                new_labels.append({
                    "class_id": cls_id,
                    "bbox": yolo_bbox,
                })
                placed_boxes.append(new_box)

                stats["total_pastes"] += 1

            # 保存增强图像和标签
            out_name = f"{img_path.stem}_aug{aug_idx}"
            cv2.imwrite(str(out_images / f"{out_name}.jpg"), aug_img)

            # 保存 YOLO 标签
            with open(out_labels / f"{out_name}.txt", "w") as f:
                for lbl in new_labels:
                    bbox = lbl["bbox"]
                    f.write(f"{lbl['class_id']} {bbox[0]:.6f} {bbox[1]:.6f} "
                            f"{bbox[2]:.6f} {bbox[3]:.6f}\n")

            stats["total_augmented"] += 1

    # 第四步：复制原始数据到输出目录（方便直接训练）
    print("[CopyPaste] 复制原始数据...")
    for img_path in tqdm(image_files, desc="复制原始"):
        shutil.copy2(img_path, out_images / img_path.name)
        src_label = Path(labels_dir) / (img_path.stem + ".txt")
        if src_label.exists():
            shutil.copy2(src_label, out_labels / src_label.name)

    # 生成 dataset.yaml
    _generate_dataset_yaml(output_dir, list(defects_by_class.keys()))

    return stats


def _generate_dataset_yaml(output_dir: str, class_ids: list[int]):
    """生成 NEU-DET 格式的 dataset.yaml"""
    class_names = [CLASS_NAMES[i] for i in sorted(class_ids) if i < len(CLASS_NAMES)]
    yaml_content = f"""# Copy-Paste 增强数据集
path: {Path(output_dir).absolute().as_posix()}
train: images
val: images
test: images

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = Path(output_dir) / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[CopyPaste] dataset.yaml 已生成: {yaml_path}")


# ==================== CLI ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy-Paste 缺陷数据增强",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python augment_defects.py -i data/datasets/neu_det/images/train -l data/datasets/neu_det/labels/train -o data/datasets/neu_det_aug
  python augment_defects.py ... --target-classes scratches,crazing --paste-count 8 --num-aug 5
        """,
    )
    parser.add_argument("-i", "--images", required=True, help="原始图像目录")
    parser.add_argument("-l", "--labels", required=True, help="YOLO 标签目录")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--num-aug", type=int, default=3, help="每图增强版本数 (默认 3)")
    parser.add_argument("--paste-count", type=int, default=3, help="每图粘贴缺陷数 (默认 3)")
    parser.add_argument("--target-classes", default=None,
                        help="逗号分隔的目标类别，如 'scratches,crazing'")
    parser.add_argument("--blend-mode", choices=["alpha", "poisson", "direct"],
                        default="alpha", help="粘贴融合方式 (默认 alpha)")
    parser.add_argument("--max-iou", type=float, default=0.15, help="最大允许重叠 IoU")
    parser.add_argument("--all-images", action="store_true",
                        help="对所有图像增强（含无缺陷图），默认仅对有缺陷图增强")
    return parser.parse_args()


def main():
    args = parse_args()

    # 解析目标类别
    target_classes = None
    if args.target_classes:
        class_name_to_id = {n: i for i, n in enumerate(CLASS_NAMES)}
        target_classes = set()
        for name in args.target_classes.split(","):
            name = name.strip()
            if name in class_name_to_id:
                target_classes.add(class_name_to_id[name])
            else:
                print(f"[CopyPaste] 警告: 未知类别 '{name}'")
        print(f"[CopyPaste] 目标类别: {args.target_classes} → IDs {target_classes}")

    t0 = time.time()
    stats = augment_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        num_augment=args.num_aug,
        paste_count=args.paste_count,
        target_classes=target_classes,
        blend_mode=args.blend_mode,
        max_iou=args.max_iou,
        skip_empty_images=not args.all_images,
    )

    elapsed = time.time() - t0
    print(f"\n[CopyPaste] 完成! 耗时 {elapsed:.1f}s")
    print(f"  增强图像: {stats.get('total_augmented', 0)} 张")
    print(f"  粘贴缺陷: {stats.get('total_pastes', 0)} 次")
    print(f"  输出目录: {args.output}")


if __name__ == "__main__":
    main()
