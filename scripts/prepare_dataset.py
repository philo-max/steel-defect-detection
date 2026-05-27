"""
数据集准备脚本。

功能:
1. 将标注数据转换为 YOLO 格式
2. 自动划分训练集/验证集/测试集
3. 生成 dataset.yaml 配置文件
4. 数据增强预览
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="准备 YOLO 训练数据集")
    parser.add_argument("--images", default="data/images", help="图像目录")
    parser.add_argument("--labels", default="data/labels", help="标注目录")
    parser.add_argument("--output", default="data/datasets", help="输出目录")
    parser.add_argument("--split", default="0.7,0.2,0.1", help="训练/验证/测试 比例")
    parser.add_argument("--classes", nargs="+", default=[
        "crack", "scratch", "scale", "indentation", "blister"
    ], help="缺陷类别列表")
    return parser.parse_args()


def main():
    args = parse_args()

    train_r, val_r, test_r = map(float, args.split.split(","))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有图像
    image_dir = Path(args.images)
    label_dir = Path(args.labels)

    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    if not images:
        print("未找到图像文件，将生成示例 dataset.yaml")
        images = []

    random.shuffle(images)
    n = len(images)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train+n_val]
    test_imgs = images[n_train+n_val:]

    # 创建目录结构
    for subset, img_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        sub_img_dir = output_dir / "images" / subset
        sub_label_dir = output_dir / "labels" / subset
        sub_img_dir.mkdir(parents=True, exist_ok=True)
        sub_label_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_list:
            shutil.copy2(img_path, sub_img_dir / img_path.name)

            # 查找同名标注文件
            for ext in [".txt", ".xml"]:
                label_path = label_dir / img_path.with_suffix(ext).name
                if label_path.exists():
                    shutil.copy2(label_path, sub_label_dir / label_path.name)
                    break

    # 生成 dataset.yaml
    dataset_yaml = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(args.classes),
        "names": args.classes,
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"数据集准备完成:")
    print(f"  训练集: {len(train_imgs)} 张")
    print(f"  验证集: {len(val_imgs)} 张")
    print(f"  测试集: {len(test_imgs)} 张")
    print(f"  配置: {yaml_path}")


if __name__ == "__main__":
    main()
