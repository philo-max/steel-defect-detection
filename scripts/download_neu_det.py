"""
下载 NEU-DET 钢铁表面缺陷数据集并转换为 YOLO 格式。

NEU-DET 数据集:
- 6 类缺陷: crazing(裂纹), inclusion(夹杂), patches(斑块), 
             pitted_surface(麻点), rolled-in_scale(轧制氧化皮), scratches(划痕)
- 1800 张灰度图 (200×200)
- Pascal VOC XML 标注格式
"""

import argparse
import os
import random
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# NEU-DET 类别映射 (英文 -> 中文)
CLASS_NAMES = [
    "crazing",        # 裂纹
    "inclusion",      # 夹杂
    "patches",        # 斑块
    "pitted_surface", # 麻点
    "rolled-in_scale",# 轧制氧化皮
    "scratches",      # 划痕
]

CLASS_CN = {
    "crazing": "裂纹",
    "inclusion": "夹杂",
    "patches": "斑块",
    "pitted_surface": "麻点",
    "rolled-in_scale": "轧制氧化皮",
    "scratches": "划痕",
}

# NEU-DET 镜像下载地址 (多个备用源)
NEU_DET_URLS = [
    # Kaggle 直链 (需要 cookie，通常不可用)
    # "https://www.kaggle.com/api/v1/datasets/kaustubhdikshit/neu-surface-defect-database/download",
    # 各大学/研究机构镜像
    "http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html",
    # GitHub Release 镜像
    "https://github.com/Charmve/Surface-Defect-Detection/releases/download/v1.0/NEU-DET.zip",
    "https://github.com/kaustubhdikshit/NEU-DET/archive/refs/heads/main.zip",
    "https://codeload.github.com/kaustubhdikshit/NEU-DET/zip/refs/heads/main",
]

# 备用: 如果网络不通，生成合成数据集
SYNTHETIC_FALLBACK = True

# 合成数据生成参数
SYNTHETIC_IMG_SIZE = 320  # 更大的图像尺寸
SYNTHETIC_PER_CLASS = 200  # 更多样本


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """下载文件，带进度显示"""
    print(f"  下载中: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                        sys.stdout.write(f"\r  [{bar}] {pct}% ({downloaded//1024}/{total//1024} KB)")
                        sys.stdout.flush()
            if total > 0:
                print()
        return True
    except Exception as e:
        print(f"\n  下载失败: {e}")
        return False


def xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """将 Pascal VOC XML 标注转为 YOLO 格式行"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_NAMES:
            continue
        cls_id = CLASS_NAMES.index(name)
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        # 转为 YOLO 归一化格式: class_id cx cy w h
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def convert_dataset(src_dir: Path, dst_dir: Path, split: tuple = (0.7, 0.2, 0.1)):
    """转换 NEU-DET 数据集为 YOLO 格式并划分"""
    # 收集所有图像-标注对
    image_dir = src_dir / "images"
    anno_dir = src_dir / "annotations"

    pairs = []
    for img_path in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.bmp")):
        xml_path = anno_dir / f"{img_path.stem}.xml"
        if xml_path.exists():
            pairs.append((img_path, xml_path))

    if not pairs:
        print("  未找到图像-标注对，检查目录结构...")
        # 尝试递归搜索
        for img_path in sorted(src_dir.rglob("*.jpg")) + sorted(src_dir.rglob("*.bmp")):
            xml_path = img_path.with_suffix(".xml")
            if xml_path.exists():
                pairs.append((img_path, xml_path))
            else:
                # 也许 XML 在 annotations 子目录
                alt_xml = img_path.parent.parent / "annotations" / f"{img_path.stem}.xml"
                if alt_xml.exists():
                    pairs.append((img_path, alt_xml))

    if not pairs:
        print("  仍然未找到，将使用合成数据集")
        return 0

    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * split[0])
    n_val = int(n * split[1])

    subsets = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    for subset_name, subset_pairs in subsets.items():
        img_out = dst_dir / "images" / subset_name
        lbl_out = dst_dir / "labels" / subset_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, xml_path in subset_pairs:
            # 复制图像
            shutil.copy2(img_path, img_out / img_path.name)
            # 转换标注
            yolo_lines = xml_to_yolo(xml_path, 200, 200)
            label_path = lbl_out / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))

        print(f"  {subset_name}: {len(subset_pairs)} 张")

    return n


def _make_steel_background(h: int, w: int):
    """生成逼真的钢板表面背景 (灰度)"""
    import cv2
    import numpy as np

    # 基础灰度 (中灰偏亮)
    base = random.randint(155, 195)
    img = np.full((h, w), base, dtype=np.float32)

    # 水平轧制纹理 (rolling marks)
    for _ in range(random.randint(20, 50)):
        y = random.randint(0, h - 1)
        intensity = random.uniform(-12, 12)
        thickness = random.randint(1, 4)
        cv2.line(img.astype(np.uint8), (0, y), (w, y),
                 int(base + intensity), thickness)

    img = img.astype(np.float32)

    # 高斯噪声
    noise = np.random.normal(0, random.uniform(3, 10), (h, w))
    img += noise

    # 低频光照变化
    x = np.linspace(0, 1, w)
    y_vals = np.linspace(0, 1, h)
    gradient = np.outer(y_vals * random.uniform(-15, 15), np.ones(w))
    gradient += np.outer(np.ones(h), x * random.uniform(-10, 10))
    img += gradient

    # Perlin-like 纹理 (简化版)
    freq = random.randint(3, 8)
    tex_x = np.sin(np.linspace(0, freq * np.pi, w)) * random.uniform(3, 8)
    tex_y = np.sin(np.linspace(0, freq * np.pi, h)) * random.uniform(3, 8)
    img += np.outer(tex_y, np.ones(w)) * 0.3
    img += np.outer(np.ones(h), tex_x) * 0.3

    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_crazing(region):
    """绘制网状裂纹 - 不规则细线网络"""
    import cv2
    import numpy as np
    h, w = region.shape[:2]
    n_lines = random.randint(4, 12)
    pts = [(random.randint(0, w - 1), random.randint(0, h - 1))
           for _ in range(n_lines)]
    # 连接邻近点形成网状
    for i, p1 in enumerate(pts):
        for j, p2 in enumerate(pts):
            if i < j and random.random() < 0.3:
                thickness = random.randint(1, 2)
                intensity = random.randint(10, 50)
                cv2.line(region, p1, p2, intensity, thickness)
    # 添加分支
    for _ in range(random.randint(2, 6)):
        x = random.randint(5, w - 5)
        y = random.randint(5, h - 5)
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(10, min(w, h) // 2)
        x2 = int(x + np.cos(angle) * length)
        y2 = int(y + np.sin(angle) * length)
        cv2.line(region, (x, y), (x2, y2),
                 random.randint(20, 60), random.randint(1, 2))


def _draw_scratches(region):
    """绘制划痕 - 直线或微弯的长痕"""
    import cv2
    import numpy as np
    h, w = region.shape[:2]
    n_scratches = random.randint(1, 3)
    for _ in range(n_scratches):
        thickness = random.randint(2, 6)
        if random.random() < 0.5:
            # 水平划过
            y = random.randint(thickness, h - thickness)
            offset = random.randint(-8, 8)
            pts = np.array([[0, y], [w // 3, y + offset],
                           [2 * w // 3, y - offset], [w, y + offset // 2]])
            cv2.polylines(region, [pts], False,
                         random.randint(15, 60), thickness)
        else:
            # 斜向划过
            cv2.line(region, (0, random.randint(0, h // 3)),
                     (w, random.randint(2 * h // 3, h)),
                     random.randint(15, 60), thickness)


def _draw_patches(region):
    """绘制斑块 - 不规则暗色区域"""
    import cv2
    import numpy as np
    h, w = region.shape[:2]
    cx, cy = w // 2 + random.randint(-15, 15), h // 2 + random.randint(-15, 15)
    rx, ry = random.randint(10, w // 3), random.randint(10, h // 3)
    n_pts = random.randint(6, 15)
    angles = sorted([random.uniform(0, 2 * np.pi) for _ in range(n_pts)])
    pts = np.array([[
        int(cx + rx * (0.6 + 0.4 * random.random()) * np.cos(a)),
        int(cy + ry * (0.6 + 0.4 * random.random()) * np.sin(a))
    ] for a in angles])
    intensity = random.randint(20, 55)
    cv2.fillPoly(region, [pts], intensity)
    # 边缘羽化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    region[:] = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)


def _draw_pitted_surface(region):
    """绘制麻点 - 密集小暗点"""
    import cv2
    import numpy as np
    h, w = region.shape[:2]
    n_dots = random.randint(20, 60)
    for _ in range(n_dots):
        px = random.randint(3, w - 3)
        py = random.randint(3, h - 3)
        radius = random.randint(1, 5)
        intensity = random.randint(20, 80)
        cv2.circle(region, (px, py), radius, intensity, -1)


def _draw_rolled_in_scale(region):
    """绘制轧制氧化皮 - 不规则片状暗区"""
    import cv2
    import numpy as np
    h, w = region.shape[:2]
    n_pieces = random.randint(1, 4)
    for _ in range(n_pieces):
        cx = random.randint(5, w - 5)
        cy = random.randint(5, h - 5)
        sz_x = random.randint(8, w // 2)
        sz_y = random.randint(5, h // 2)
        n_pts = random.randint(5, 10)
        pts = np.array([[
            cx + int(sz_x * (random.random() - 0.5)),
            cy + int(sz_y * (random.random() - 0.5))
        ] for _ in range(n_pts)])
        hull = cv2.convexHull(pts)
        intensity = random.randint(25, 65)
        cv2.fillPoly(region, [hull], intensity)


def _draw_inclusion(region):
    """绘制夹杂物 - 明亮不规则斑点"""
    import cv2
    import numpy as np
    h, w = region.shape[:2]
    n_inclusions = random.randint(1, 4)
    for _ in range(n_inclusions):
        cx = random.randint(8, w - 8)
        cy = random.randint(8, h - 8)
        rx = random.randint(3, 12)
        ry = random.randint(3, 12)
        angle = random.uniform(0, 180)
        intensity = random.randint(200, 250)
        cv2.ellipse(region, (cx, cy), (rx, ry), angle, 0, 360, intensity, -1)


_DRAW_FN = {
    "crazing": _draw_crazing,
    "inclusion": _draw_inclusion,
    "patches": _draw_patches,
    "pitted_surface": _draw_pitted_surface,
    "rolled-in_scale": _draw_rolled_in_scale,
    "scratches": _draw_scratches,
}


def generate_synthetic_dataset(dst_dir: Path, num_per_class: int = 80):
    """生成逼真的合成钢铁缺陷数据集"""
    import cv2
    import numpy as np

    SZ = SYNTHETIC_IMG_SIZE
    print(f"  生成逼真合成数据集 ({SZ}×{SZ}, 每类 {num_per_class} 张, "
          f"共 {num_per_class * 6} 张)...")

    for subset in ["train", "val", "test"]:
        (dst_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (dst_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

    total = 0
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        draw_fn = _DRAW_FN.get(cls_name)
        for i in range(num_per_class):
            # 钢板背景
            img = _make_steel_background(SZ, SZ)

            # 缺陷区域 (随机位置、大小)
            w = random.randint(SZ // 8, SZ // 3)
            h = random.randint(SZ // 8, SZ // 3)
            margin = SZ // 16
            x1 = random.randint(margin, SZ - w - margin)
            y1 = random.randint(margin, SZ - h - margin)
            x2 = x1 + w
            y2 = y1 + h

            # 提取 ROI 并绘制缺陷
            defect_roi = img[y1:y2, x1:x2].copy()
            draw_fn(defect_roi)
            img[y1:y2, x1:x2] = defect_roi

            # 转 BGR 三通道 (灰度复制到三个通道)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # 分配 train/val/test
            r = random.random()
            if r < 0.7:
                subset = "train"
            elif r < 0.9:
                subset = "val"
            else:
                subset = "test"

            img_name = f"{cls_name}_{i:04d}.jpg"
            cv2.imwrite(str(dst_dir / "images" / subset / img_name), img_bgr)

            # YOLO 标注
            cx = ((x1 + x2) / 2) / SZ
            cy = ((y1 + y2) / 2) / SZ
            bw = (x2 - x1) / SZ
            bh = (y2 - y1) / SZ
            label = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            label_path = dst_dir / "labels" / subset / f"{cls_name}_{i:04d}.txt"
            label_path.write_text(label)
            total += 1

    print(f"  合成数据集: {total} 张 (train/val/test 按 7:2:1 分配)")
    return total


def write_dataset_yaml(dst_dir: Path, nc: int, names: list[str]):
    """生成 YOLO dataset.yaml"""
    content = (
        f"# NEU-DET 钢铁表面缺陷数据集\n"
        f"path: {dst_dir.absolute().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: {nc}\n"
        f"names:\n"
    )
    for name in names:
        content += f"  - {name}\n"

    yaml_path = dst_dir / "dataset.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    print(f"  dataset.yaml -> {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="下载/准备 NEU-DET 数据集")
    parser.add_argument("--output", default="data/datasets/neu_det", help="输出目录")
    parser.add_argument("--download-dir", default="data/tmp", help="下载缓存目录")
    parser.add_argument("--synthetic-only", action="store_true", help="直接生成合成数据，跳过下载")
    parser.add_argument("--synthetic-count", type=int, default=80,
                        help="合成数据每类数量 (默认80, 共480张)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = False

    # 尝试下载 NEU-DET
    if not args.synthetic_only:
        tmp_dir = Path(args.download_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        zip_path = tmp_dir / "neu_det.zip"

        print("[1/3] 下载 NEU-DET 数据集...")
        for url in NEU_DET_URLS:
            if download_file(url, zip_path):
                break
        else:
            print("  所有下载源均失败，切换到合成数据模式")

        if zip_path.exists() and zip_path.stat().st_size > 1000:
            print("[2/3] 解压并转换数据集...")
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmp_dir)
                # 查找提取后的根目录
                extracted_dirs = [d for d in tmp_dir.iterdir() if d.is_dir() and d.name != zip_path.name]
                if extracted_dirs:
                    src = extracted_dirs[0]
                    n = convert_dataset(src, output_dir)
                    if n > 0:
                        success = True
                        print(f"  ✓ 成功转换 {n} 张图像")
            except zipfile.BadZipFile:
                print("  ZIP 文件损坏，切换到合成数据模式")

        # 清理临时文件
        if zip_path.exists():
            zip_path.unlink()

    # 后备: 合成数据集
    if not success and SYNTHETIC_FALLBACK:
        print(f"[{'2' if args.synthetic_only else '3'}/3] 生成合成钢铁缺陷数据集...")
        n = generate_synthetic_dataset(output_dir, args.synthetic_count)
        if n > 0:
            success = True

    if not success:
        print("✗ 数据集准备失败")
        sys.exit(1)

    print("[最后] 生成 dataset.yaml...")
    write_dataset_yaml(output_dir, len(CLASS_NAMES), CLASS_NAMES)

    print(f"\n✓ 数据集准备完成 -> {output_dir}")
    print(f"  类别数: {len(CLASS_NAMES)}")
    for name in CLASS_NAMES:
        print(f"    - {name} ({CLASS_CN.get(name, '?')})")


if __name__ == "__main__":
    main()
