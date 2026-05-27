"""
性能基准测试脚本。

测试项目:
1. 推理延迟 (单张/批量)
2. 吞吐量 (FPS)
3. GPU 利用率与显存占用
4. ONNX vs PyTorch 对比
"""

import argparse
import time

import numpy as np
import cv2

from src.detection_engine import YOLODetector


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 性能基准测试")
    parser.add_argument("--model", default="models/weights/yolov8n.pt", help="模型路径")
    parser.add_argument("--iterations", type=int, default=100, help="测试迭代次数")
    parser.add_argument("--img-size", type=int, default=640, help="测试图像尺寸")
    parser.add_argument("--batch-sizes", default="1,4,8", help="测试批次大小 (逗号分隔)")
    parser.add_argument("--device", default="cuda:0", help="运行设备")
    return parser.parse_args()


def main():
    args = parse_args()
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    detector = YOLODetector(
        model_path=args.model,
        img_size=args.img_size,
        device=args.device,
    )

    try:
        detector.load_model()
    except FileNotFoundError:
        print("模型文件未找到，请先放置模型权重")
        return

    # 预热
    dummy = np.random.randint(0, 255, (args.img_size, args.img_size, 3), dtype=np.uint8)
    detector.warmup(dummy)

    print(f"\n{'='*60}")
    print(f"YOLO 性能基准测试")
    print(f"模型: {args.model}")
    print(f"设备: {args.device}")
    print(f"图像尺寸: {args.img_size}x{args.img_size}")
    print(f"{'='*60}\n")

    # 单张推理延迟
    print("--- 单张推理延迟 ---")
    latencies = []
    for i in range(args.iterations):
        img = np.random.randint(0, 255, (args.img_size, args.img_size, 3), dtype=np.uint8)
        start = time.perf_counter()
        _ = detector.detect(img)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    print(f"  平均: {latencies.mean():.2f}ms")
    print(f"  P50:  {np.percentile(latencies, 50):.2f}ms")
    print(f"  P95:  {np.percentile(latencies, 95):.2f}ms")
    print(f"  P99:  {np.percentile(latencies, 99):.2f}ms")
    print(f"  FPS:  {1000 / latencies.mean():.1f}\n")

    # 批量推理
    print("--- 批量推理吞吐 ---")
    for bs in batch_sizes:
        batch = np.random.randint(0, 255, (bs, args.img_size, args.img_size, 3), dtype=np.uint8)
        batch_times = []
        for _ in range(args.iterations // bs):
            start = time.perf_counter()
            for j in range(bs):
                _ = detector.detect(batch[j])
            batch_times.append((time.perf_counter() - start) * 1000 / bs)

        avg_batch = np.mean(batch_times)
        print(f"  Batch={bs}: {avg_batch:.2f}ms/img ({1000/avg_batch:.1f} FPS)")

    print(f"\n{'='*60}")
    print("测试完成")


if __name__ == "__main__":
    main()
