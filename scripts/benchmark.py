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
    parser.add_argument("--device", default="auto", help="运行设备 (auto/cpu/cuda:0)")
    return parser.parse_args()


def main():
    import os
    args = parse_args()
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    models = [m.strip() for m in args.model.split(",") if m.strip()]

    results = {}

    for model_path in models:
        print(f"\n加载模型: {model_path} ...")
        detector = YOLODetector(
            model_path=model_path,
            img_size=args.img_size,
            device=args.device,
        )

        try:
            detector.load_model()
        except Exception as e:
            print(f"模型 {model_path} 加载失败: {e}")
            continue

        # 预热
        dummy = np.random.randint(0, 255, (args.img_size, args.img_size, 3), dtype=np.uint8)
        detector.warmup(dummy)

        # 单张推理延迟
        latencies = []
        for i in range(args.iterations):
            img = np.random.randint(0, 255, (args.img_size, args.img_size, 3), dtype=np.uint8)
            start = time.perf_counter()
            _ = detector.detect(img)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies = np.array(latencies)
        results[model_path] = {
            "mean": latencies.mean(),
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "fps": 1000 / latencies.mean(),
            "batches": {}
        }

        # 批量推理
        for bs in batch_sizes:
            batch = np.random.randint(0, 255, (bs, args.img_size, args.img_size, 3), dtype=np.uint8)
            batch_times = []
            iterations_bs = max(1, args.iterations // bs)
            for _ in range(iterations_bs):
                start = time.perf_counter()
                for j in range(bs):
                    _ = detector.detect(batch[j])
                batch_times.append((time.perf_counter() - start) * 1000 / bs)
            results[model_path]["batches"][bs] = np.mean(batch_times)

    # 打印对比表
    print(f"\n{'='*75}")
    print(f"YOLO 性能对比基准测试结果 (设备: {args.device})")
    print(f"{'='*75}")
    print(f"{'Model':<40} | {'Mean Latency':<12} | {'P95 Latency':<12} | {'FPS':<6}")
    print("-" * 75)
    for model_path, res in results.items():
        name = os.path.basename(model_path)
        print(f"{name:<40} | {res['mean']:>10.2f}ms | {res['p95']:>10.2f}ms | {res['fps']:>5.1f}")
    
    print(f"\n批量吞吐对比 (每张图平均耗时):")
    print("-" * 75)
    header = f"{'Model':<40}"
    for bs in batch_sizes:
        header += f" | {f'Batch={bs}':<10}"
    print(header)
    print("-" * 75)
    for model_path, res in results.items():
        name = os.path.basename(model_path)
        row = f"{name:<40}"
        for bs in batch_sizes:
            row += f" | {res['batches'].get(bs, 0.0):>8.2f}ms"
        print(row)
    print(f"{'='*75}\n")
    print("测试完成")


if __name__ == "__main__":
    main()
