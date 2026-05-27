"""
日志分析脚本 - 分析检测系统运行日志，提取关键指标和异常。

功能:
1. 检测延迟分布统计
2. 缺陷检出率趋势
3. VLM API 调用异常统计
4. 系统资源使用分析
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="分析检测系统日志")
    parser.add_argument("--log", default="logs/app.log", help="日志文件路径")
    parser.add_argument("--days", type=int, default=7, help="分析最近 N 天")
    return parser.parse_args()


def analyze_log(log_path: str, days: int = 7):
    """分析日志文件中的关键指标"""
    path = Path(log_path)
    if not path.exists():
        print(f"日志文件不存在: {log_path}")
        return

    latencies = []
    defect_counts = []
    errors = defaultdict(int)
    total_lines = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            total_lines += 1

            # 提取推理延迟
            match = re.search(r"inference_time_ms[:\s]+([0-9.]+)", line)
            if match:
                latencies.append(float(match.group(1)))

            # 提取缺陷数量
            match = re.search(r"defect_count[:\s]+(\d+)", line)
            if match:
                defect_counts.append(int(match.group(1)))

            # 提取错误
            if "ERROR" in line or "error" in line.lower():
                # 分类错误类型
                if "VLM" in line or "vlm" in line:
                    errors["VLM_API"] += 1
                elif "camera" in line.lower():
                    errors["Camera"] += 1
                elif "model" in line.lower():
                    errors["Model"] += 1
                else:
                    errors["Other"] += 1

    print(f"\n{'='*50}")
    print(f"日志分析报告: {log_path}")
    print(f"总行数: {total_lines}")
    print(f"{'='*50}\n")

    if latencies:
        import statistics
        print("--- 推理延迟 (ms) ---")
        print(f"  平均: {statistics.mean(latencies):.2f}")
        print(f"  中位数: {statistics.median(latencies):.2f}")
        print(f"  最小: {min(latencies):.2f}")
        print(f"  最大: {max(latencies):.2f}")
        print(f"  采样数: {len(latencies)}")
    else:
        print("未找到推理延迟数据")

    if defect_counts:
        print(f"\n--- 缺陷统计 ---")
        print(f"  总检出次数: {sum(defect_counts)}")
        print(f"  平均每次检测缺陷数: {sum(defect_counts)/len(defect_counts):.1f}")
        print(f"  检出率 (有缺陷的比例): {sum(1 for c in defect_counts if c > 0)/len(defect_counts)*100:.1f}%")

    if errors:
        print(f"\n--- 错误统计 ---")
        for error_type, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count} 次")

    print(f"\n{'='*50}")


if __name__ == "__main__":
    args = parse_args()
    analyze_log(args.log, args.days)
