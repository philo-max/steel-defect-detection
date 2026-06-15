"""
批量图像检测处理器 - 支持文件夹批量检测和报表导出

功能:
1. 批量图像检测 (YOLO + VLM)
2. 实时进度显示
3. 结果导出 (CSV/Excel/HTML)
4. 缺陷统计汇总
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .base_detector import InferenceResult


@dataclass
class BatchResult:
    """单张图像的批量检测结果"""
    image_path: str
    image_name: str
    success: bool
    yolo_detections: list = field(default_factory=list)
    vlm_detections: list = field(default_factory=list)
    defect_count: int = 0
    defect_types: str = ""
    max_confidence: float = 0.0
    inference_time_ms: float = 0.0
    error_msg: str = ""
    timestamp: str = ""


class BatchProcessor:
    """批量检测处理器"""

    # 支持的图像格式
    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    def __init__(
        self,
        yolo_detector,
        vlm_detector=None,
        output_dir: str = "data/batch_results",
        save_annotated: bool = True,
        save_defect_only: bool = False,
    ):
        self.yolo = yolo_detector
        self.vlm = vlm_detector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_annotated = save_annotated
        self.save_defect_only = save_defect_only

    def process_folder(
        self,
        folder_path: str,
        progress_callback: Optional[Callable] = None,
        use_vlm: bool = False,
        vlm_threshold: float = 0.5,
    ) -> list[BatchResult]:
        """
        批量处理文件夹中的所有图像

        Args:
            folder_path: 图像文件夹路径
            progress_callback: 进度回调函数 (current, total, result)
            use_vlm: 是否对低置信度检测结果使用VLM复核
            vlm_threshold: VLM介入的置信度阈值

        Returns:
            list[BatchResult]: 所有图像的检测结果
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")

        # 收集所有图像文件
        image_files = []
        for ext in self.SUPPORTED_FORMATS:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))

        image_files = sorted(list(set(image_files)))
        total = len(image_files)

        if total == 0:
            print(f"[WARN] 文件夹中没有支持的图像文件: {folder_path}")
            return []

        print(f"[INFO] 发现 {total} 张图像，开始批量检测...")

        results = []
        annotated_dir = self.output_dir / "annotated"
        if self.save_annotated:
            annotated_dir.mkdir(exist_ok=True)

        for i, img_path in enumerate(image_files):
            result = self._process_single_image(
                img_path, use_vlm=use_vlm, vlm_threshold=vlm_threshold
            )
            results.append(result)

            # 保存标注图像
            if self.save_annotated and result.success:
                self._save_annotated_image(result, img_path, annotated_dir)

            # 进度回调
            if progress_callback:
                progress_callback(i + 1, total, result)

            # 控制台进度
            status = "✓" if result.success else "✗"
            defect_info = f"[{result.defect_count} defects]" if result.defect_count > 0 else "[OK]"
            print(f"  [{i+1}/{total}] {status} {img_path.name} {defect_info}")

        # 导出汇总报告
        self._export_results(results)

        return results

    def _process_single_image(
        self,
        img_path: Path,
        use_vlm: bool = False,
        vlm_threshold: float = 0.5,
    ) -> BatchResult:
        """处理单张图像"""
        result = BatchResult(
            image_path=str(img_path),
            image_name=img_path.name,
            timestamp=datetime.now().isoformat(),
        )

        start_time = time.perf_counter()

        try:
            # 读取图像
            image = cv2.imread(str(img_path))
            if image is None:
                result.error_msg = "无法读取图像"
                return result

            # YOLO检测
            yolo_result = self.yolo.detect(image)
            result.yolo_detections = [
                {
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox": d.bbox,
                }
                for d in yolo_result.detections
            ]
            result.defect_count = len(yolo_result.detections)

            # VLM复核 (对低置信度或没有检测到的情况)
            if use_vlm and self.vlm is not None:
                need_vlm = (
                    len(yolo_result.detections) == 0 or
                    any(d.confidence < vlm_threshold for d in yolo_result.detections)
                )
                if need_vlm:
                    vlm_result = self.vlm.detect(image)
                    if not vlm_result.error:
                        result.vlm_detections = [
                            {
                                "class_name": d.class_name,
                                "confidence": round(d.confidence, 4),
                            }
                            for d in vlm_result.detections
                        ]
                        # 如果VLM检测到更多缺陷，更新计数
                        if len(vlm_result.detections) > result.defect_count:
                            result.defect_count = len(vlm_result.detections)

            # 统计信息
            all_confs = [d["confidence"] for d in result.yolo_detections]
            if result.vlm_detections:
                all_confs.extend([d["confidence"] for d in result.vlm_detections])
            result.max_confidence = max(all_confs) if all_confs else 0.0

            # 缺陷类型汇总
            types = set()
            for d in result.yolo_detections:
                types.add(d["class_name"])
            for d in result.vlm_detections:
                types.add(d["class_name"])
            result.defect_types = ",".join(sorted(types))

            result.success = True
            result.inference_time_ms = (time.perf_counter() - start_time) * 1000

        except Exception as e:
            result.error_msg = str(e)
            result.success = False

        return result

    def _save_annotated_image(
        self,
        result: BatchResult,
        img_path: Path,
        annotated_dir: Path,
    ) -> None:
        """保存标注后的图像"""
        if result.defect_count == 0 and self.save_defect_only:
            return

        try:
            image = cv2.imread(str(img_path))
            if image is None:
                return

            # 绘制检测框
            h, w = image.shape[:2]
            for det in result.yolo_detections:
                bbox = det["bbox"]
                x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
                conf = det["confidence"]
                color = (0, 0, 255) if conf > 0.7 else (0, 165, 255) if conf > 0.4 else (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class_name']} {conf:.2f}"
                cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 添加状态栏
            status = f"DEFECTS: {result.defect_count}" if result.defect_count > 0 else "PASS"
            bar_color = (0, 0, 200) if result.defect_count > 0 else (0, 140, 0)
            cv2.rectangle(image, (0, 0), (w, 30), (20, 20, 20), -1)
            cv2.putText(image, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)

            output_path = annotated_dir / f"annotated_{img_path.name}"
            cv2.imwrite(str(output_path), image)

        except Exception as e:
            print(f"[WARN] 保存标注图像失败 {img_path.name}: {e}")

    def _export_results(self, results: list[BatchResult]) -> dict:
        """导出检测结果到多种格式"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建DataFrame
        data = []
        for r in results:
            data.append({
                "图像名称": r.image_name,
                "图像路径": r.image_path,
                "检测状态": "成功" if r.success else f"失败: {r.error_msg}",
                "缺陷数量": r.defect_count,
                "缺陷类型": r.defect_types,
                "最高置信度": round(r.max_confidence, 4),
                "YOLO检测数": len(r.yolo_detections),
                "VLM检测数": len(r.vlm_detections),
                "推理时间(ms)": round(r.inference_time_ms, 2),
                "检测时间": r.timestamp,
            })

        df = pd.DataFrame(data)

        # 导出CSV
        csv_path = self.output_dir / f"batch_report_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 导出Excel (带格式)
        excel_path = self.output_dir / f"batch_report_{timestamp}.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='检测结果', index=False)

                # 添加统计汇总表
                stats = self._generate_statistics(results)
                stats_df = pd.DataFrame([stats])
                stats_df.to_excel(writer, sheet_name='统计汇总', index=False)
        except Exception as e:
            print(f"[WARN] Excel导出失败 (可能缺少openpyxl): {e}")
            excel_path = None

        # 导出HTML报告
        html_path = self.output_dir / f"batch_report_{timestamp}.html"
        html_content = self._generate_html_report(results, df)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # 导出JSON (详细结果)
        json_path = self.output_dir / f"batch_report_{timestamp}.json"
        json_data = {
            "summary": self._generate_statistics(results),
            "results": [
                {
                    "image_name": r.image_name,
                    "success": r.success,
                    "defect_count": r.defect_count,
                    "defect_types": r.defect_types,
                    "yolo_detections": r.yolo_detections,
                    "vlm_detections": r.vlm_detections,
                    "inference_time_ms": r.inference_time_ms,
                }
                for r in results
            ]
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"\n[INFO] 报告已导出:")
        print(f"  CSV:  {csv_path}")
        if excel_path:
            print(f"  Excel: {excel_path}")
        print(f"  HTML: {html_path}")
        print(f"  JSON: {json_path}")

        return {
            "csv": str(csv_path),
            "excel": str(excel_path) if excel_path else None,
            "html": str(html_path),
            "json": str(json_path),
        }

    def _generate_statistics(self, results: list[BatchResult]) -> dict:
        """生成统计汇总"""
        total = len(results)
        success = sum(1 for r in results if r.success)
        failed = total - success
        with_defects = sum(1 for r in results if r.defect_count > 0)
        clean = total - with_defects

        # 缺陷类型统计
        type_counts = {}
        for r in results:
            if r.defect_types:
                for t in r.defect_types.split(","):
                    type_counts[t] = type_counts.get(t, 0) + 1

        total_defects = sum(r.defect_count for r in results)
        avg_time = sum(r.inference_time_ms for r in results if r.success) / max(success, 1)

        return {
            "总图像数": total,
            "检测成功": success,
            "检测失败": failed,
            "有缺陷": with_defects,
            "无缺陷": clean,
            "总缺陷数": total_defects,
            "平均每张缺陷数": round(total_defects / max(total, 1), 2),
            "平均推理时间(ms)": round(avg_time, 2),
            "缺陷类型分布": type_counts,
        }

    def _generate_html_report(self, results: list[BatchResult], df: pd.DataFrame) -> str:
        """生成HTML报告"""
        stats = self._generate_statistics(results)

        # 缺陷类型分布图表数据
        type_dist = stats.get("缺陷类型分布", {})
        type_labels = list(type_dist.keys())
        type_values = list(type_dist.values())

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>钢铁表面缺陷检测 - 批量检测报告</title>
    <style>
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a237e; margin-bottom: 10px; }}
        .subtitle {{ color: #666; margin-bottom: 30px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 30px; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-value {{ font-size: 36px; font-weight: bold; margin: 10px 0; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
        .stat-card.success {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .stat-card.warning {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .stat-card.info {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #1a237e; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #e0e0e0; }}
        tr:hover {{ background: #f5f5f5; }}
        .defect-badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .defect-yes {{ background: #ffebee; color: #c62828; }}
        .defect-no {{ background: #e8f5e9; color: #2e7d32; }}
        .timestamp {{ color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 钢铁表面缺陷检测报告</h1>
        <p class="subtitle">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">总图像数</div>
                <div class="stat-value">{stats['总图像数']}</div>
            </div>
            <div class="stat-card success">
                <div class="stat-label">检测成功</div>
                <div class="stat-value">{stats['检测成功']}</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-label">有缺陷</div>
                <div class="stat-value">{stats['有缺陷']}</div>
            </div>
            <div class="stat-card info">
                <div class="stat-label">总缺陷数</div>
                <div class="stat-value">{stats['总缺陷数']}</div>
            </div>
        </div>

        <h2>📊 详细结果</h2>
        {df.to_html(index=False, classes='data-table', escape=False).replace('<table', '<table')}

        <div class="timestamp">
            <p>本报告由钢铁表面缺陷检测系统自动生成</p>
        </div>
    </div>
</body>
</html>"""
        return html
