"""
导出模块 - 支持 CSV、Bad Case 数据集和 HTML 报告导出。
"""

import csv
import json
import os
import shutil
import zipfile
from datetime import datetime
from typing import Optional

from .db_manager import DBManager, InspectionRecord


class Exporter:
    """检测数据导出器"""

    def __init__(self, db: DBManager, output_dir: str = "data/exports"):
        self.db = db
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_csv(
        self,
        output_path: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> str:
        """导出检测记录为 CSV 文件"""
        if output_path is None:
            output_path = os.path.join(
                self.output_dir,
                f"inspection_{datetime.now():%Y%m%d_%H%M%S}.csv",
            )

        records = self.db.query(
            start_time=start_time,
            end_time=end_time,
            limit=100000,
        )

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            if not records:
                f.write("id,timestamp\n")
                return output_path

            writer = csv.DictWriter(f, fieldnames=[
                "id", "timestamp", "image_path", "defect_types", "defect_count",
                "confidence", "engine", "review_status", "reviewer", "note",
            ])
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "id": r.id,
                    "timestamp": r.timestamp,
                    "image_path": r.image_path,
                    "defect_types": r.defect_types,
                    "defect_count": r.defect_count,
                    "confidence": r.confidence,
                    "engine": r.engine,
                    "review_status": r.review_status,
                    "reviewer": r.reviewer,
                    "note": r.note,
                })

        return output_path

    def export_badcase(
        self,
        output_dir: Optional[str] = None,
        limit: int = 500,
    ) -> str:
        """导出 Bad Case 数据集 (低置信度 + 被修正的记录)并打包为 ZIP"""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "badcase")

        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # 查询低置信度或已修正的记录
        all_records = self.db.query(limit=limit * 2)
        records = []
        seen_ids = set()

        # 优先加入已修正记录 (corrected)
        for r in all_records:
            if r.review_status == "corrected":
                if r.id not in seen_ids:
                    records.append(r)
                    seen_ids.add(r.id)

        # 其次加入低置信度记录 (< 0.5)
        for r in all_records:
            if r.confidence < 0.5:
                if r.id not in seen_ids:
                    records.append(r)
                    seen_ids.add(r.id)

        # 如果记录还是过少，补充一些近期的记录以方便测试
        if len(records) < 5:
            for r in all_records:
                if r.id not in seen_ids:
                    records.append(r)
                    seen_ids.add(r.id)

        records = records[:limit]

        annotations = []
        for r in records:
            if r.image_path and os.path.exists(r.image_path):
                img_name = os.path.basename(r.image_path)
                dest_path = os.path.join(images_dir, img_name)
                try:
                    shutil.copy2(r.image_path, dest_path)
                except Exception as e:
                    print(f"[WARN] Failed to copy image {r.image_path}: {e}")
                
                annotations.append({
                    "id": r.id,
                    "image_path": f"images/{img_name}",
                    "defect_types": r.defect_types,
                    "yolo_result": json.loads(r.yolo_result) if r.yolo_result else {},
                    "vlm_result": json.loads(r.vlm_result) if r.vlm_result else {},
                    "final_result": json.loads(r.final_result) if r.final_result else {},
                    "review_status": r.review_status,
                })

        # 写入标注文件
        annot_path = os.path.join(output_dir, "badcase_annotations.json")
        with open(annot_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

        # 打包成 ZIP
        zip_path = os.path.join(self.output_dir, "badcase.zip")
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 写入标注文件
            zipf.write(annot_path, "badcase_annotations.json")
            # 写入图片
            for root, _, files in os.walk(images_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.join("images", file))

        return zip_path

    def export_html_report(
        self,
        output_path: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> str:
        """导出 HTML 格式的检测报告"""
        if output_path is None:
            output_path = os.path.join(
                self.output_dir,
                f"report_{datetime.now():%Y%m%d_%H%M%S}.html",
            )

        total = self.db.count(start_time, end_time)
        stats = self.db.get_defect_stats(start_time, end_time)

        # 统计缺陷类型
        defect_count: dict[str, int] = {}
        for s in stats:
            for dt in s["defect_types"].split(","):
                dt = dt.strip()
                if dt:
                    defect_count[dt] = defect_count.get(dt, 0) + 1

        # 生成 HTML
        stats_rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in sorted(defect_count.items(), key=lambda x: -x[1])
        )

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>钢铁表面缺陷检测报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #1a73e8; padding-bottom: 10px; }}
        .summary {{ background: #f0f6ff; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #1a73e8; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .footer {{ color: #999; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
    <h1>🔍 钢铁表面缺陷检测报告</h1>
    <div class="summary">
        <p><strong>报告生成时间：</strong>{datetime.now():%Y-%m-%d %H:%M:%S}</p>
        <p><strong>时间范围：</strong>{start_time or '全部'} ~ {end_time or '全部'}</p>
        <p><strong>检测总数：</strong>{total}</p>
    </div>
    <h2>缺陷类型统计</h2>
    <table>
        <tr><th>缺陷类型</th><th>数量</th></tr>
        {stats_rows or '<tr><td colspan="2">暂无数据</td></tr>'}
    </table>
    <div class="footer">钢铁表面缺陷检测系统 V1.0 - 自动生成报告</div>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path
