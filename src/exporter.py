"""
导出模块 - 支持 CSV、Bad Case 数据集和 HTML 报告导出。
"""

import csv
import json
import os
import base64
from datetime import datetime
from typing import Optional

from .db_manager import DBManager, InspectionRecord


def _encode_image_base64(image_path: str) -> str:
    """将图像编码为 Base64 数据 URI"""
    if not image_path or not os.path.exists(image_path):
        return ""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


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
                f.write("无数据\n")
                return output_path

            writer = csv.DictWriter(f, fieldnames=[
                "id", "timestamp", "defect_types", "defect_count",
                "confidence", "review_status", "reviewer", "note",
            ])
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "id": r.id,
                    "timestamp": r.timestamp,
                    "defect_types": r.defect_types,
                    "defect_count": r.defect_count,
                    "confidence": r.confidence,
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
        """导出 Bad Case 数据集 (低置信度 + 被修正的记录)"""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "badcase")

        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # 查询低置信度或已修正的记录
        records = self.db.query(review_status="corrected", limit=limit)
        records += self.db.query(limit=limit)  # 再补充一些最近的

        annotations = []
        for r in records[:limit]:
            if r.image_path and os.path.exists(r.image_path):
                annotations.append({
                    "image_path": r.image_path,
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

        return output_dir

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

    def export_inspection_report(
        self,
        output_path: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> str:
        """导出专业质检报告 (含图像 + 坐标 + 统计分析)"""
        if output_path is None:
            output_path = os.path.join(
                self.output_dir,
                f"inspection_report_{datetime.now():%Y%m%d_%H%M%S}.html",
            )

        records = self.db.query(start_time=start_time, end_time=end_time, limit=200)
        total = len(records)
        total_defects = sum(r.defect_count or 0 for r in records)
        defect_rate = (total_defects / total * 100) if total > 0 else 0
        reviewed = sum(1 for r in records if r.review_status == "confirmed")

        # 缺陷类型统计
        defect_type_counts: dict[str, int] = {}
        confidences = []
        for r in records:
            for dt in (r.defect_types or "").split(","):
                dt = dt.strip()
                if dt:
                    defect_type_counts[dt] = defect_type_counts.get(dt, 0) + 1
            if r.confidence:
                confidences.append(r.confidence)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        # 生成记录行
        rows = ""
        for r in records[:50]:  # 最多50条详情
            yolo = json.loads(r.yolo_result) if r.yolo_result else {}
            dets = yolo.get("detections", [])
            det_html = ""
            if dets:
                det_html = "<ul style='margin:0;padding-left:16px;font-size:12px'>"
                for d in dets[:5]:
                    bbox = d.get("bbox", [0, 0, 0, 0])
                    cn = d.get("class_name", "?")
                    conf = d.get("confidence", 0)
                    det_html += (
                        f"<li>{cn} {conf:.1%} "
                        f"[{bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f},{bbox[3]:.3f}]</li>"
                    )
                det_html += "</ul>"

            img_b64 = _encode_image_base64(r.image_path)
            img_tag = f'<img src="data:image/jpeg;base64,{img_b64}" style="max-width:120px;border-radius:6px">' if img_b64 else "—"

            status_badge = {
                "pending": '<span style="background:#FEFCBF;color:#975A16;padding:2px 8px;border-radius:10px;font-size:11px">待审核</span>',
                "confirmed": '<span style="background:#C6F6D5;color:#276749;padding:2px 8px;border-radius:10px;font-size:11px">已确认</span>',
                "corrected": '<span style="background:#FED7D7;color:#C53030;padding:2px 8px;border-radius:10px;font-size:11px">已修正</span>',
            }.get(r.review_status or "pending", "—")

            rows += f"""<tr>
                <td>{img_tag}</td>
                <td>{r.id}</td>
                <td>{(r.timestamp or '')[:19]}</td>
                <td>{r.defect_types or '—'}</td>
                <td>{r.defect_count or 0}</td>
                <td>{r.confidence:.1%}</td>
                <td>{status_badge}</td>
                <td style="font-size:11px">{det_html}</td>
            </tr>"""

        # 类型统计行
        type_rows = "".join(
            f"<tr><td>{k}</td><td style='text-align:center;font-weight:700'>{v}</td>"
            f"<td>{v/total_defects*100:.1f}%</td></tr>"
            for k, v in sorted(defect_type_counts.items(), key=lambda x: -x[1])
        ) if total_defects > 0 else '<tr><td colspan="3">暂无数据</td></tr>'

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>钢铁表面缺陷检测报告 - {datetime.now():%Y-%m-%d}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Microsoft YaHei','SimHei',sans-serif; color:#1a202c; background:#f1f5f9; }}
.header {{ background:linear-gradient(135deg,#0d47a1,#1565c0); color:#fff; padding:28px 32px; }}
.header h1 {{ font-size:22px; letter-spacing:2px; }}
.header p {{ font-size:13px; opacity:0.85; margin-top:4px; }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; padding:24px 32px; max-width:1100px; margin:0 auto; }}
.card {{ background:#fff; border-radius:12px; padding:20px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06); }}
.card .value {{ font-size:28px; font-weight:800; }}
.card .label {{ font-size:11px; color:#94a3b8; text-transform:uppercase; margin-top:6px; letter-spacing:0.5px; }}
.container {{ max-width:1100px; margin:0 auto; padding:0 32px 40px; }}
.section {{ background:#fff; border-radius:12px; padding:24px; margin-bottom:20px; box-shadow:0 2px 8px rgba(0,0,0,0.06); }}
.section h2 {{ font-size:16px; color:#0d47a1; margin-bottom:16px; padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ background:#f8fafc; color:#475569; font-weight:600; padding:10px 12px; text-align:left; border-bottom:2px solid #e2e8f0; font-size:11px; text-transform:uppercase; letter-spacing:0.5px; }}
td {{ padding:10px 12px; border-bottom:1px solid #f1f5f9; vertical-align:middle; }}
tr:hover {{ background:#f8fafc; }}
.footer {{ text-align:center; color:#94a3b8; font-size:11px; padding:20px; border-top:1px solid #e2e8f0; margin-top:20px; }}
.detail-box {{ max-height:500px; overflow-y:auto; }}
@media print {{ body {{ background:#fff; }} .header {{ -webkit-print-color-adjust:exact; }} }}
</style>
</head>
<body>
<div class="header">
    <h1>🔬 钢铁表面缺陷检测报告</h1>
    <p>Steel Surface Defect Inspection Report · 自动生成 · {datetime.now():%Y-%m-%d %H:%M}</p>
</div>

<div class="cards">
    <div class="card">
        <div class="value" style="color:#1e40af">{total}</div>
        <div class="label">📊 检测总数</div>
    </div>
    <div class="card">
        <div class="value" style="color:#dc2626">{total_defects}</div>
        <div class="label">⚠️ 检出缺陷</div>
    </div>
    <div class="card">
        <div class="value" style="color:#d97706">{defect_rate:.1f}%</div>
        <div class="label">📈 缺陷率</div>
    </div>
    <div class="card">
        <div class="value" style="color:#16a34a">{(avg_conf*100):.1f}%</div>
        <div class="label">✅ 均置信度</div>
    </div>
</div>

<div class="container">
    <div class="section">
        <h2>📋 缺陷类型分布</h2>
        <table>
            <tr><th>缺陷类型</th><th style="text-align:center">数量</th><th style="text-align:center">占比</th></tr>
            {type_rows}
        </table>
    </div>

    <div class="section">
        <h2>🔍 检测记录详情</h2>
        <p style="font-size:11px;color:#94a3b8;margin-bottom:12px">时间范围: {start_time or '全部'} ~ {end_time or '全部'} · 共 {total} 条 · 已审核 {reviewed} 条</p>
        <div class="detail-box">
        <table>
            <tr><th style="width:130px">图像</th><th>ID</th><th>时间</th><th>缺陷类型</th><th style="text-align:center">数量</th><th style="text-align:center">置信度</th><th>状态</th><th>坐标详情</th></tr>
            {rows or '<tr><td colspan="8" style="text-align:center;padding:40px;color:#94a3b8">暂无检测记录</td></tr>'}
        </table>
        </div>
    </div>
</div>

<div class="footer">
    钢铁表面缺陷检测系统 V1.0 · YOLO + VLM 双引擎架构 · 对照需求规格说明书 V1.0 · 报告自动生成于 {datetime.now():%Y-%m-%d %H:%M:%S}
</div>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path
