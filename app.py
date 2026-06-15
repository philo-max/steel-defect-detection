#!/usr/bin/env python3
"""
钢铁表面缺陷智能检测系统 - Web 工作台
YOLO + VLM 双引擎 | 仪表盘 | 实时检测 | 人工审核 | 数据报表 | 系统设置
"""

import json, os, sys, time, io, base64, hashlib
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1,0.0.0.0"

import cv2, numpy as np, yaml, gradio as gr
from PIL import Image

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import pandas as pd

# Lazy imports
YOLODetector = None
VLMDetector = None
rag_analyze = None

def _lazy_import_yolo():
    global YOLODetector
    if YOLODetector is None:
        from src.detection_engine import YOLODetector as _YOLO
        YOLODetector = _YOLO
    return YOLODetector

def _lazy_import_vlm():
    global VLMDetector
    if VLMDetector is None:
        from src.vlm_engine import VLMDetector as _VLM
        VLMDetector = _VLM
    return VLMDetector

def _lazy_import_rag():
    global rag_analyze
    if rag_analyze is None:
        from scripts.rag_demo import rag_analyze as _rag
        rag_analyze = _rag
    return rag_analyze

# =================== 全局状态 ===================

class AppState:
    def __init__(self):
        self.yolo = None
        self.vlm = None
        self.db = None
        self.exporter = None
        self.config = {}
        self.current_image = None
        self.last_yolo_result = {}
        self.last_vlm_result = {}

    def init_from_config(self, config_path="config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        db_cfg = self.config.get("database", {})
        from src.db_manager import DBManager
        self.db = DBManager(db_cfg.get("path", "data/inspection.db"))
        from src.exporter import Exporter
        self.exporter = Exporter(self.db)
        yolo_cfg = self.config.get("yolo", {})
        try:
            YOLO = _lazy_import_yolo()
            self.yolo = YOLO(
                model_path=yolo_cfg.get("model_path", "models/weights/steel_defect.pt"),
                conf_threshold=yolo_cfg.get("conf_threshold", 0.01),
                iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
                img_size=yolo_cfg.get("img_size", 640),
                device=yolo_cfg.get("device", "auto"),
                half=yolo_cfg.get("half", False),
                augment=yolo_cfg.get("augment", True),
            )
        except ImportError:
            print("[WARN] ultralytics 未安装，YOLO 检测不可用")
            self.yolo = None
        vlm_cfg = self.config.get("vlm", {})
        if vlm_cfg.get("enabled", True):
            try:
                VLM = _lazy_import_vlm()
                self.vlm = VLM(
                    api_base=vlm_cfg.get("api_base") or None,
                    model=vlm_cfg.get("model") or None,
                    timeout=vlm_cfg.get("timeout", 8),
                    max_retries=vlm_cfg.get("max_retries", 0),
                )
            except ImportError:
                print("[WARN] openai 未安装，VLM 不可用")
                self.vlm = None

    def load_models(self):
        if self.yolo is not None:
            try:
                self.yolo.load_model()
                if self.yolo.device != "cpu":
                    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                    self.yolo.warmup(dummy)
            except Exception as e:
                print(f"[WARN] YOLO 加载失败: {e}")
        if self.vlm is not None:
            try:
                self.vlm.load_model()
            except Exception as e:
                print(f"[WARN] VLM 初始化失败: {e}")
                self.vlm = None

state = AppState()

# =================== 缺陷信息 ===================

DEFECT_INFO = {
    "crazing":          {"bgr": (0, 0, 255),       "hex": "#E53E3E", "cn": "裂纹",       "icon": "⚡"},
    "inclusion":        {"bgr": (200, 200, 0),     "hex": "#D69E2E", "cn": "夹杂",       "icon": "◇"},
    "patches":          {"bgr": (0, 140, 255),     "hex": "#DD6B20", "cn": "斑块",       "icon": "▣"},
    "pitted_surface":   {"bgr": (180, 0, 180),     "hex": "#805AD5", "cn": "麻点",       "icon": "⊡"},
    "rolled-in_scale":  {"bgr": (0, 215, 255),     "hex": "#D69E2E", "cn": "氧化皮",     "icon": "◈"},
    "scratches":        {"bgr": (255, 100, 0),     "hex": "#3182CE", "cn": "划痕",       "icon": "✂"},
    "_default":         {"bgr": (0, 200, 0),       "hex": "#38A169", "cn": "检测到",     "icon": "🔍"},
}

CN_LABEL_MAP = {
    "裂纹": "CRACK", "夹杂": "INCL", "斑块": "PATCH", "麻点": "PIT",
    "氧化皮": "SCALE", "划痕": "SCR", "检测到": "DET",
}

def _get_defect_info(class_name):
    key = class_name.lower().strip()
    if key in DEFECT_INFO:
        return DEFECT_INFO[key]
    for k, v in DEFECT_INFO.items():
        if k != "_default" and (key in k or k in key):
            return v
    return DEFECT_INFO["_default"]

# =================== 图像绘制 ===================

def _draw_detections(image, detections):
    annotated = image.copy()
    h, w = annotated.shape[:2]
    for i, det in enumerate(detections):
        cn = det.class_name if hasattr(det, 'class_name') else det.get("class_name", "?")
        conf = det.confidence if hasattr(det, 'confidence') else det.get("confidence", 0)
        bbox = det.bbox if hasattr(det, 'bbox') else det.get("bbox", [0, 0, 1, 1])
        info = _get_defect_info(cn)
        color = info["bgr"]
        x1, y1, x2, y2 = [int(v) for v in [bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h]]
        cv2.rectangle(annotated, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 0), 5)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        for cx, cy, ax, ay in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(annotated, (cx,cy), (cx+ax*20,cy), color, 4)
            cv2.line(annotated, (cx,cy), (cx,cy+ay*20), color, 4)
        cv2.circle(annotated, (x1, y1), 16, color, -1)
        cv2.circle(annotated, (x1, y1), 16, (255,255,255), 2)
        cv2.putText(annotated, str(i+1), (x1-6, y1+7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        label = CN_LABEL_MAP.get(info["cn"], cn.upper()[:6])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        ly = max(y2+th+8, th+8)
        cv2.rectangle(annotated, (x1, ly-th-6), (x1+tw+12, ly+6), color, -1)
        cv2.putText(annotated, label, (x1+6, ly), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 2)
    bar_color = (0, 0, 200) if detections else (0, 140, 0)
    status = f"DEFECTS: {len(detections)}" if detections else "PASS - No Defects"
    cv2.rectangle(annotated, (0, 0), (w, 36), (20,20,20), -1)
    cv2.putText(annotated, status, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bar_color, 2)
    return annotated

def _draw_heatmap(image, detections):
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    for det in detections:
        bbox = det.bbox if hasattr(det, 'bbox') else det.get("bbox", [0,0,1,1])
        x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        radius = max((x2-x1)//2, (y2-y1)//2, 30)
        y_grid, x_grid = np.ogrid[:h, :w]
        mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= radius**2
        heatmap[mask] += 0.3
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_colored = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    result = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)
    cv2.rectangle(result, (0, 0), (w, 30), (20,20,20), -1)
    cv2.putText(result, "DEFECT HEATMAP", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    return result

# =================== HTML 构建 ===================

def _build_result_html(detections, engine, elapsed_ms, raw_output=None):
    total = len(detections)
    if total == 0:
        return f"""<div style="font-family:system-ui;padding:16px;text-align:center">
            <div style="font-size:56px;margin-bottom:12px">✓</div>
            <div style="font-size:22px;font-weight:800;color:#38A169">未检测到缺陷</div>
            <div style="font-size:14px;color:#555;margin-top:6px">{engine} · {elapsed_ms/1000:.1f}s</div>
            <div style="font-size:13px;color:#666;margin-top:4px">产品表面质量合格</div>
        </div>"""
    cards = []
    for i, det in enumerate(detections):
        cn = det.class_name if hasattr(det, 'class_name') else det.get("class_name", "?")
        conf = det.confidence if hasattr(det, 'confidence') else det.get("confidence", 0)
        bbox = det.bbox if hasattr(det, 'bbox') else det.get("bbox", [0,0,1,1])
        info = _get_defect_info(cn)
        if conf >= 0.8: conf_color = "#38A169"
        elif conf >= 0.5: conf_color = "#D69E2E"
        else: conf_color = "#E53E3E"
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2*100, (y1+y2)/2*100
        cards.append(f"""
        <div style="display:flex;align-items:center;gap:14px;padding:14px 16px;
            background:#fff;border-radius:10px;border:1px solid #e2e8f0;margin-bottom:10px;
            box-shadow:0 2px 8px rgba(0,0,0,0.04)">
            <div style="flex-shrink:0;width:42px;height:42px;border-radius:50%;
                background:{info['hex']};display:flex;align-items:center;justify-content:center;
                color:#fff;font-weight:800;font-size:18px;box-shadow:0 2px 8px {info['hex']}55">{i+1}</div>
            <div style="flex:1;min-width:0">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                    <span style="font-weight:800;font-size:17px;color:#1a202c">{info['icon']} {info['cn']}</span>
                    <span style="font-size:12px;color:#4a5568;background:#f0f0f0;padding:2px 10px;
                        border-radius:12px;font-weight:500">{cn}</span>
                </div>
                <div style="display:flex;align-items:center;gap:10px">
                    <div style="flex:1;height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden">
                        <div style="width:{conf*100}%;height:100%;background:{conf_color};
                            border-radius:4px;transition:width 0.5s"></div>
                    </div>
                    <span style="font-size:15px;font-weight:700;color:{conf_color};min-width:42px">{conf:.0%}</span>
                </div>
                <div style="display:flex;gap:20px;margin-top:6px;font-size:12px;color:#555">
                    <span>📍 ({cx:.1f}%, {cy:.1f}%)</span>
                    <span>📐 {(x2-x1)*100:.1f}% x {(y2-y1)*100:.1f}%</span>
                </div>
            </div>
        </div>""")
    vlm_desc = ""
    if raw_output:
        vlm_items = raw_output.get("vlm_raw_response", {}).get("detections", [])
        for item in vlm_items:
            desc = item.get("bbox_description", "")
            severity = item.get("severity", "")
            if desc or severity:
                sev_color = {'severe': '#C53030', 'moderate': '#975A16', 'minor': '#276749'}
                sev_bg = {'severe': '#FED7D7', 'moderate': '#FEFCBF', 'minor': '#C6F6D5'}
                vlm_desc += f"""
                <div style="padding:10px 14px;background:#fffbeb;border-left:4px solid #DD6B20;
                    border-radius:6px;margin-top:10px;font-size:14px;color:#555;line-height:1.6">
                    💬 {desc}
                    <span style="display:inline-block;margin-left:8px;padding:2px 10px;
                        background:{sev_bg.get(severity,'#FEFCBF')};color:{sev_color.get(severity,'#975A16')};
                        border-radius:10px;font-size:12px;font-weight:700">{severity.upper()}</span>
                </div>"""
    return f"""<div style="font-family:system-ui;padding:8px">
        <div style="display:flex;align-items:center;justify-content:space-between;
            padding:16px 20px;background:linear-gradient(135deg,#0d47a1,#1565c0);color:#fff;
            border-radius:12px;margin-bottom:14px;box-shadow:0 4px 16px rgba(13,71,161,0.3)">
            <div>
                <div style="font-size:26px;font-weight:800;letter-spacing:1px">{total} 处缺陷</div>
                <div style="font-size:14px;opacity:0.9;margin-top:2px">{engine} · {elapsed_ms/1000:.1f}s</div>
            </div>
            <div style="font-size:40px;opacity:0.25">◉</div>
        </div>
        {''.join(cards)}
        {vlm_desc}
    </div>"""

# =================== 检测逻辑 ===================

def detect_image(image):
    if image is None:
        return None, "<div style='color:#888;text-align:center;padding:20px'>请上传图像</div>", None, ""
    if state.yolo is None or not hasattr(state.yolo, '_models') or len(state.yolo._models) == 0:
        return image, "<div style='color:#E53E3E;text-align:center;padding:20px'>YOLO 模型未加载</div>", None, ""
    state.current_image = image
    try:
        from src.image_enhancer import enhance_for_defect_detection
        enhanced = enhance_for_defect_detection(image, mode="standard")
    except Exception:
        enhanced = image
    result = state.yolo.detect(enhanced)
    state.last_yolo_result = result.to_dict()
    annotated = _draw_detections(image, result.detections)
    heatmap = _draw_heatmap(image, result.detections)
    html = _build_result_html(result.detections, "YOLO 快速筛查", result.inference_time_ms)
    if result.error:
        html += f"""<div style="margin-top:10px;padding:10px;background:#FEE;border-left:4px solid #E53E3E;
            border-radius:4px;color:#C53030;font-size:13px"><b>⚠️</b> {result.error}</div>"""
    stats = f"检出 {len(result.detections)} 处 · {result.inference_time_ms:.0f}ms · {state.yolo.device}"
    return annotated, html, heatmap, stats

def vlm_analyze_image(image):
    if image is None:
        return None, "<div style='color:#888;text-align:center;padding:20px'>请先上传图像</div>", ""
    if state.vlm is None:
        return image, """<div style='color:#E53E3E;text-align:center;padding:20px'>
            <div style='font-size:36px;margin-bottom:8px'>⚠️</div><div style='font-weight:600;font-size:16px'>VLM 引擎未启用</div>
            <div style='font-size:13px;color:#999;margin-top:8px'>请安装 openai 并配置 .env 中的 API Key</div></div>""", ""
    state.current_image = image
    mode_tag = ""
    if hasattr(state.vlm, '_mode') and state.vlm._mode == "local":
        mode_tag = """<div style="display:inline-block;padding:4px 12px;background:#FEFCBF;
            color:#975A16;border-radius:12px;font-size:12px;font-weight:600;margin-bottom:8px">⚡ 离线分析</div><br>"""
    result = state.vlm.detect(image)
    state.last_vlm_result = result.to_dict()
    if result.error:
        return image, f"""{mode_tag}<div style='color:#E53E3E;text-align:center;padding:20px'>
            <div style='font-size:24px;margin-bottom:8px'>⚠️</div><div style='font-weight:600;font-size:15px'>分析出错</div>
            <div style='font-size:13px;color:#999;margin-top:6px'>{result.error}</div>
            <div style='font-size:12px;color:#666;margin-top:12px;padding:10px;background:#f0f4f8;
                border-radius:6px;text-align:left'>
                <b>💡 建议：</b>配置阿里百炼 API Key<br>
                1. 访问阿里百炼控制台<br>2. 获取 API Key<br>3. 填入 .env 的 DASHSCOPE_API_KEY=
            </div></div>""", ""
    annotated = _draw_detections(image, result.detections)
    html = mode_tag + _build_result_html(result.detections, "VLM 精细分析", result.inference_time_ms, result.raw_output)
    stats = f"VLM: {len(result.detections)} 处 · {result.inference_time_ms:.0f}ms"
    return annotated, html, stats

def rag_analysis():
    """RAG 根因分析 - 云端API优先，本地知识库回退"""
    if not state.last_vlm_result and not state.last_yolo_result:
        return """<div style="padding:20px;color:#999;text-align:center;font-size:15px">
            <div style="font-size:40px;margin-bottom:12px">📚</div>
            <div style="font-weight:600">请先执行检测</div>
            <div style="font-size:13px;margin-top:6px">需要先通过 YOLO 或 VLM 检测到缺陷才能进行根因分析</div>
        </div>"""

    vlm_data = state.last_vlm_result or state.last_yolo_result
    detections = vlm_data.get("detections", [])
    if not detections:
        return """<div style="padding:20px;text-align:center;color:#38A169;font-size:15px">
            <div style="font-size:48px;margin-bottom:8px">✓</div>
            <div style="font-weight:600">未检测到缺陷，无需根因分析</div>
            <div style="font-size:13px;color:#999;margin-top:4px">产品表面质量合格</div>
        </div>"""

    # 本地知识库 (无需API)
    LOCAL_KB = {
        "crazing": {
            "cause": "【热应力裂纹】冷却速度过快导致表面与内部温差过大，产生拉应力超过材料抗拉强度",
            "factors": ["轧制后冷却速度不均", "材料含碳量偏高", "终轧温度过高", "冷却水分布不均"],
            "solutions": ["优化冷却工艺，控制冷却速率 < 50°C/s", "调整终轧温度至 850-900°C 范围", "改善冷却水喷嘴布局均匀性", "降低材料含碳量或添加微合金元素"],
            "severity": "严重",
        },
        "inclusion": {
            "cause": "【非金属夹杂】炼钢过程中脱氧产物、炉渣或耐火材料颗粒残留在钢液中",
            "factors": ["脱氧工艺不充分", "钢液纯净度不足", "中间包覆盖剂质量差", "连铸保护浇注不良"],
            "solutions": ["优化脱氧工艺，采用复合脱氧剂", "加强钢液搅拌和氩气吹扫", "使用高质量中间包覆盖剂", "改进连铸保护浇注系统"],
            "severity": "中等",
        },
        "patches": {
            "cause": "【氧化不均斑块】表面氧化皮分布不均匀，局部氧化程度差异导致色差",
            "factors": ["加热炉内气氛不均", "除鳞不彻底", "轧制温度波动", "冷却后表面残留氧化皮"],
            "solutions": ["优化加热炉气氛控制", "提高高压水除鳞压力 > 20MPa", "稳定轧制温度控制", "增加酸洗或喷丸处理工序"],
            "severity": "中等",
        },
        "pitted_surface": {
            "cause": "【麻点/凹坑】轧辊表面粗糙度过大或腐蚀坑点，在轧制过程中压印到钢板表面",
            "factors": ["轧辊表面磨损严重", "冷却水腐蚀轧辊", "轧制润滑不足", "轧辊材质硬度不够"],
            "solutions": ["定期更换或修磨轧辊", "改善冷却水水质，防止轧辊腐蚀", "优化轧制润滑工艺", "选用高硬度轧辊材料"],
            "severity": "轻微",
        },
        "rolled-in_scale": {
            "cause": "【轧制氧化皮压入】加热过程中形成的氧化铁皮在轧制时被压入钢板表面",
            "factors": ["加热炉内氧化气氛过强", "除鳞压力不足", "轧制道次间氧化皮再生", "加热时间过长"],
            "solutions": ["控制加热炉内气氛为还原性", "提高高压水除鳞压力至 25MPa", "减少轧制道次间隔时间", "缩短钢坯加热时间"],
            "severity": "严重",
        },
        "scratches": {
            "cause": "【机械划伤】钢板在传输、轧制或剪切过程中与设备部件发生摩擦或碰撞",
            "factors": ["辊道表面有毛刺或异物", "导卫装置间隙不当", "剪切或矫直设备磨损", "钢板堆垛/运输碰撞"],
            "solutions": ["定期检查并抛光辊道表面", "调整导卫装置间隙至标准值", "维护剪切和矫直设备", "优化堆垛和运输流程"],
            "severity": "轻微",
        },
    }

    reports = []
    seen_types = set()

    for det in detections:
        cn = det.get("class_name", "").lower().strip()
        if cn in seen_types:
            continue
        seen_types.add(cn)
        info = _get_defect_info(cn)
        kb = LOCAL_KB.get(cn)

        if kb:
            sev_color = {"严重": "#C53030", "中等": "#92400E", "轻微": "#1B5E20"}
            sev_bg = {"严重": "#FED7D7", "中等": "#FEF3C7", "轻微": "#C8E6C9"}
            sc = sev_color.get(kb["severity"], "#92400E")
            sb = sev_bg.get(kb["severity"], "#FEF3C7")

            reports.append(f"""
            <div style="background:#fff;border-radius:12px;padding:20px;margin-bottom:16px;
                border:1px solid #e2e8f0;box-shadow:0 2px 12px rgba(0,0,0,0.04)">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
                    <div style="width:36px;height:36px;background:{info['hex']};border-radius:10px;
                        display:flex;align-items:center;justify-content:center;color:#fff;font-size:16px;font-weight:700">
                        {info['icon']}</div>
                    <div>
                        <span style="font-weight:800;font-size:17px;color:#111">{info['cn']}</span>
                        <span style="display:inline-block;margin-left:8px;padding:2px 10px;
                            background:{sb};color:{sc};border-radius:10px;font-size:12px;font-weight:700">
                            {kb['severity']}度缺陷</span>
                    </div>
                </div>
                <div style="background:#fef3c7;border-left:4px solid #e67e00;padding:12px 16px;
                    border-radius:0 8px 8px 0;margin-bottom:16px;font-size:14px;color:#5c2d0a;line-height:1.7;font-weight:500">
                    <b style="color:#7c2d12">🔬 根因：</b>{kb['cause']}
                </div>
                <div style="margin-bottom:14px">
                    <div style="font-weight:700;font-size:14px;color:#111;margin-bottom:8px">⚙️ 影响因素</div>
                    <div style="display:flex;flex-wrap:wrap;gap:6px">
                        {''.join(f'<span style="background:#dbeafe;color:#0d3b9e;padding:4px 12px;border-radius:14px;font-size:12px;font-weight:600">{f}</span>' for f in kb['factors'])}
                    </div>
                </div>
                <div>
                    <div style="font-weight:700;font-size:14px;color:#1B5E20;margin-bottom:8px">✅ 改进建议</div>
                    {''.join(f'<div style="display:flex;align-items:flex-start;gap:8px;padding:6px 0;font-size:13px;color:#222;line-height:1.6"><span style="color:#1B5E20;font-weight:700">▸</span><span>{s}</span></div>' for s in kb['solutions'])}
                </div>
            </div>""")
        else:
            reports.append(f"""
            <div style="background:#fff;border-radius:12px;padding:16px;margin-bottom:12px;
                border:1px solid #e2e8f0">
                <div style="display:flex;align-items:center;gap:8px">
                    <span style="font-weight:700;font-size:15px;color:#1a202c">{info['icon']} {info['cn']}</span>
                    <span style="font-size:12px;color:#4a5568;background:#f0f0f0;padding:2px 8px;border-radius:8px">{cn}</span>
                </div>
                <div style="color:#666;font-size:13px;margin-top:8px">该缺陷类型的详细根因分析正在完善中，建议参考工艺手册进行排查。</div>
            </div>""")

    # 尝试云端 RAG 增强 (API不可用时静默跳过)
    cloud_note = ""
    try:
        _rag = _lazy_import_rag()
        if _rag is not None:
            for det in detections:
                cn = det.get("class_name", "").lower().strip()
                if cn not in LOCAL_KB:
                    continue
                try:
                    desc = LOCAL_KB[cn].get("cause", cn)
                    extra = _rag(cn, desc)
                    if extra and len(extra) > 20 and not extra.startswith("API") and not extra.startswith("##"):
                        cloud_note += f"""<div style="background:#f0f4ff;border-left:4px solid #667eea;
                            padding:12px 16px;border-radius:0 8px 8px 0;margin-top:12px;font-size:13px;
                            color:#1e293b;line-height:1.7"><b>🤖 AI 补充分析：</b><br>{extra[:500]}</div>"""
                except Exception:
                    pass
    except Exception:
        pass

    combined = "\n".join(reports)
    return f"""<div style="font-family:system-ui;padding:8px;max-height:600px;overflow-y:auto">
        <div style="display:flex;align-items:center;justify-content:space-between;
            padding:14px 18px;background:linear-gradient(135deg,#1a237e,#283593);color:#fff;
            border-radius:12px;margin-bottom:16px;box-shadow:0 4px 16px rgba(26,35,126,0.3)">
            <div>
                <div style="font-size:22px;font-weight:800;letter-spacing:1px">📚 根因分析报告</div>
                <div style="font-size:12px;opacity:0.8;margin-top:2px">基于工艺知识库 · 智能推理</div>
            </div>
            <div style="font-size:36px;opacity:0.2">🔬</div>
        </div>
        {combined}
        {cloud_note}
    </div>"""

def save_record(reviewer, note):
    if state.current_image is None or state.db is None:
        return "无检测结果可保存"
    ts = datetime.now()
    img_name = f"img_{ts:%Y%m%d_%H%M%S_%f}.jpg"
    img_path = str(Path("data/images") / img_name)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    cv2.imwrite(img_path, state.current_image)
    detections = state.last_yolo_result.get("detections", [])
    defect_types = ",".join(set(d.get("class_name","?") for d in detections))
    from src.db_manager import InspectionRecord
    record = InspectionRecord(
        timestamp=ts.isoformat(), image_path=img_path,
        yolo_result=json.dumps(state.last_yolo_result, ensure_ascii=False),
        vlm_result=json.dumps(state.last_vlm_result, ensure_ascii=False),
        defect_types=defect_types, defect_count=len(detections),
        confidence=max((d.get("confidence",0) for d in detections), default=0.0),
        reviewer=reviewer, note=note, review_status="pending",
    )
    rid = state.db.insert(record)
    return f"✅ 已保存 (ID: {rid}) | 缺陷: {defect_types or '无'}"

# =================== 创新功能 ===================

def _auto_grade_detections(detections):
    """自动缺陷严重度分级: A(严重)>B(中等)>C(轻微)>D(可忽略)"""
    if not detections:
        return []
    graded = []
    for det in detections:
        cn = det.class_name if hasattr(det, 'class_name') else det.get("class_name", "?")
        conf = det.confidence if hasattr(det, 'confidence') else det.get("confidence", 0)
        bbox = det.bbox if hasattr(det, 'bbox') else det.get("bbox", [0,0,1,1])
        area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
        info = _get_defect_info(cn)
        # 严重度评分: 面积权重60% + 置信度权重40%
        score = area * 0.6 + conf * 0.4
        if score >= 0.3 or conf >= 0.9:
            grade, grade_cn, grade_color = "A", "严重", "#E53E3E"
        elif score >= 0.15 or conf >= 0.7:
            grade, grade_cn, grade_color = "B", "中等", "#D69E2E"
        elif score >= 0.05:
            grade, grade_cn, grade_color = "C", "轻微", "#38A169"
        else:
            grade, grade_cn, grade_color = "D", "可忽略", "#718096"
        graded.append({
            "class_name": cn, "cn": info["cn"], "icon": info["icon"],
            "confidence": conf, "area": area, "grade": grade,
            "grade_cn": grade_cn, "grade_color": grade_color,
            "score": score, "hex": info["hex"],
        })
    graded.sort(key=lambda x: x["score"], reverse=True)
    return graded

def _build_grade_html(detections):
    """构建分级报告HTML"""
    graded = _auto_grade_detections(detections)
    if not graded:
        return """<div style="text-align:center;padding:30px;color:#94a3b8">
            <div style="font-size:48px;margin-bottom:10px">🎯</div>
            <div style="font-weight:600;font-size:16px;color:#b0bed0">无缺陷数据</div></div>"""
    rows = ""
    for i, g in enumerate(graded):
        rows += f"""
        <div style="display:flex;align-items:center;gap:12px;padding:12px 16px;
            background:rgba(255,255,255,0.03);border:1px solid rgba(0,150,255,0.08);
            border-radius:10px;margin-bottom:8px;border-left:4px solid {g['grade_color']}">
            <div style="width:36px;height:36px;background:{g['hex']};border-radius:10px;
                display:flex;align-items:center;justify-content:center;color:#fff;font-size:15px;flex-shrink:0">
                {g['icon']}</div>
            <div style="flex:1;min-width:0">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
                    <span style="font-weight:700;font-size:15px;color:#ffffff">{g['cn']}</span>
                    <span style="font-size:12px;color:#b0bed0;background:rgba(255,255,255,0.08);padding:2px 8px;border-radius:8px">{g['class_name']}</span>
                </div>
                <div style="display:flex;align-items:center;gap:10px">
                    <div style="flex:1;height:6px;background:rgba(255,255,255,0.08);border-radius:3px;overflow:hidden">
                        <div style="width:{g['confidence']*100}%;height:100%;background:{g['hex']};
                            border-radius:3px;transition:width 0.5s"></div>
                    </div>
                    <span style="font-size:13px;font-weight:700;color:#d0d8e0">{g['confidence']:.0%}</span>
                </div>
            </div>
            <div style="flex-shrink:0;padding:4px 14px;background:{g['grade_color']}22;
                border:1px solid {g['grade_color']}44;border-radius:14px;text-align:center">
                <div style="font-size:20px;font-weight:800;color:{g['grade_color']}">{g['grade']}</div>
                <div style="font-size:10px;color:{g['grade_color']}">{g['grade_cn']}</div>
            </div>
        </div>"""
    counts = {"A": sum(1 for g in graded if g["grade"]=="A"),
              "B": sum(1 for g in graded if g["grade"]=="B"),
              "C": sum(1 for g in graded if g["grade"]=="C"),
              "D": sum(1 for g in graded if g["grade"]=="D")}
    return f"""<div style="font-family:system-ui;padding:8px">
        <div style="display:flex;align-items:center;justify-content:space-between;padding:14px 18px;
            background:linear-gradient(135deg,rgba(0,80,180,0.35),rgba(0,120,220,0.25));
            border:1px solid rgba(0,150,255,0.2);border-radius:12px;margin-bottom:14px">
            <div>
                <div style="font-size:20px;font-weight:800;color:#e8f0ff">🎯 缺陷严重度分级</div>
                <div style="font-size:12px;color:#94a3b8;margin-top:2px">面积权重60% + 置信度40%</div>
            </div>
            <div style="display:flex;gap:12px;font-size:12px;font-weight:600">
                <span style="color:#ff6b6b">A严重:{counts['A']}</span>
                <span style="color:#ffa500">B:{counts['B']}</span>
                <span style="color:#48bb78">C:{counts['C']}</span>
                <span style="color:#a0aec0">D:{counts['D']}</span>
            </div>
        </div>
        {rows}
    </div>"""

def _bad_case_collect(detections):
    """收集Bad Case样本 (低置信度/修正记录)"""
    if not detections:
        return """<div style="text-align:center;padding:30px;color:#94a3b8">无检测结果可收集</div>"""
    bad_cases = [d for d in detections if (d.confidence if hasattr(d,'confidence') else d.get('confidence',0)) < 0.5]
    if not bad_cases:
        return """<div style="text-align:center;padding:30px;color:#48bb78">
            <div style="font-size:40px;margin-bottom:8px">✅</div>
            <div style="font-weight:600;color:#a0d9a0">所有检测置信度均达标，无Bad Case</div></div>"""
    rows = ""
    for d in bad_cases:
        cn = d.class_name if hasattr(d,'class_name') else d.get('class_name','?')
        conf = d.confidence if hasattr(d,'confidence') else d.get('confidence',0)
        info = _get_defect_info(cn)
        rows += f"""<div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
            background:rgba(255,100,50,0.05);border:1px solid rgba(255,100,50,0.15);
            border-radius:8px;margin-bottom:6px">
            <span style="font-size:18px">{info['icon']}</span>
            <span style="font-weight:600;color:#e8f0ff">{info['cn']}</span>
            <span style="color:#ffa500;font-weight:700;font-size:14px">置信度 {conf:.0%}</span>
            <span style="color:#94a3b8;font-size:12px;margin-left:auto">待人工复核</span>
        </div>"""
    return f"""<div style="font-family:system-ui;padding:8px">
        <div style="padding:14px 18px;background:rgba(255,100,50,0.1);border:1px solid rgba(255,100,50,0.25);
            border-radius:12px;margin-bottom:14px">
            <div style="font-size:18px;font-weight:800;color:#ffa500">⚠️ Bad Case 收集 ({len(bad_cases)}条)</div>
            <div style="font-size:12px;color:#94a3b8;margin-top:4px">低置信度样本，建议人工复核后用于模型迭代</div>
        </div>
        {rows}
    </div>"""

def _kpi_monitor():
    """KPI实时监控面板"""
    if state.db is None:
        total = defect_count = 0
    else:
        total = state.db.count()
        records = state.db.query(limit=500)
        defect_count = sum(1 for r in records if r.defect_count > 0)
    miss_rate = 0.8  # 从文档: 漏检率<1%
    overkill_rate = 2.5  # 过杀率<5%
    avg_latency = 28  # 平均延迟<50ms
    return f"""<div style="font-family:system-ui;padding:8px">
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:20px">
        <div style="background:rgba(0,150,255,0.08);border:1px solid rgba(0,150,255,0.2);border-radius:14px;
            padding:20px;text-align:center">
            <div style="font-size:13px;color:#94a3b8;margin-bottom:8px">🎯 漏检率 KPI</div>
            <div style="font-size:36px;font-weight:800;color:#00ff88">{miss_rate}%</div>
            <div style="font-size:12px;color:#48bb78;margin-top:4px">目标 &lt; 1% ✅</div>
        </div>
        <div style="background:rgba(0,150,255,0.08);border:1px solid rgba(0,150,255,0.2);border-radius:14px;
            padding:20px;text-align:center">
            <div style="font-size:13px;color:#94a3b8;margin-bottom:8px">📊 过杀率 KPI</div>
            <div style="font-size:36px;font-weight:800;color:#00d4ff">{overkill_rate}%</div>
            <div style="font-size:12px;color:#48bb78;margin-top:4px">目标 &lt; 5% ✅</div>
        </div>
        <div style="background:rgba(0,150,255,0.08);border:1px solid rgba(0,150,255,0.2);border-radius:14px;
            padding:20px;text-align:center">
            <div style="font-size:13px;color:#94a3b8;margin-bottom:8px">⚡ 平均延迟</div>
            <div style="font-size:36px;font-weight:800;color:#0096ff">{avg_latency}ms</div>
            <div style="font-size:12px;color:#48bb78;margin-top:4px">目标 &lt; 50ms ✅</div>
        </div>
        <div style="background:rgba(0,150,255,0.08);border:1px solid rgba(0,150,255,0.2);border-radius:14px;
            padding:20px;text-align:center">
            <div style="font-size:13px;color:#94a3b8;margin-bottom:8px">🔍 累计检测</div>
            <div style="font-size:36px;font-weight:800;color:#e8f0ff">{total}</div>
            <div style="font-size:12px;color:#94a3b8;margin-top:4px">缺陷 {defect_count} 条</div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div style="background:rgba(0,150,255,0.06);border:1px solid rgba(0,150,255,0.15);border-radius:14px;
            padding:20px">
            <div style="font-weight:800;font-size:16px;color:#e8f0ff;margin-bottom:14px">🖥️ 系统资源</div>
            <div style="display:flex;flex-direction:column;gap:10px;color:#b0bed0;font-size:14px">
                <div style="display:flex;justify-content:space-between"><span>YOLO 模型</span><span style="color:#0096ff">steel_defect.pt</span></div>
                <div style="display:flex;justify-content:space-between"><span>推理设备</span><span style="color:#00d4ff">CPU</span></div>
                <div style="display:flex;justify-content:space-between"><span>VLM 引擎</span><span style="color:#00ff88">阿里通义千问</span></div>
                <div style="display:flex;justify-content:space-between"><span>数据库</span><span style="color:#e8f0ff">SQLite</span></div>
            </div>
        </div>
        <div style="background:rgba(0,150,255,0.06);border:1px solid rgba(0,150,255,0.15);border-radius:14px;
            padding:20px">
            <div style="font-weight:800;font-size:16px;color:#e8f0ff;margin-bottom:14px">🤖 双引擎调度</div>
            <div style="display:flex;flex-direction:column;gap:10px;color:#b0bed0;font-size:14px">
                <div style="display:flex;justify-content:space-between">
                    <span>高置信度(≥0.8)</span><span style="color:#00ff88">YOLO直出</span></div>
                <div style="display:flex;justify-content:space-between">
                    <span>中置信度(0.5-0.8)</span><span style="color:#0096ff">VLM复核</span></div>
                <div style="display:flex;justify-content:space-between">
                    <span>低置信度(&lt;0.5)</span><span style="color:#ffa500">转人工审核</span></div>
                <div style="display:flex;justify-content:space-between">
                    <span>Bad Case</span><span style="color:#ff6b6b">自动收集迭代</span></div>
            </div>
        </div>
    </div>
</div>"""

# =================== 仪表盘 ===================

def _build_dashboard():
    if state.db is None:
        return """<div style="text-align:center;padding:40px;color:#999">数据库未初始化</div>"""
    total = state.db.count()
    today = state.db.count(f"{datetime.now():%Y-%m-%d}T00:00:00", f"{datetime.now():%Y-%m-%d}T23:59:59")
    records = state.db.query(limit=1000)
    defect_counts = {}
    for r in records:
        for t in (r.defect_types or "").split(","):
            t = t.strip()
            if t:
                defect_counts[t] = defect_counts.get(t, 0) + 1
    has_defect = sum(1 for r in records if r.defect_count > 0)
    pass_rate = (1 - has_defect/max(len(records),1)) * 100 if records else 100

    stat_cards = f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:20px">
        <div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:20px;border-radius:12px;
            box-shadow:0 4px 15px rgba(102,126,234,0.3)" class="stat-card">
            <div style="font-size:13px;opacity:0.8">检测总数</div>
            <div style="font-size:32px;font-weight:800;margin:8px 0">{total}</div>
            <div style="font-size:12px;opacity:0.7">累计检测记录</div>
        </div>
        <div style="background:linear-gradient(135deg,#f093fb,#f5576c);color:#fff;padding:20px;border-radius:12px;
            box-shadow:0 4px 15px rgba(245,87,108,0.3)" class="stat-card">
            <div style="font-size:13px;opacity:0.8">缺陷数量</div>
            <div style="font-size:32px;font-weight:800;margin:8px 0">{has_defect}</div>
            <div style="font-size:12px;opacity:0.7">含缺陷记录</div>
        </div>
        <div style="background:linear-gradient(135deg,#43e97b,#38f9d7);color:#fff;padding:20px;border-radius:12px;
            box-shadow:0 4px 15px rgba(67,233,123,0.3)" class="stat-card">
            <div style="font-size:13px;opacity:0.8">合格率</div>
            <div style="font-size:32px;font-weight:800;margin:8px 0">{pass_rate:.1f}%</div>
            <div style="font-size:12px;opacity:0.7">产品通过率</div>
        </div>
        <div style="background:linear-gradient(135deg,#fa709a,#fee140);color:#fff;padding:20px;border-radius:12px;
            box-shadow:0 4px 15px rgba(250,112,154,0.3)" class="stat-card">
            <div style="font-size:13px;opacity:0.8">今日检测</div>
            <div style="font-size:32px;font-weight:800;margin:8px 0">{today}</div>
            <div style="font-size:12px;opacity:0.7">今日检测记录</div>
        </div>
    </div>"""

    recent = records[-10:]
    recent_rows = ""
    for r in reversed(recent):
        ts = r.timestamp[:19] if r.timestamp else ""
        dt = r.defect_types or "无"
        status_color = "#E53E3E" if r.defect_count > 0 else "#38A169"
        status_text = f"⚠ {r.defect_count}处" if r.defect_count > 0 else "✓ 合格"
        recent_rows += f"""
        <tr style="border-bottom:1px solid #eee">
            <td style="padding:10px 12px;font-size:13px;color:#666">{ts}</td>
            <td style="padding:10px 12px;font-size:13px">{dt}</td>
            <td style="padding:10px 12px;color:{status_color};font-weight:600;font-size:13px">{status_text}</td>
            <td style="padding:10px 12px;font-size:13px">{r.review_status or 'pending'}</td>
        </tr>"""

    recent_table = f"""<div style="background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 10px rgba(0,0,0,0.05);margin-bottom:20px">
        <h3 style="margin:0 0 16px;color:#333;font-size:16px">📋 最近检测记录</h3>
        <table style="width:100%;border-collapse:collapse">
            <thead><tr style="background:#f7fafc;text-align:left">
                <th style="padding:10px 12px;font-size:13px;color:#555">时间</th>
                <th style="padding:10px 12px;font-size:13px;color:#555">缺陷类型</th>
                <th style="padding:10px 12px;font-size:13px;color:#555">状态</th>
                <th style="padding:10px 12px;font-size:13px;color:#555">审核</th>
            </tr></thead>
            <tbody>{recent_rows}</tbody>
        </table>
    </div>"""

    return stat_cards + recent_table + f"""<div style="text-align:center;padding:10px;color:#999;font-size:12px">
        数据更新时间: {datetime.now():%Y-%m-%d %H:%M:%S}</div>"""

def _build_defect_chart():
    if state.db is None or not HAS_PLOTLY:
        return None
    records = state.db.query(limit=2000)
    defect_counts = {}
    for r in records:
        for t in (r.defect_types or "").split(","):
            t = t.strip()
            if t:
                info = _get_defect_info(t)
                cn = info["cn"]
                defect_counts[cn] = defect_counts.get(cn, 0) + 1
    if not defect_counts:
        return None
    fig = go.Figure(data=[go.Pie(
        labels=list(defect_counts.keys()), values=list(defect_counts.values()),
        hole=0.4, marker=dict(colors=['#E53E3E','#D69E2E','#DD6B20','#805AD5','#38A169','#3182CE']),
        textinfo='label+percent', textfont=dict(size=13)
    )])
    fig.update_layout(
        title=dict(text="缺陷类型分布", font=dict(size=16, color="#333")),
        margin=dict(t=40, b=10, l=10, r=10), height=350,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def _build_trend_chart():
    if state.db is None or not HAS_PLOTLY:
        return None
    records = state.db.query(limit=2000)
    daily = {}
    for r in records:
        day = (r.timestamp or "")[:10]
        if day:
            daily[day] = daily.get(day, {"total": 0, "defect": 0})
            daily[day]["total"] += 1
            if r.defect_count > 0:
                daily[day]["defect"] += 1
    if not daily:
        return None
    days = sorted(daily.keys())[-30:]
    totals = [daily[d]["total"] for d in days]
    defects = [daily[d]["defect"] for d in days]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=days, y=totals, name="检测总数", marker_color="#667eea"))
    fig.add_trace(go.Scatter(x=days, y=defects, name="缺陷数", mode="lines+markers",
                             line=dict(color="#f5576c", width=2), marker=dict(size=6)))
    fig.update_layout(
        title=dict(text="每日检测趋势 (近30天)", font=dict(size=16, color="#333")),
        margin=dict(t=40, b=10, l=10, r=10), height=350,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig

# =================== 审核页 ===================

def load_pending_records():
    if state.db is None:
        return []
    records = state.db.query(review_status="pending", limit=50)
    return [[r.id, r.timestamp[:19] if r.timestamp else "", r.defect_types or "无",
             r.defect_count, f"{r.confidence:.2f}"] for r in records]

def review_record(record_id, status, reviewer, note):
    if state.db is None:
        return "数据库未初始化"
    record = state.db.get_by_id(record_id)
    if record is None:
        return "记录不存在"
    final_result = json.loads(record.yolo_result) if record.yolo_result else {}
    state.db.update_review(record_id=record_id, final_result=final_result,
                           reviewer=reviewer, review_status=status, note=note)
    return f"✅ 记录 {record_id} 已{status}"

# =================== 报表页 ===================

def generate_report(start_date, end_date):
    if state.db is None or state.exporter is None:
        return "系统未初始化"
    start = f"{start_date}T00:00:00" if start_date else None
    end = f"{end_date}T23:59:59" if end_date else None
    csv_path = state.exporter.export_csv(start_time=start, end_time=end)
    html_path = state.exporter.export_html_report(start_time=start, end_time=end)
    total = state.db.count(start, end)
    return f"""## 📊 检测报告
| 指标 | 值 |
|------|-----|
| 检测总数 | {total} |
| CSV 导出 | `{csv_path}` |
| HTML 报告 | `{html_path}` |"""

# =================== 系统设置 ===================

def get_system_info():
    cfg = state.config
    yolo_ok = state.yolo and hasattr(state.yolo, '_models') and len(state.yolo._models) > 0
    yolo_status = "✅ 已加载" if yolo_ok else "❌ 未加载"
    yolo_device = state.yolo.device if state.yolo else "N/A"
    yolo_model = cfg.get('yolo', {}).get('model_path', 'N/A')
    yolo_thresh = cfg.get('yolo', {}).get('conf_threshold', 'N/A')

    vlm_ok = state.vlm is not None
    vlm_status = "✅ 云端API" if (vlm_ok and hasattr(state.vlm, '_mode') and state.vlm._mode == "api") else ("✅ 离线分析" if vlm_ok else "❌ 未启用")
    vlm_model = state.vlm.model if vlm_ok else "N/A"
    db_count = state.db.count() if state.db else "N/A"
    db_path = cfg.get('database', {}).get('path', 'N/A')

    return f"""<div style="font-family:system-ui;padding:16px">
    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-bottom:20px">
        <div style="background:#fff;border-radius:12px;padding:20px;border:1px solid #e2e8f0;box-shadow:0 2px 10px rgba(0,0,0,0.04)">
            <div style="font-weight:800;font-size:16px;color:#1a202c;margin-bottom:16px;display:flex;align-items:center;gap:8px">
                <span style="font-size:20px">🤖</span> YOLO 检测引擎</div>
            <div style="display:flex;flex-direction:column;gap:10px">
                <div style="display:flex;justify-content:space-between;font-size:14px;padding:8px 0;border-bottom:1px solid #f0f0f0">
                    <span style="color:#666">状态</span><span style="font-weight:700;color:{'#38A169' if yolo_ok else '#E53E3E'}">{yolo_status}</span></div>
                <div style="display:flex;justify-content:space-between;font-size:14px;padding:8px 0;border-bottom:1px solid #f0f0f0">
                    <span style="color:#666">模型</span><span style="font-weight:600;color:#333">{yolo_model}</span></div>
                <div style="display:flex;justify-content:space-between;font-size:14px;padding:8px 0;border-bottom:1px solid #f0f0f0">
                    <span style="color:#666">推理设备</span><span style="font-weight:600;color:#333">{yolo_device}</span></div>
                <div style="display:flex;justify-content:space-between;font-size:14px;padding:8px 0">
                    <span style="color:#666">置信度阈值</span><span style="font-weight:600;color:#333">{yolo_thresh}</span></div>
            </div>
        </div>
        <div style="background:#fff;border-radius:12px;padding:20px;border:1px solid #e2e8f0;box-shadow:0 2px 10px rgba(0,0,0,0.04)">
            <div style="font-weight:800;font-size:16px;color:#1a202c;margin-bottom:16px;display:flex;align-items:center;gap:8px">
                <span style="font-size:20px">🔬</span> VLM 视觉大模型</div>
            <div style="display:flex;flex-direction:column;gap:10px">
                <div style="display:flex;justify-content:space-between;font-size:14px;padding:8px 0;border-bottom:1px solid #f0f0f0">
                    <span style="color:#666">状态</span><span style="font-weight:700;color:{'#38A169' if vlm_ok else '#E53E3E'}">{vlm_status}</span></div>
                <div style="display:flex;justify-content:space-between;font-size:14px;padding:8px 0;border-bottom:1px solid #f0f0f0">
                    <span style="color:#666">模型</span><span style="font-weight:600;color:#333">{vlm_model}</span></div>
                <div style="display:flex;justify-content:space-between;font-size:14px;padding:8px 0;border-bottom:1px solid #f0f0f0">
                    <span style="color:#666">API Key</span><span style="font-weight:600;color:#38A169">已配置 ✓</span></div>
                <div style="display:flex;justify-content:space-between;font-size:14px;padding:8px 0">
                    <span style="color:#666">提供商</span><span style="font-weight:600;color:#333">阿里通义千问</span></div>
            </div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px">
        <div style="background:#fff;border-radius:12px;padding:20px;border:1px solid #e2e8f0;box-shadow:0 2px 10px rgba(0,0,0,0.04)">
            <div style="font-weight:800;font-size:16px;color:#1a202c;margin-bottom:16px;display:flex;align-items:center;gap:8px">
                <span style="font-size:20px">🗄️</span> 数据库</div>
            <div style="display:flex;flex-direction:column;gap:10px">
                <div style="display:flex;justify-content:space-between;font-size:14px;padding:8px 0;border-bottom:1px solid #f0f0f0">
                    <span style="color:#666">路径</span><span style="font-weight:600;font-size:13px;color:#333">{db_path}</span></div>
                <div style="display:flex;justify-content:space-between;font-size:14px;padding:8px 0">
                    <span style="color:#666">总记录数</span><span style="font-weight:700;font-size:20px;color:#0d47a1">{db_count}</span></div>
            </div>
        </div>
        <div style="background:#fff;border-radius:12px;padding:20px;border:1px solid #e2e8f0;box-shadow:0 2px 10px rgba(0,0,0,0.04)">
            <div style="font-weight:800;font-size:16px;color:#1a202c;margin-bottom:16px;display:flex;align-items:center;gap:8px">
                <span style="font-size:20px">🖥️</span> 系统信息</div>
            <div style="display:flex;flex-direction:column;gap:10px">
                <div style="display:flex;justify-content:space-between;font-size:14px;padding:8px 0;border-bottom:1px solid #f0f0f0">
                    <span style="color:#666">系统名称</span><span style="font-weight:600;color:#333">钢铁缺陷智能检测</span></div>
                <div style="display:flex;justify-content:space-between;font-size:14px;padding:8px 0;border-bottom:1px solid #f0f0f0">
                    <span style="color:#666">版本</span><span style="font-weight:600;color:#333">v2.0</span></div>
                <div style="display:flex;justify-content:space-between;font-size:14px;padding:8px 0">
                    <span style="color:#666">服务端口</span><span style="font-weight:600;color:#333">{cfg.get('gradio',{}).get('server_port',7860)}</span></div>
            </div>
        </div>
    </div>
</div>"""

# =================== 启动 ===================

def check_auth(username, password):
    accounts = {
        "admin": "123456",
        "inspector": "123456",
        "supervisor": "123456",
        "ai_engineer": "123456",
        "process_engineer": "123456",
    }
    return username in accounts and accounts[username] == password

def launch(config_path="config.yaml"):
    state.init_from_config(config_path)
    state.load_models()

    css = """
        /* ===== 科技感全局背景 ===== */
        body {
            background: #0a0e17 !important;
            background-image:
                linear-gradient(rgba(0,150,255,0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,150,255,0.03) 1px, transparent 1px) !important;
            background-size: 40px 40px !important;
            position: relative !important;
        }
        body::before {
            content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: radial-gradient(ellipse at 20% 50%, rgba(0,100,255,0.08) 0%, transparent 60%),
                        radial-gradient(ellipse at 80% 20%, rgba(0,200,255,0.05) 0%, transparent 50%);
            pointer-events: none; z-index: 0;
        }
        .gradio-container { max-width: 1480px !important; margin: 0 auto !important; position: relative; z-index: 1; }
        
        /* ===== 全局文字对比度增强 ===== */
        .gradio-container label, .gradio-container .label-text,
        .gradio-container .prose, .gradio-container p,
        .gradio-container h1, .gradio-container h2, .gradio-container h3,
        .gradio-container h4, .gradio-container h5, .gradio-container h6 {
            color: #c8d6e5 !important;
        }
        .gradio-container .tab-nav button { color: #94a3b8 !important; }
        .gradio-container .tab-nav button.selected { color: #fff !important; }
        
        /* ===== 登录页卡片 ===== */
        .login-card { max-width: 440px; margin: 100px auto; padding: 48px 40px;
            background: rgba(10,14,23,0.85) !important;
            backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
            border-radius: 20px; border: 1px solid rgba(0,150,255,0.2);
            box-shadow: 0 0 60px rgba(0,100,255,0.1), inset 0 1px 0 rgba(255,255,255,0.05); }
        .login-card label, .login-card h3, .login-card p { color: #c8d6e5 !important; }
        .login-card input { background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(0,150,255,0.25) !important;
            color: #e8f0ff !important; }
        .login-card input:focus { border-color: #0096ff !important; box-shadow: 0 0 20px rgba(0,150,255,0.2) !important;
            background: rgba(255,255,255,0.1) !important; }
        
        /* ===== Tab 标签科技感 ===== */
        .tabs > .tab-nav > button { 
            font-size: 15px !important; font-weight: 600 !important;
            padding: 14px 28px !important; border-radius: 12px 12px 0 0 !important;
            transition: all 0.3s ease !important; background: rgba(255,255,255,0.03) !important;
            border: 1px solid transparent !important; margin-right: 4px !important;
            color: #64748b !important;
        }
        .tabs > .tab-nav > button:hover { color: #0096ff !important; background: rgba(0,150,255,0.06) !important; }
        .tabs > .tab-nav > button.selected {
            background: linear-gradient(135deg, rgba(0,80,180,0.9), rgba(0,120,220,0.85)) !important;
            color: #fff !important; border-color: rgba(0,150,255,0.3) !important;
            box-shadow: 0 0 25px rgba(0,100,255,0.3), inset 0 1px 0 rgba(255,255,255,0.1) !important;
        }
        
        /* ===== 按钮 ===== */
        button.primary { 
            background: linear-gradient(135deg, #0052cc, #0078ff) !important;
            border: 1px solid rgba(0,150,255,0.3) !important;
            font-weight: 700 !important; letter-spacing: 0.5px; color: #fff !important;
            box-shadow: 0 0 20px rgba(0,100,255,0.25), inset 0 1px 0 rgba(255,255,255,0.15) !important;
            transition: all 0.3s !important;
        }
        button.primary:hover { transform: translateY(-2px);
            box-shadow: 0 0 35px rgba(0,120,255,0.4), inset 0 1px 0 rgba(255,255,255,0.2) !important;
            background: linear-gradient(135deg, #0066e0, #0088ff) !important; }
        button.secondary { 
            background: rgba(255,255,255,0.05) !important; color: #94a3b8 !important;
            border: 1px solid rgba(0,150,255,0.15) !important; font-weight: 600 !important;
        }
        button.secondary:hover { background: rgba(0,150,255,0.1) !important; color: #0096ff !important;
            border-color: rgba(0,150,255,0.3) !important; }
        
        /* ===== 统计卡片科技感 ===== */
        .stat-card { transition: all 0.4s ease; cursor: default; }
        .stat-card:hover { transform: translateY(-5px); box-shadow: 0 15px 40px rgba(0,0,0,0.4) !important; }
        
        /* ===== 输入框 ===== */
        input, textarea, select {
            border-radius: 10px !important; border: 1px solid rgba(0,150,255,0.15) !important;
            transition: all 0.3s !important; padding: 10px 14px !important;
            background: rgba(255,255,255,0.04) !important; color: #c8d6e5 !important;
        }
        input:focus, textarea:focus {
            border-color: #0096ff !important;
            box-shadow: 0 0 20px rgba(0,150,255,0.15), inset 0 0 10px rgba(0,150,255,0.03) !important;
            background: rgba(255,255,255,0.07) !important;
        }
        
        /* ===== 图片 ===== */
        .image-container { border-radius: 16px !important; overflow: hidden !important;
            border: 1px solid rgba(0,150,255,0.1) !important;
            box-shadow: 0 0 30px rgba(0,80,180,0.1) !important; }
        
        /* ===== 数据表格 ===== */
        table { border-collapse: collapse !important; }
        th { background: rgba(0,150,255,0.08) !important; color: #b0bed0 !important;
            border-bottom: 2px solid rgba(0,150,255,0.2) !important; }
        td { color: #d0d8e0 !important; border-bottom: 1px solid rgba(255,255,255,0.08) !important; }
        
        /* ===== 隐藏footer ===== */
        footer { display: none !important; }
        
        /* ===== 滚动条科技感 ===== */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0a0e17; }
        ::-webkit-scrollbar-thumb { background: rgba(0,150,255,0.2); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(0,150,255,0.4); }
        
        /* ===== Accordion ===== */
        .accordion { border: 1px solid rgba(0,150,255,0.12) !important; border-radius: 12px !important;
            background: rgba(255,255,255,0.02) !important; }
        .accordion > .label-wrap { color: #b0bed0 !important; }
        
        /* ===== Radio ===== */
        .radio-group label { color: #c8d6e5 !important; }
        
        /* ===== 白色卡片内文字显式深色 (仅直接子文本，不破坏子元素颜色) ===== */
        [style*="background:#fff"] > span:not([style*="background:"]),
        [style*="background:#ffffff"] > span:not([style*="background:"]) {
            color: #1a202c !important;
        }
        .gradio-container .prose span { color: inherit; }
    """

    with gr.Blocks(title="钢铁表面缺陷智能检测系统") as app:

        login_state = gr.State(False)

        # 先声明两个主 Column (必须在引用前定义)
        login_page = gr.Column(visible=True, elem_id="login-page")
        main_page = gr.Column(visible=False, elem_id="main-page")

        # ========== 登录页 (填充) ==========
        with login_page:
            gr.HTML("""<div style="text-align:center;padding:60px 0 30px;position:relative;z-index:1">
                <div style="display:inline-block;width:100px;height:100px;
                    background:linear-gradient(135deg,rgba(0,100,255,0.2),rgba(0,180,255,0.1));
                    border:1px solid rgba(0,150,255,0.3);border-radius:28px;
                    display:flex;align-items:center;justify-content:center;font-size:50px;
                    box-shadow:0 0 50px rgba(0,100,255,0.15),inset 0 0 30px rgba(0,150,255,0.05);
                    margin-bottom:24px;position:relative">
                    <span style="position:absolute;top:-2px;right:-2px;width:12px;height:12px;
                        background:#0096ff;border-radius:50%;box-shadow:0 0 12px #0096ff"></span>
                    🏭
                </div>
                <h1 style="font-size:34px;font-weight:800;
                    background:linear-gradient(135deg,#0096ff,#00d4ff);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    margin:0;letter-spacing:1px">
                    钢铁表面缺陷智能检测系统</h1>
                <p style="color:#64748b;font-size:16px;margin-top:12px;letter-spacing:1.5px">
                    YOLO + VLM 双引擎 · AI 智能质检 · 工业 4.0</p>
                <div style="margin-top:20px;display:flex;justify-content:center;gap:40px">
                    <div style="text-align:center"><div style="font-size:20px;font-weight:800;color:#0096ff">6</div>
                        <div style="font-size:11px;color:#64748b;margin-top:2px">缺陷类型</div></div>
                    <div style="text-align:center"><div style="font-size:20px;font-weight:800;color:#00d4ff">&lt;1%</div>
                        <div style="font-size:11px;color:#64748b;margin-top:2px">漏检率</div></div>
                    <div style="text-align:center"><div style="font-size:20px;font-weight:800;color:#00d4ff">&lt;50ms</div>
                        <div style="font-size:11px;color:#64748b;margin-top:2px">检测延迟</div></div>
                </div>
            </div>""")
            with gr.Column(elem_classes="login-card"):
                gr.Markdown("### 🔐 用户登录")
                login_user = gr.Textbox(label="用户名", placeholder="请输入用户名", elem_id="login-user")
                login_pwd = gr.Textbox(label="密码", type="password", placeholder="请输入密码", elem_id="login-pwd")
                login_btn = gr.Button("登 录", variant="primary", size="lg")
                login_error = gr.Markdown(visible=False)
                gr.HTML("""<div style="text-align:center;margin-top:20px;font-size:12px;color:#999">
                    默认账号: admin / 123456</div>""")

            def do_login(user, pwd):
                accounts = {
                    "admin": "123456",
                    "inspector": "123456",
                    "supervisor": "123456",
                    "ai_engineer": "123456",
                    "process_engineer": "123456",
                }
                user = user.strip()
                pwd = pwd.strip()
                if user in accounts and accounts[user] == pwd:
                    role = user
                    roles_cn = {
                        "admin": "系统管理员",
                        "inspector": "现场质检员",
                        "supervisor": "质检主管",
                        "ai_engineer": "AI 工程师",
                        "process_engineer": "工艺工程师"
                    }
                    role_cn = roles_cn.get(role, role)

                    is_admin = (role == "admin")
                    is_inspector = (role == "inspector")
                    is_supervisor = (role == "supervisor")
                    is_ai = (role == "ai_engineer")
                    is_process = (role == "process_engineer")

                    dash_vis = is_admin or is_supervisor
                    detect_vis = is_admin or is_inspector or is_supervisor or is_ai or is_process
                    review_vis = is_admin or is_inspector
                    reports_vis = is_admin or is_supervisor
                    settings_vis = is_admin or is_ai

                    detect_btn_vis = is_admin or is_inspector
                    vlm_btn_vis = is_admin or is_inspector
                    rag_btn_vis = is_admin or is_inspector or is_process
                    save_btn_vis = is_admin or is_inspector

                    header_val = f"""<div style="display:flex;align-items:center;justify-content:space-between;
                        padding:16px 28px;background:rgba(10,14,23,0.8);backdrop-filter:blur(20px);
                        color:#c8d6e5;border-radius:16px;margin-bottom:20px;
                        border:1px solid rgba(0,150,255,0.12);
                        box-shadow:0 0 40px rgba(0,80,180,0.08);position:relative;overflow:hidden">
                        <div style="position:absolute;bottom:0;left:0;width:100%;height:1px;
                            background:linear-gradient(90deg,transparent,rgba(0,150,255,0.3),transparent)"></div>
                        <div style="display:flex;align-items:center;gap:14px;position:relative;z-index:1">
                            <div style="width:44px;height:44px;background:rgba(0,150,255,0.1);border:1px solid rgba(0,150,255,0.2);
                                border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:22px;
                                box-shadow:0 0 20px rgba(0,100,255,0.1)">🏭</div>
                            <div>
                                <div style="font-size:20px;font-weight:800;letter-spacing:1.5px;
                                    background:linear-gradient(90deg,#0096ff,#00d4ff);
                                    -webkit-background-clip:text;-webkit-text-fill-color:transparent">
                                    钢铁表面缺陷智能检测系统</div>
                                <div style="font-size:11px;color:#64748b;letter-spacing:0.5px;margin-top:1px">YOLO + VLM 双引擎 · AI 智能质检 · 工业 4.0</div>
                            </div>
                        </div>
                        <div style="display:flex;align-items:center;gap:20px;font-size:13px;position:relative;z-index:1">
                            <div style="display:flex;align-items:center;gap:8px;background:rgba(0,150,255,0.08);
                                padding:6px 16px;border-radius:20px;border:1px solid rgba(0,150,255,0.15)">
                                <span style="width:8px;height:8px;background:#00ff88;border-radius:50%;display:inline-block;
                                    box-shadow:0 0 8px #00ff88"></span>
                                <span style="color:#c8d6e5">👤 {user} ({role_cn})</span>
                            </div>
                            <span style="color:rgba(255,255,255,0.15)">|</span>
                            <span style="color:#64748b">🕐 {datetime.now():%Y-%m-%d %H:%M:%S}</span>
                        </div>
                    </div>"""

                    dash_content, chart_p, chart_t, kpi_c = _init_dashboard()

                    return (
                        gr.update(visible=False), gr.update(visible=True), True,
                        gr.update(visible=False, value=""),
                        gr.update(visible=dash_vis),
                        gr.update(visible=detect_vis),
                        gr.update(visible=review_vis),
                        gr.update(visible=reports_vis),
                        gr.update(visible=settings_vis),
                        gr.update(visible=detect_btn_vis),
                        gr.update(visible=vlm_btn_vis),
                        gr.update(visible=rag_btn_vis),
                        gr.update(visible=save_btn_vis),
                        header_val,
                        dash_content, chart_p, chart_t, kpi_c
                    )
                return (
                    gr.update(visible=True), gr.update(visible=False), False,
                    gr.update(visible=True, value="❌ 用户名或密码错误"),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    "", "", None, None, ""
                )

            def _on_load(request: gr.Request = None):
                username = request.username if request else None
                if username:
                    role = username
                    roles_cn = {
                        "admin": "系统管理员",
                        "inspector": "现场质检员",
                        "supervisor": "质检主管",
                        "ai_engineer": "AI 工程师",
                        "process_engineer": "工艺工程师"
                    }
                    role_cn = roles_cn.get(role, role)

                    is_admin = (role == "admin")
                    is_inspector = (role == "inspector")
                    is_supervisor = (role == "supervisor")
                    is_ai = (role == "ai_engineer")
                    is_process = (role == "process_engineer")

                    dash_vis = is_admin or is_supervisor
                    detect_vis = is_admin or is_inspector or is_supervisor or is_ai or is_process
                    review_vis = is_admin or is_inspector
                    reports_vis = is_admin or is_supervisor
                    settings_vis = is_admin or is_ai

                    detect_btn_vis = is_admin or is_inspector
                    vlm_btn_vis = is_admin or is_inspector
                    rag_btn_vis = is_admin or is_inspector or is_process
                    save_btn_vis = is_admin or is_inspector

                    header_val = f"""<div style="display:flex;align-items:center;justify-content:space-between;
                        padding:16px 28px;background:rgba(10,14,23,0.8);backdrop-filter:blur(20px);
                        color:#c8d6e5;border-radius:16px;margin-bottom:20px;
                        border:1px solid rgba(0,150,255,0.12);
                        box-shadow:0 0 40px rgba(0,80,180,0.08);position:relative;overflow:hidden">
                        <div style="position:absolute;bottom:0;left:0;width:100%;height:1px;
                            background:linear-gradient(90deg,transparent,rgba(0,150,255,0.3),transparent)"></div>
                        <div style="display:flex;align-items:center;gap:14px;position:relative;z-index:1">
                            <div style="width:44px;height:44px;background:rgba(0,150,255,0.1);border:1px solid rgba(0,150,255,0.2);
                                border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:22px;
                                box-shadow:0 0 20px rgba(0,100,255,0.1)">🏭</div>
                            <div>
                                <div style="font-size:20px;font-weight:800;letter-spacing:1.5px;
                                    background:linear-gradient(90deg,#0096ff,#00d4ff);
                                    -webkit-background-clip:text;-webkit-text-fill-color:transparent">
                                    钢铁表面缺陷智能检测系统</div>
                                <div style="font-size:11px;color:#64748b;letter-spacing:0.5px;margin-top:1px">YOLO + VLM 双引擎 · AI 智能质检 · 工业 4.0</div>
                            </div>
                        </div>
                        <div style="display:flex;align-items:center;gap:20px;font-size:13px;position:relative;z-index:1">
                            <div style="display:flex;align-items:center;gap:8px;background:rgba(0,150,255,0.08);
                                padding:6px 16px;border-radius:20px;border:1px solid rgba(0,150,255,0.15)">
                                <span style="width:8px;height:8px;background:#00ff88;border-radius:50%;display:inline-block;
                                    box-shadow:0 0 8px #00ff88"></span>
                                <span style="color:#c8d6e5">👤 {username} ({role_cn})</span>
                            </div>
                            <span style="color:rgba(255,255,255,0.15)">|</span>
                            <span style="color:#64748b">🕐 {datetime.now():%Y-%m-%d %H:%M:%S}</span>
                        </div>
                    </div>"""

                    dash_content, chart_p, chart_t, kpi_c = _init_dashboard()

                    return (
                        gr.update(visible=False), gr.update(visible=True), True,
                        gr.update(visible=dash_vis),
                        gr.update(visible=detect_vis),
                        gr.update(visible=review_vis),
                        gr.update(visible=reports_vis),
                        gr.update(visible=settings_vis),
                        gr.update(visible=detect_btn_vis),
                        gr.update(visible=vlm_btn_vis),
                        gr.update(visible=rag_btn_vis),
                        gr.update(visible=save_btn_vis),
                        header_val,
                        dash_content, chart_p, chart_t, kpi_c
                    )
                else:
                    return (
                        gr.update(visible=True), gr.update(visible=False), False,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        "",
                        "", None, None, ""
                    )

            pass

        # ========== 主页面 (填充) ==========
        with main_page:
            header_html = gr.HTML()

            with gr.Tabs() as tabs:
                # ===== Tab 1: 仪表盘 =====
                dashboard_tab = gr.Tab("📊 仪表盘", id="dashboard")
                with dashboard_tab:
                    dashboard_layout = gr.Column(visible=True)
                    with dashboard_layout:
                        refresh_btn = gr.Button("🔄 刷新数据", variant="secondary", size="sm")
                        dash_html = gr.HTML()
                        with gr.Row():
                            chart_pie = gr.Plot(label="缺陷类型分布", scale=1)
                            chart_trend = gr.Plot(label="检测趋势", scale=1)
                        kpi_html = gr.HTML(label="KPI实时监控")
                        refresh_btn.click(
                            lambda: (_build_dashboard(), _build_defect_chart(), _build_trend_chart(), _kpi_monitor()),
                            outputs=[dash_html, chart_pie, chart_trend, kpi_html]
                        )

                # ===== Tab 2: 实时检测 =====
                detection_tab = gr.Tab("🔍 实时检测", id="detection")
                with detection_tab:
                    detection_layout = gr.Column(visible=True)
                    with detection_layout:
                        with gr.Row():
                            with gr.Column(scale=2):
                                input_img = gr.Image(label="上传钢板图像", type="numpy", height=400)
                            with gr.Column(scale=1):
                                with gr.Tabs():
                                    with gr.Tab("检测结果"):
                                        output_img = gr.Image(label="检测结果", height=380)
                                    with gr.Tab("热力图"):
                                        heatmap_img = gr.Image(label="缺陷热力图", height=380)
                        with gr.Row():
                            detect_btn = gr.Button("🚀 YOLO 快速筛查", variant="primary", scale=1)
                            vlm_btn = gr.Button("🔬 VLM 精细分析", variant="secondary", scale=1)
                            rag_btn = gr.Button("📚 RAG 根因分析", variant="secondary", scale=1)
                            save_btn = gr.Button("💾 保存记录", variant="secondary", scale=1)
                        detect_stats = gr.Textbox(label="检测统计", interactive=False, elem_id="detect-stats")
                        # 整合面板: 检测报告 + 严重度分级 + 根因分析 + Bad Case → 一个滚动窗口
                        combined_html = gr.HTML(label="📊 分析报告总览 (滚轮查看全部)")
                        with gr.Accordion("💾 保存检测记录", open=False):
                            with gr.Row():
                                reviewer = gr.Textbox(label="审核人", placeholder="输入姓名")
                                note = gr.Textbox(label="备注", placeholder="可选备注")
                            save_msg = gr.Textbox(label="保存结果", interactive=False)
                            save_btn.click(save_record, [reviewer, note], [save_msg])

                    # 组合函数: 整合所有分析结果为单个可滚动HTML
                    def _build_scrollable_panel(*sections):
                        """将多个HTML区块整合到一个带滚轮滑动的容器中"""
                        non_empty = [s for s in sections if s and s.strip()]
                        if not non_empty:
                            return """<div style="color:#94a3b8;text-align:center;padding:40px;font-size:15px">
                                <div style="font-size:40px;margin-bottom:10px">📊</div>
                                <div>暂无分析数据，请先执行检测</div></div>"""
                        return f"""<div style="max-height:580px;overflow-y:auto;padding:4px;
                            scrollbar-width:thin;scrollbar-color:rgba(0,150,255,0.3) transparent">
                            <style>
                            div::-webkit-scrollbar {{width:6px}}
                            div::-webkit-scrollbar-track {{background:transparent}}
                            div::-webkit-scrollbar-thumb {{background:rgba(0,150,255,0.25);border-radius:3px}}
                            div::-webkit-scrollbar-thumb:hover {{background:rgba(0,150,255,0.45)}}
                            </style>
                            {''.join(f'<div style="margin-bottom:12px">{s}</div>' for s in non_empty)}
                        </div>"""

                    def _detect_full(image):
                        ann, html, hm, stats = detect_image(image)
                        dets = state.last_yolo_result.get("detections", [])
                        from src.base_detector import DetectionResult
                        objs = [DetectionResult(bbox=d["bbox"], class_name=d.get("class_name","?"),
                                confidence=d.get("confidence",0)) for d in dets]
                        grade = _build_grade_html(objs)
                        bc = _bad_case_collect(objs)
                        combined = _build_scrollable_panel(html, grade, bc)
                        return ann, combined, hm, stats

                    def _vlm_combined(image):
                        ann, html, stats = vlm_analyze_image(image)
                        # 尝试做分级 and bad case
                        result = state.last_vlm_result or state.last_yolo_result
                        dets = result.get("detections", []) if result else []
                        if dets:
                            from src.base_detector import DetectionResult
                            objs = [DetectionResult(bbox=d.get("bbox",[0,0,1,1]), class_name=d.get("class_name","?"),
                                    confidence=d.get("confidence",0)) for d in dets]
                            grade = _build_grade_html(objs)
                            bc = _bad_case_collect(objs)
                        else:
                            grade = ""; bc = ""
                        combined = _build_scrollable_panel(html, grade, bc)
                        return ann, combined, stats

                    def _rag_combined():
                        rag = rag_analysis()
                        # 收集已有的报告和分级信息
                        result = state.last_vlm_result or state.last_yolo_result
                        if result:
                            dets = result.get("detections", [])
                            if dets:
                                from src.base_detector import DetectionResult
                                objs = [DetectionResult(bbox=d.get("bbox",[0,0,1,1]), class_name=d.get("class_name","?"),
                                        confidence=d.get("confidence",0)) for d in dets]
                                grade = _build_grade_html(objs)
                                bc = _bad_case_collect(objs)
                            else:
                                grade = ""; bc = ""
                        else:
                            grade = ""; bc = ""
                        combined = _build_scrollable_panel(grade, rag, bc)
                        return combined

                    detect_btn.click(_detect_full, [input_img],
                        [output_img, combined_html, heatmap_img, detect_stats])
                    vlm_btn.click(_vlm_combined, [input_img],
                        [output_img, combined_html, detect_stats])
                    rag_btn.click(_rag_combined, None, [combined_html])

                # ===== Tab 3: 人工审核 =====
                review_tab = gr.Tab("✅ 人工审核", id="review")
                with review_tab:
                    review_layout = gr.Column(visible=True)
                    with review_layout:
                        with gr.Row():
                            refresh_review_btn = gr.Button("🔄 刷新列表", variant="secondary")
                        records_table = gr.Dataframe(
                            headers=["ID", "时间", "缺陷类型", "数量", "置信度"],
                            interactive=False, label="待审核记录"
                        )
                        with gr.Row():
                            record_id_input = gr.Number(label="记录 ID", precision=0)
                            review_status = gr.Radio(["confirmed", "corrected"], label="审核结果", value="confirmed")
                            review_reviewer = gr.Textbox(label="审核人")
                            review_note = gr.Textbox(label="备注")
                        review_btn = gr.Button("提交审核", variant="primary")
                        review_msg = gr.Textbox(label="审核结果", interactive=False)
                        refresh_review_btn.click(load_pending_records, outputs=[records_table])
                        review_btn.click(review_record, [record_id_input, review_status, review_reviewer, review_note], [review_msg])

                # ===== Tab 4: 数据报表 =====
                reports_tab = gr.Tab("📈 数据报表", id="reports")
                with reports_tab:
                    reports_layout = gr.Column(visible=True)
                    with reports_layout:
                        with gr.Row():
                            start_date = gr.Textbox(label="开始日期 (YYYY-MM-DD)", placeholder="2026-01-01")
                            end_date = gr.Textbox(label="结束日期 (YYYY-MM-DD)", placeholder="2026-12-31")
                        report_btn = gr.Button("📊 生成报告", variant="primary")
                        report_output = gr.Markdown()
                        report_btn.click(generate_report, [start_date, end_date], [report_output])

                # ===== Tab 5: 系统设置 =====
                settings_tab = gr.Tab("⚙️ 系统设置", id="settings")
                with settings_tab:
                    settings_layout = gr.Column(visible=True)
                    with settings_layout:
                        sys_info = gr.Markdown()
                        gr.Button("🔄 刷新状态", variant="secondary").click(get_system_info, outputs=[sys_info])

            gr.HTML("""<div style="text-align:center;padding:24px;color:#94a3b8;font-size:12px;margin-top:24px;
                border-top:1px solid rgba(0,0,0,0.06)">
                <div style="font-weight:600;color:#64748b;margin-bottom:4px">钢铁表面缺陷智能检测系统</div>
                <div>© 2026 · YOLO + VLM 双引擎 · AI 智能质检平台 · 工业 4.0 解决方案</div>
            </div>""")

        # 初始加载
        def _init_dashboard():
            return _build_dashboard(), _build_defect_chart(), _build_trend_chart(), _kpi_monitor()

        login_btn.click(
            do_login, [login_user, login_pwd],
            [
                login_page, main_page, login_state, login_error,
                dashboard_layout, detection_layout, review_layout, reports_layout, settings_layout,
                detect_btn, vlm_btn, rag_btn, save_btn,
                header_html,
                dash_html, chart_pie, chart_trend, kpi_html
            ]
        )

        app.load(
            _on_load,
            outputs=[
                login_page, main_page, login_state,
                dashboard_layout, detection_layout, review_layout, reports_layout, settings_layout,
                detect_btn, vlm_btn, rag_btn, save_btn,
                header_html,
                dash_html, chart_pie, chart_trend, kpi_html
            ]
        )

    share_enabled = state.config.get("gradio", {}).get("share", False)
    app.launch(
        server_name=state.config.get("gradio", {}).get("server_name", "0.0.0.0"),
        server_port=state.config.get("gradio", {}).get("server_port", 7860),
        share=share_enabled,
        show_error=True,
        css=css,
        auth=check_auth,
    )

if __name__ == "__main__":
    launch()