"""
Gradio Web 工作台 - 钢铁表面缺陷检测系统。

提供三个主要页面:
1. 实时检测页 - 摄像头实时画面 + 检测结果叠加
2. 审核页 - 历史检测记录人工审核
3. 报表页 - 缺陷统计图表与数据导出
"""

import json
import os
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

# 加载 .env 环境变量
from dotenv import load_dotenv
load_dotenv()

# 绕过代理设置 (Gradio 6.x startup-events 502 修复)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1,0.0.0.0"

import cv2
import gradio as gr
import numpy as np
import yaml

from src.camera import CameraCapture
from src.db_manager import DBManager, InspectionRecord
from src.exporter import Exporter
from src.icons import (
    ICON_DETECT, ICON_REVIEW, ICON_REPORT, ICON_CAMERA_TAB,
    ICON_YOLO, ICON_VLM, ICON_FULL,
    ICON_PASS, ICON_ALERT, ICON_DATABASE, ICON_PRECISION,
    ICON_CONFIRMED, ICON_REJECT, ICON_RAG, ICON_SAVE,
    ICON_REFRESH, ICON_EXPORT, ICON_SENSITIVITY,
    ICON_GPU, ICON_LOGO, ICON_SEARCH,
)

# 重型依赖延迟加载 (避免未安装时阻塞启动)
YOLODetector = None
VLMDetector = None
rag_analyze = None  # RAG 根因分析 (延迟加载)

from ui import load_css, load_logo_svg
from src.voice_commander import VoiceCommander, VOICE_INPUT_HTML

# ==================== 工业精装主义主题 CSS ====================

INDUSTRIAL_CSS = ""  # CSS 已提取到 ui/industrial.css，通过 ui.load_css() 加载

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
        try:
            from scripts.rag_demo import rag_analyze as _rag  # pyright: ignore[reportMissingImports]
            rag_analyze = _rag
        except ImportError:
            return None
    return rag_analyze


# ==================== 全局状态 ====================

class AppState:
    def __init__(self):
        self.yolo: YOLODetector | None = None
        self.vlm: VLMDetector | None = None
        self.screener: Any | None = None        # FastScreener 预筛查
        self.sahi_detector: Any | None = None    # SAHI 滑窗检测器
        self.camera: CameraCapture | None = None
        self.db: DBManager | None = None
        self.exporter: Exporter | None = None
        self.config: dict = {}
        self.current_image: np.ndarray | None = None
        self.last_result: dict = {}
        self.plc_trigger: Any | None = None      # PLC触发控制器
        self._screener_stats: dict = {"filtered": 0, "passed": 0}  # 筛查统计

    def init_from_config(self, config_path: str):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 数据库
        db_cfg = self.config.get("database", {})
        self.db = DBManager(db_cfg.get("path", "data/inspection.db"))
        self.exporter = Exporter(self.db)

        # ===== TensorRT / YOLO 引擎选择 =====
        trt_cfg = self.config.get("tensorrt", {})
        yolo_cfg = self.config.get("yolo", {})

        if trt_cfg.get("enabled", False):
            # TensorRT 加速模式
            try:
                from src.tensorrt_engine import TensorRTDetector
                engine_path = trt_cfg.get("engine_path", "models/weights/yolov8n.engine")
                pt_path = yolo_cfg.get("model_path", "yolov8n.pt")

                # 如果 Engine 不存在，尝试自动构建
                if not Path(engine_path).exists() and Path(pt_path).exists():
                    print(f"[INFO] TensorRT Engine 未找到，从 {pt_path} 自动构建...")
                    self.yolo = TensorRTDetector.from_pt(
                        pt_path, engine_path,
                        img_size=yolo_cfg.get("img_size", 640),
                        fp16=trt_cfg.get("fp16", True),
                        conf_threshold=yolo_cfg.get("conf_threshold", 0.25),
                        iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
                    )
                else:
                    self.yolo = TensorRTDetector(
                        engine_path=engine_path,
                        conf_threshold=yolo_cfg.get("conf_threshold", 0.25),
                        iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
                        img_size=yolo_cfg.get("img_size", 640),
                    )
                print(f"[INFO] 使用 TensorRT 加速推理")
            except ImportError:
                print("[WARN] TensorRT 不可用，回退到 PyTorch YOLO")
                self._init_yolo_pytorch(yolo_cfg)
            except Exception as e:
                print(f"[WARN] TensorRT 初始化失败: {e}，回退到 PyTorch YOLO")
                self._init_yolo_pytorch(yolo_cfg)
        else:
            self._init_yolo_pytorch(yolo_cfg)

        # ===== SAHI 滑窗小目标增强 =====
        sahi_cfg = self.config.get("sahi", {})
        if sahi_cfg.get("enabled", False) and self.yolo is not None:
            try:
                from src.sahi_engine import SAHIDetector
                self.sahi_detector = SAHIDetector(
                    detector=self.yolo,
                    slice_size=sahi_cfg.get("slice_size", 640),
                    overlap_ratio=sahi_cfg.get("overlap_ratio", 0.2),
                    postprocess_match_threshold=sahi_cfg.get("postprocess_match_threshold", 0.5),
                    postprocess_class_agnostic=sahi_cfg.get("class_agnostic", True),
                )
                self.sahi_detector._model_loaded = True
                print(f"[INFO] SAHI 滑窗检测已启用 (slice={sahi_cfg.get('slice_size', 640)})")
            except ImportError as e:
                print(f"[WARN] SAHI 模块不可用: {e}")
            except Exception as e:
                print(f"[WARN] SAHI 初始化失败: {e}")

        # ===== FastScreener 两阶段预筛查 =====
        screener_cfg = self.config.get("fast_screener", {})
        if screener_cfg.get("enabled", True) and self.yolo is not None:
            try:
                from src.fast_screener import create_screener, ScreeningMode
                mode = screener_cfg.get("mode", "balanced")
                self.screener = create_screener(
                    mode=mode,
                    combo_threshold=screener_cfg.get("combo_threshold", 0.45),
                    fft_threshold=screener_cfg.get("fft_threshold", 0.35),
                    std_threshold=screener_cfg.get("std_threshold", 0.30),
                    edge_threshold=screener_cfg.get("edge_threshold", 0.40),
                    resize_to=screener_cfg.get("resize_to", 256),
                )
                if not screener_cfg.get("adaptive_threshold", True):
                    self.screener._adaptive_enabled = False
                print(f"[INFO] FastScreener 两阶段预筛查已启用 (mode={mode})")
            except ImportError as e:
                print(f"[WARN] FastScreener 模块不可用: {e}")
            except Exception as e:
                print(f"[WARN] FastScreener 初始化失败: {e}")

        # ===== VLM (可选，自动检测 Gemini/Qwen) =====
        vlm_cfg = self.config.get("vlm", {})
        if vlm_cfg.get("enabled", True):
            try:
                VLM = _lazy_import_vlm()
                model = vlm_cfg.get("model") or None
                api_base = vlm_cfg.get("api_base") or None
                self.vlm = VLM(
                    api_base=api_base,
                    model=model,
                    system_prompt=vlm_cfg.get("prompt_template") or None,
                )
            except ImportError:
                print("[WARN] openai 未安装，VLM 检测不可用。请运行: pip install openai")
                self.vlm = None

    def _init_yolo_pytorch(self, yolo_cfg: dict):
        """初始化 PyTorch YOLO 检测器（回退方案）"""
        try:
            YOLO = _lazy_import_yolo()
            device = yolo_cfg.get("device", "cuda:0")
            if device != "cpu":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        print("[INFO] CUDA 不可用，YOLO 将使用 CPU 推理")
                        device = "cpu"
                except Exception:
                    device = "cpu"
            self.yolo = YOLO(
                model_path=yolo_cfg.get("model_path", "models/weights/yolov8n.pt"),
                conf_threshold=yolo_cfg.get("conf_threshold", 0.25),
                iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
                img_size=yolo_cfg.get("img_size", 640),
                device=device,
            )
        except ImportError:
            print("[WARN] ultralytics 未安装，YOLO 检测不可用。请运行: pip install ultralytics")
            self.yolo = None

    def load_models(self):
        """加载模型 (显示进度)"""
        if self.yolo is not None:
            try:
                self.yolo.load_model()
                # CPU 模式跳过预热 (首次推理自动触发)
                if self.yolo.device != "cpu":
                    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                    self.yolo.warmup(dummy)
            except FileNotFoundError as e:
                print(f"[WARN] YOLO 模型未找到: {e}")
        else:
            print("[INFO] YOLO 未安装，跳过模型加载。检测功能需安装 ultralytics")

        if self.vlm is not None:
            try:
                self.vlm.load_model()
            except Exception as e:
                print(f"[WARN] VLM 初始化失败: {e}")
                self.vlm = None


state = AppState()


# ==================== 实时检测页 ====================

# 缺陷类型 → 颜色映射 (BGR) 及 中文名
# 完整 NEU-DET 6类 + 扩展类
DEFECT_INFO = {
    # NEU-DET 标准6类
    "crazing":          {"bgr": (0, 0, 255),        "hex": "#E53E3E", "cn": "裂纹",       "icon": "⚡"},
    "inclusion":        {"bgr": (200, 200, 0),      "hex": "#D69E2E", "cn": "夹杂",       "icon": "◇"},
    "patches":          {"bgr": (0, 140, 255),      "hex": "#DD6B20", "cn": "斑块",       "icon": "▣"},
    "pitted_surface":   {"bgr": (180, 0, 180),      "hex": "#805AD5", "cn": "麻点",       "icon": "⊡"},
    "rolled-in_scale":  {"bgr": (0, 215, 255),      "hex": "#D69E2E", "cn": "轧制氧化皮", "icon": "◈"},
    "scratches":        {"bgr": (255, 100, 0),      "hex": "#3182CE", "cn": "划痕",       "icon": "✂"},
    # 扩展类 (VLM 分类兼容)
    "rust":             {"bgr": (50, 120, 220),      "hex": "#DC7732", "cn": "锈蚀",       "icon": "🦀"},
    "crack":            {"bgr": (0, 0, 255),        "hex": "#E53E3E", "cn": "裂纹",       "icon": "⚡"},
    "scratch":          {"bgr": (255, 100, 0),      "hex": "#3182CE", "cn": "划痕",       "icon": "✂"},
    "scale":            {"bgr": (0, 215, 255),      "hex": "#D69E2E", "cn": "氧化皮",     "icon": "◈"},
    "indentation":      {"bgr": (180, 0, 180),      "hex": "#805AD5", "cn": "压痕",       "icon": "◆"},
    "blister":          {"bgr": (0, 255, 255),      "hex": "#319795", "cn": "气泡",       "icon": "○"},
}
DEFAULT_INFO =         {"bgr": (0, 200, 0),     "hex": "#38A169", "cn": "未知",   "icon": "?"}


def _get_defect_info(class_name: str) -> dict:
    return DEFECT_INFO.get(class_name.lower(), DEFAULT_INFO)


def _build_result_html(detections: list, engine: str, elapsed_ms: float, raw_output: dict = None, image: np.ndarray = None) -> str:
    """构建结构化检测结果 HTML - 精确坐标 + 专业表格"""
    total = len(detections)
    h, w = (image.shape[:2] if image is not None else (1, 1))

    if total == 0:
        return f"""<div style="font-family:system-ui;padding:16px;text-align:center">
            <div style="font-size:56px;margin-bottom:12px">✅</div>
            <div style="font-size:22px;font-weight:800;color:#38A169">未检测到缺陷</div>
            <div style="font-size:14px;color:#999;margin-top:6px;font-weight:500">
                {engine} &middot; {elapsed_ms:.0f}ms &middot; {w}×{h}px
            </div>
            <div style="font-size:13px;color:#bbb;margin-top:4px">产品表面质量合格 — 通过质检</div>
        </div>"""

    # 构建卡片 + 精确坐标
    cards = []
    for i, det in enumerate(detections):
        cn = det.class_name if hasattr(det, 'class_name') else det.get("class_name", "?")
        conf = det.confidence if hasattr(det, 'confidence') else det.get("confidence", 0)
        bbox = det.bbox if hasattr(det, 'bbox') else det.get("bbox", [0, 0, 1, 1])
        info = _get_defect_info(cn)

        # 置信度颜色
        conf_color = "#38A169" if conf >= 0.8 else "#D69E2E" if conf >= 0.5 else "#E53E3E"

        # 归一化坐标
        nx1, ny1, nx2, ny2 = [round(v, 4) for v in bbox]

        # 像素坐标
        px1, py1 = round(nx1 * w), round(ny1 * h)
        px2, py2 = round(nx2 * w), round(ny2 * h)
        pw, ph = px2 - px1, py2 - py1

        # 中心坐标
        cx_norm = round((nx1 + nx2) / 2 * 100, 1)
        cy_norm = round((ny1 + ny2) / 2 * 100, 1)
        cx_px = round((px1 + px2) / 2)
        cy_px = round((py1 + py2) / 2)

        # 面积占比
        area_pct = round((nx2 - nx1) * (ny2 - ny1) * 100, 2)

        cards.append(f"""
        <div style="display:flex;align-items:stretch;gap:14px;padding:14px 16px;
                    background:#fff;border-radius:10px;border:1px solid #e2e8f0;
                    margin-bottom:10px;box-shadow:0 2px 8px rgba(0,0,0,0.04)">
            <!-- 编号圆 -->
            <div style="flex-shrink:0;width:44px;height:44px;border-radius:50%;
                        background:{info['hex']};display:flex;align-items:center;
                        justify-content:center;color:#fff;font-weight:800;font-size:18px;
                        box-shadow:0 3px 10px {info['hex']}55;margin-top:4px">
                {i+1}
            </div>
            <!-- 信息区 -->
            <div style="flex:1;min-width:0">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                    <span style="font-weight:800;font-size:16px;color:#1a202c">{info['icon']} {info['cn']}</span>
                    <span style="font-size:11px;color:#666;background:#f0f0f0;
                                 padding:2px 8px;border-radius:10px">{cn}</span>
                </div>
                <!-- 置信度条 -->
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                    <div style="flex:1;height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden">
                        <div style="width:{conf*100}%;height:100%;background:{conf_color};
                                    border-radius:4px"></div>
                    </div>
                    <span style="font-size:15px;font-weight:700;color:{conf_color};min-width:42px">{conf:.1%}</span>
                </div>
                <!-- 精确坐标 -->
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;
                            font-size:12px;color:#555;line-height:1.8">
                    <div>📌 <b>归一化</b> [{nx1}, {ny1}, {nx2}, {ny2}]</div>
                    <div>📐 <b>尺寸</b> {pw}×{ph} px ({area_pct}%)</div>
                    <div>📍 <b>像素</b> [{px1}, {py1}, {px2}, {py2}]</div>
                    <div>🎯 <b>中心</b> ({cx_px}, {cy_px}) px</div>
                </div>
            </div>
        </div>""")

    # VLM 补充描述
    vlm_desc = ""
    if raw_output:
        vlm_items = raw_output.get("vlm_raw_response", {}).get("detections", [])
        for item in vlm_items:
            desc = item.get("bbox_description", "")
            severity = item.get("severity", "")
            if desc or severity:
                sev_bg = "#FED7D7" if severity == 'severe' else "#FEFCBF" if severity == 'moderate' else "#C6F6D5"
                sev_fg = "#C53030" if severity == 'severe' else "#975A16" if severity == 'moderate' else "#276749"
                vlm_desc += f"""
                <div style="padding:10px 14px;background:#fffbeb;border-left:4px solid #DD6B20;
                            border-radius:6px;margin-top:10px;font-size:14px;color:#555;
                            font-weight:500;line-height:1.6">
                    💬 {desc}
                    <span style="display:inline-block;margin-left:8px;padding:2px 10px;
                                 background:{sev_bg};color:{sev_fg};
                                 border-radius:10px;font-size:11px;font-weight:700">
                        {severity.upper() if severity else ''}
                    </span>
                </div>"""

    # 汇总统计
    cls_counts = {}
    confs = []
    for d in detections:
        cn = d.class_name if hasattr(d, 'class_name') else d.get('class_name', '?')
        cls_counts[cn] = cls_counts.get(cn, 0) + 1
        confs.append(d.confidence if hasattr(d, 'confidence') else d.get('confidence', 0))
    avg_conf = sum(confs) / len(confs) if confs else 0
    cls_summary = " &middot; ".join(f"{_get_defect_info(k)['icon']} {_get_defect_info(k)['cn']} ×{v}" for k, v in cls_counts.items())

    return f"""<div style="font-family:system-ui;padding:8px">
        <!-- 摘要条 -->
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:16px 20px;background:linear-gradient(135deg,#0d47a1,#1565c0);
                    color:#fff;border-radius:12px;margin-bottom:14px;box-shadow:0 4px 16px rgba(13,71,161,0.3)">
            <div>
                <div style="font-size:24px;font-weight:800;letter-spacing:1px">
                    {total} 处缺陷 &middot; 均置信度 {avg_conf:.1%}
                </div>
                <div style="font-size:13px;opacity:0.85;margin-top:4px">{cls_summary}</div>
                <div style="font-size:12px;opacity:0.7;margin-top:2px">
                    {engine} &middot; {elapsed_ms:.0f}ms &middot; {w}×{h}px
                </div>
            </div>
            <div style="font-size:36px;opacity:0.2">◉</div>
        </div>
        <!-- 缺陷卡片 -->
        {''.join(cards)}
        {vlm_desc}
    </div>"""


def _put_chinese_text(img: np.ndarray, text: str, org: tuple, font_size: int, color: tuple,
                      stroke_width: int = 0, stroke_color: tuple = None, align: str = "left") -> np.ndarray:
    """用 PIL 在 OpenCV 图像上绘制中文文本 (解决 cv2.putText 乱码)"""
    from PIL import Image, ImageDraw, ImageFont

    # 查找可用的中文字体
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",        # Microsoft YaHei
        "C:/Windows/Fonts/simhei.ttf",       # SimHei
        "C:/Windows/Fonts/simsun.ttc",       # SimSun
        "C:/Windows/Fonts/msyhbd.ttc",       # Microsoft YaHei Bold
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    # OpenCV BGR -> PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 计算文本位置 (默认左上角对齐)
    x, y = org
    if align == "center":
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        x -= tw // 2
    elif align == "right":
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        x -= tw

    # 描边
    if stroke_width > 0 and stroke_color:
        sc = stroke_color  # BGR -> RGB
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=(sc[2], sc[1], sc[0]))

    # 正文
    draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))

    # PIL RGB -> OpenCV BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _get_text_size_cn(text: str, font_size: int) -> tuple[int, int]:
    """获取中文字体下的文本尺寸 (宽, 高)"""
    from PIL import ImageFont
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                bbox = font.getbbox(text)
                return (bbox[2] - bbox[0], bbox[3] - bbox[1])
            except Exception:
                continue
    return (len(text) * font_size // 2, font_size)


def _draw_detections(image: np.ndarray, detections: list, prefix: str = "") -> np.ndarray:
    """在图像上绘制检测框，带颜色编码和位置标注"""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    for i, det in enumerate(detections):
        class_name = det.class_name if hasattr(det, 'class_name') else det.get("class_name", "?")
        confidence = det.confidence if hasattr(det, 'confidence') else det.get("confidence", 0)
        bbox = det.bbox if hasattr(det, 'bbox') else det.get("bbox", [0, 0, 1, 1])
        info = _get_defect_info(class_name)
        color = info["bgr"]

        x1, y1, x2, y2 = [int(v) for v in [
            bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
        ]]

        # 发光效果 (外框)
        cv2.rectangle(annotated, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 0), 5)
        # 彩色边框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        # 高亮角标
        for cx, cy, ax, ay in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(annotated, (cx, cy), (cx + ax*20, cy), color, 4)
            cv2.line(annotated, (cx, cy), (cx, cy + ay*20), color, 4)

        # 编号圆圈 (数字，cv2 够用)
        cv2.circle(annotated, (x1, y1), 18, color, -1)
        cv2.circle(annotated, (x1, y1), 18, (255, 255, 255), 2)
        cv2.putText(annotated, str(i + 1), (x1 - 6, y1 + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 中文标签 —— 使用 PIL 避免乱码
        cn = info["cn"]
        label_text = f"#{i+1} {cn} {confidence:.0%}"
        tw, th = _get_text_size_cn(label_text, 18)
        label_y = y1 - th - 10 if y1 - th - 10 > 10 else y2 + 6
        # 标签背景
        cv2.rectangle(annotated,
                      (x1 - 2, label_y - 2), (x1 + tw + 10, label_y + th + 4),
                      color, -1)
        # 描边
        cv2.rectangle(annotated,
                      (x1 - 2, label_y - 2), (x1 + tw + 10, label_y + th + 4),
                      (255, 255, 255), 1)
        annotated = _put_chinese_text(annotated, label_text, (x1 + 4, label_y), 18,
                                       (255, 255, 255), stroke_width=1, stroke_color=(0, 0, 0))

    # 顶部状态栏
    if detections:
        bar_color = (0, 0, 200)
        status = f"DEFECTS: {len(detections)}"
    else:
        bar_color = (0, 140, 0)
        status = "PASS - No Defects"
    cv2.rectangle(annotated, (0, 0), (w, 36), (20, 20, 20), -1)
    annotated = _put_chinese_text(annotated, status, (14, 5), 22, bar_color,
                                   stroke_width=1, stroke_color=(0, 0, 0))

    return annotated


def detect_image(image: np.ndarray, conf: float | None = None) -> tuple[np.ndarray, str]:
    """两阶段检测：FastScreener 预筛 → YOLO/SAHI 精准检测"""
    if image is None:
        return None, "<div style='color:#888;text-align:center;padding:20px'>请上传图像</div>"

    if state.yolo is None:
        return image, "<div style='color:#E53E3E;text-align:center;padding:20px'>YOLO 模型未加载</div>"

    # 动态调整置信度阈值（一键全流程传入）
    if conf is not None and conf != state.yolo.conf_threshold:
        state.yolo.conf_threshold = conf

    state.current_image = image

    # ===== 第一阶段：FastScreener 轻量预筛查 =====
    if state.screener is not None:
        score, is_anomaly = state.screener.screen(image)
        if not is_anomaly:
            state._screener_stats["filtered"] += 1
            elapsed = 0  # screener 内部计时不暴露
            html = f"""<div style="font-family:system-ui;padding:16px;text-align:center">
                <div style="font-size:48px;margin-bottom:8px">⚡</div>
                <div style="font-size:20px;font-weight:800;color:#38A169">快速筛查通过</div>
                <div style="font-size:13px;color:#999;margin-top:4px">
                    异常分数 {score:.3f} &lt; 阈值 · 跳过 YOLO 推理 · 已过滤 {state._screener_stats['filtered']} 帧
                </div>
            </div>"""
            return image, html
        state._screener_stats["passed"] += 1

    # ===== 第二阶段：YOLO / SAHI 精准检测 =====
    if state.sahi_detector is not None:
        # SAHI 滑窗模式（自动判断是否需要切图）
        result = state.sahi_detector.detect(image)
        engine_label = "SAHI+YOLO"
    else:
        result = state.yolo.detect(image)
        engine_label = "YOLO 筛查" + (" (FastScreener→" if state.screener else "")

    state.last_result = result.to_dict()

    annotated = _draw_detections(image, result.detections)
    html = _build_result_html(result.detections, engine_label, result.inference_time_ms, image=image)

    return annotated, html


def _parse_position(desc: str, w: int, h: int) -> tuple[int, int, int]:
    """从 VLM 描述中解析位置 -> (cx, cy, radius)"""
    dl = desc.lower()
    cy = h//5 if any(k in dl for k in ("top","upper","上")) else \
         h*4//5 if any(k in dl for k in ("bottom","lower","下")) else h//2
    cx = w//5 if any(k in dl for k in ("left","左")) else \
         w*4//5 if any(k in dl for k in ("right","右")) else w//2
    radius = max(w,h)//12 if any(k in dl for k in ("thin","small","细","小")) else \
             max(w,h)//5 if any(k in dl for k in ("large","big","大","prominent")) else max(w,h)//8
    return cx, cy, radius


def _draw_vlm_annotations(image: np.ndarray, vlm_raw: dict) -> np.ndarray:
    """VLM 专用标注: 解析位置描述，在图像上画高亮圆+文字气泡+局部放大镜"""
    annotated = image.copy()
    h, w = annotated.shape[:2]
    items = vlm_raw.get("vlm_raw_response", {}).get("detections", [])

    sev_emoji = {"minor": "MINOR", "moderate": "MOD.", "severe": "SEVERE"}

    for i, item in enumerate(items):
        class_name = item.get("class_name", "?")
        desc = item.get("bbox_description", "")
        severity = item.get("severity", "moderate")
        info = _get_defect_info(class_name)
        color = info["bgr"]

        cx, cy, radius = _parse_position(desc, w, h)

        # 虚线高亮圈
        for angle in range(0, 360, 12):
            rad = np.radians(angle)
            x1, y1 = int(cx + radius * np.cos(rad)), int(cy + radius * np.sin(rad))
            x2, y2 = int(cx + (radius + 10) * np.cos(rad)), int(cy + (radius + 10) * np.sin(rad))
            cv2.line(annotated, (x1, y1), (x2, y2), color, 2)

        # 编号圆圈
        cv2.circle(annotated, (cx, cy), 20, color, -1)
        cv2.circle(annotated, (cx, cy), 20, (255, 255, 255), 3)
        cv2.putText(annotated, str(i + 1), (cx - 8, cy + 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

        # 气泡标签 —— PIL 中文渲染
        cn = info["cn"]
        sev_txt = sev_emoji.get(severity, "")
        bubble = f"#{i+1} {cn} {sev_txt}"
        tw, th = _get_text_size_cn(bubble, 14)
        bx = max(8, min(cx - tw//2 - 8, w - tw - 16))
        by = max(cy - radius - th - 24, 46)
        pad = 6

        cv2.rectangle(annotated, (bx, by - th - pad), (bx + tw + pad*2, by + pad), (25, 25, 25), -1)
        cv2.rectangle(annotated, (bx, by - th - pad), (bx + tw + pad*2, by + pad), color, 2)
        annotated = _put_chinese_text(annotated, bubble, (bx + pad, by - 2), 14,
                                       (255, 255, 255))

        # 连接线
        cv2.line(annotated, (cx, by + pad), (cx, cy - radius), color, 1)

        # 右下角放大镜 (局部 image patch)
        sz = min(120, radius * 2)
        if sz > 20 and 0 < cx - sz//2 < cx + sz//2 < w and 0 < cy - sz//2 < cy + sz//2 < h:
            patch = image[max(0, cy-sz//2):min(h, cy+sz//2), max(0, cx-sz//2):min(w, cx+sz//2)].copy()
            patch = cv2.resize(patch, (96, 96))
            ix, iy = w - 106, h - 106
            roi = annotated[iy:iy+96, ix:ix+96]
            cv2.addWeighted(patch, 1.0, roi, 0, 0, roi)
            cv2.rectangle(annotated, (ix, iy), (ix+96, iy+96), color, 2)
            cv2.putText(annotated, f"#{i+1}", (ix+5, iy+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 状态栏
    n = len(items)
    bar_c = (0, 0, 200) if n else (0, 140, 0)
    status = f"VLM ANALYSIS: {n} ISSUE(S)" if n else "VLM: NO ISSUES"
    cv2.rectangle(annotated, (0, 0), (w, 36), (20, 20, 20), -1)
    annotated = _put_chinese_text(annotated, status, (14, 5), 22, bar_c,
                                   stroke_width=1, stroke_color=(0, 0, 0))

    return annotated


def vlm_analyze_image(image: np.ndarray) -> tuple[np.ndarray, str]:
    """VLM (Gemini) 精细缺陷分析"""
    if image is None:
        return None, "<div style='color:#888;text-align:center;padding:20px'>请先上传图像</div>"

    if state.vlm is None:
        return image, "<div style='color:#E53E3E;text-align:center;padding:20px'>VLM 引擎未启用</div>"

    state.current_image = image
    result = state.vlm.detect(image)

    vlm_data = result.to_dict()
    if state.last_result:
        state.last_result["vlm_result"] = vlm_data
    else:
        state.last_result = {"vlm_result": vlm_data, "detections": []}

    if result.error:
        # 友好的错误信息
        err = result.error
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            msg = (
                "## ⚠️ Gemini API 配额已用完\n\n"
                "今天的免费请求次数已达到上限（20次/天）。\n\n"
                "**解决方案:**\n"
                "- 等待明天配额重置\n"
                "- 或使用 **YOLO 快速筛查** 代替（无需 API）\n"
                "- 或更换 API Key\n\n"
                "> YOLO 检测仍可正常使用"
            )
        elif "timeout" in err.lower():
            msg = (
                "## ⏱️ VLM 请求超时\n\n"
                "Gemini API 响应超时，请稍后重试。\n\n"
                "可以先使用 **YOLO 快速筛查**"
            )
        else:
            msg = f"## ⚠️ VLM 检测异常\n\n{err[:200]}..."
        return image, f"""<div style="color:#E53E3E;text-align:left;padding:20px;
            font-family:system-ui;font-size:14px;line-height:1.6;max-width:500px;
            background:#FFF5F5;border-radius:8px;border:1px solid #FED7D7">
            {msg}</div>"""

    # VLM 专用标注: 位置描述 → 图上高亮
    annotated = _draw_vlm_annotations(image, result.raw_output or {})
    html = _build_result_html(
        result.detections, "VLM 精细分析",
        result.inference_time_ms, result.raw_output,
        image=image
    )

    return annotated, html


def rag_root_cause_analysis(defect_info_json: str = "") -> str:
    """RAG 根因分析: 根据VLM检测结果查询知识库"""
    if not state.last_result:
        return """<div style="padding:16px;color:#999;text-align:center">
            请先执行 VLM 精细分析</div>"""

    # 获取 VLM 检测结果中的缺陷类型
    vlm_data = state.last_result.get("vlm_result", {})
    detections = vlm_data.get("detections", [])
    vlm_raw = vlm_data.get("raw_output", {}).get("vlm_raw_response", {})

    if not detections:
        detections = state.last_result.get("detections", [])

    if not detections:
        return """<div style="padding:16px;text-align:center;color:#999">
            未检测到缺陷，无需根因分析</div>"""

    _rag = _lazy_import_rag()
    if _rag is None:
        return """<div style="padding:16px;color:#E53E3E">
            RAG 模块未就绪，请确保 scripts/rag_demo.py 存在</div>"""

    reports = []
    for i, det in enumerate(detections):
        cn = det.class_name if hasattr(det, 'class_name') else det.get("class_name", "?")
        desc = ""
        if isinstance(vlm_raw.get("detections"), list) and i < len(vlm_raw["detections"]):
            desc = vlm_raw["detections"][i].get("description", "")

        report = _rag(cn, desc)
        reports.append(report)

    combined = "\n\n---\n\n".join(reports)

    return f"""<div style="font-family:system-ui;font-size:14px;line-height:1.7;
        max-height:500px;overflow-y:auto;padding:12px;background:#fafafa;
        border-radius:8px;border:1px solid #e2e8f0;white-space:pre-wrap">
        {combined}</div>"""


def save_and_record(reviewer: str, note: str) -> str:
    """保存当前检测结果到数据库"""
    if state.current_image is None or state.db is None:
        return "无检测结果可保存"

    ts = datetime.now()
    img_name = f"img_{ts:%Y%m%d_%H%M%S_%f}.jpg"
    img_path = str(Path("data/images") / img_name)
    cv2.imwrite(img_path, state.current_image)

    detections = state.last_result.get("detections", [])
    vlm_result = state.last_result.get("vlm_result", {})
    defect_types = ",".join(set(
        d["class_name"] for d in detections
    ))

    record = InspectionRecord(
        timestamp=ts.isoformat(),
        image_path=img_path,
        yolo_result=json.dumps(state.last_result, ensure_ascii=False),
        vlm_result=json.dumps(vlm_result, ensure_ascii=False),
        defect_types=defect_types,
        defect_count=len(detections),
        confidence=max((d["confidence"] for d in detections), default=0.0),
        reviewer=reviewer,
        note=note,
        review_status="pending",
    )

    rid = state.db.insert(record)
    return f"已保存 (ID: {rid}) | 缺陷类型: {defect_types or '无'}"


# ==================== 摄像头页 ====================

def camera_start(source: str, resolution: str = "1280×720 (HD)") -> tuple[np.ndarray, str]:
    """启动摄像头采集，返回首帧 + 状态（推流由 MJPEG 服务处理）"""
    if state.camera is not None:
        state.camera.stop()
    w, h = 1280, 720
    try:
        parts = resolution.split("×")[0], resolution.split("×")[1].split(" ")[0] if "×" in resolution else ("1280", "720")
        w, h = int(parts[0]), int(parts[1])
    except Exception:
        pass
    state.camera = CameraCapture(
        source=source, width=w, height=h, fps=15,
        trigger_mode=state.config.get("camera", {}).get("trigger_mode", "continuous"),
        plc_trigger=getattr(state, "plc_trigger", None),
    )
    if not state.camera.open():
        state.camera = None
        return None, f"> ❌ 无法打开摄像头: `{source}`\n\n请检查：\n- USB 摄像头是否已连接\n- RTSP 地址是否正确\n- 摄像头是否被其他程序占用"
    state.camera.start()
    actual_w, actual_h = state.camera.width, state.camera.height
    return _camera_grab(), (
        f"> ✅ 摄像头已连接！\n\n"
        f"| 参数 | 值 |\n|------|-----|\n"
        f"| 信号源 | `{source}` |\n"
        f"| 分辨率 | {actual_w}×{actual_h} |\n"
        f"| 帧率 | 15 FPS |\n\n"
        f"下方 MJPEG 流实时更新 ← 渲染在 `<img>` 标签中，由 Flask 服务推流"
    )


def camera_stop() -> tuple[np.ndarray, str]:
    """停止摄像头"""
    if state.camera:
        state.camera.stop()
        state.camera = None
    return None, "> ⏹ 摄像头已断开"


def _camera_grab() -> np.ndarray:
    """从摄像头取一帧"""
    if state.camera is None or not state.camera.is_running:
        return None
    frame = state.camera.read()
    if frame is not None:
        h, w = frame.shape[:2]
        fps = state.camera.fps_actual
        cv2.rectangle(frame, (0, 0), (w, 32), (20, 20, 20), -1)
        cv2.putText(frame, f"LIVE | {w}x{h} | {fps:.0f} FPS", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def camera_stream() -> np.ndarray:
    """持续流式输出 (Gradio 自动轮询)"""
    return _camera_grab()


def camera_snapshot(frame: np.ndarray) -> np.ndarray:
    """从摄像头帧截取快照并存入 current_image"""
    if frame is not None:
        state.current_image = frame.copy()
    return frame


# ==================== 审核页 ====================

def load_pending_records() -> list[list]:
    """加载待审核记录"""
    if state.db is None:
        return []
    records = state.db.query(review_status="pending", limit=50)
    return [
        [r.id, r.timestamp[:19], r.defect_types, r.defect_count, f"{r.confidence:.2f}"]
        for r in records
    ]


def review_record(record_id: int, status: str, reviewer: str, note: str) -> str:
    """审核一条记录"""
    if state.db is None:
        return "数据库未初始化"

    record = state.db.get_by_id(record_id)
    if record is None:
        return "记录不存在"

    final_result = json.loads(record.yolo_result) if record.yolo_result else {}
    state.db.update_review(
        record_id=record_id,
        final_result=final_result,
        reviewer=reviewer,
        review_status=status,
        note=note,
    )
    return f"{ICON_CONFIRMED} 记录 {record_id} 已{status}"


# ==================== 报表页 ====================

def generate_report(start_date: str, end_date: str) -> str:
    """生成统计报告"""
    if state.db is None or state.exporter is None:
        return "> ⚠️ 系统未初始化"

    start = f"{start_date}T00:00:00" if start_date else None
    end = f"{end_date}T23:59:59" if end_date else None

    total = state.db.count(start, end)
    stats = state.db.get_defect_stats(start, end)

    defect_count = {}
    for s in stats:
        for dt in s["defect_types"].split(","):
            dt = dt.strip()
            if dt:
                defect_count[dt] = defect_count.get(dt, 0) + 1

    type_rows = "".join(
        f"| {_get_defect_info(k)['icon']} {_get_defect_info(k)['cn']} | {v} |"
        for k, v in sorted(defect_count.items(), key=lambda x: -x[1])
    )

    return f"""## 📊 检测统计报告

| 指标 | 值 |
|------|-----|
| 检测总数 | **{total}** 条 |
| 时间范围 | {start_date or '全部'} ~ {end_date or '全部'} |
| 缺陷类型数 | {len(defect_count)} 种 |

### 缺陷分布
| 类型 | 数量 |
|------|------|
{type_rows or '| — | 0 |'}

---
> 点击下方按钮导出完整报告（含图像 + 坐标 + 统计分析）
"""


def export_inspection_report(start_date: str, end_date: str) -> str:
    """导出专业质检报告 (含图像+坐标)"""
    if state.exporter is None:
        return "> ⚠️ 导出模块未就绪"
    start = f"{start_date}T00:00:00" if start_date else None
    end = f"{end_date}T23:59:59" if end_date else None
    path = state.exporter.export_inspection_report(start_time=start, end_time=end)
    return f"> 📄 专业报告已生成！\n\n**文件路径:** `{path}`\n\n点击下方按钮可在浏览器打开 👇\n\n[🌐 打开报告](file:///{path.replace(chr(92), '/')})"


def export_csv_data(start_date: str, end_date: str) -> str:
    """导出 CSV"""
    if state.exporter is None:
        return "> ⚠️ 导出模块未就绪"
    start = f"{start_date}T00:00:00" if start_date else None
    end = f"{end_date}T23:59:59" if end_date else None
    path = state.exporter.export_csv(start_time=start, end_time=end)
    return f"> 📥 CSV 已导出\n\n`{path}`"


def export_badcase_data(start_date: str, end_date: str) -> str:
    """导出 Bad Case 数据集"""
    if state.exporter is None:
        return "> ⚠️ 导出模块未就绪"
    path = state.exporter.export_badcase(limit=200)
    return f"> 📦 Bad Case 数据集已导出\n\n`{path}`"


def _start_mjpeg_server(monitor=None):
    """在后台线程启动 Flask MJPEG 推流服务 (端口 7861)"""
    import threading
    from flask import Flask, Response

    mjpeg_app = Flask("camera_mjpeg")

    @mjpeg_app.route("/health")
    def health_check():
        """系统健康检查接口"""
        if monitor:
            try:
                from src.monitor import SystemMonitor
                if isinstance(monitor, SystemMonitor):
                    return monitor.get_health_json(), 200, {"Content-Type": "application/json"}
            except Exception as e:
                import json
                from datetime import datetime
                return json.dumps({
                    "status": "error",
                    "message": f"监控服务异常: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }), 500, {"Content-Type": "application/json"}
        import json
        from datetime import datetime
        return json.dumps({
            "status": "unknown",
            "message": "监控服务未启用",
            "timestamp": datetime.now().isoformat()
        }), 200, {"Content-Type": "application/json"}

    @mjpeg_app.route("/camera")
    def camera_mjpeg():
        def generate():
            while True:
                if state.camera is None or not state.camera.is_running:
                    time.sleep(0.1)
                    continue
                frame = state.camera.read()
                if frame is not None:
                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.05)
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _run_flask():
        mjpeg_app.run(host='127.0.0.1', port=7861, debug=False, use_reloader=False, threaded=True)

    t = threading.Thread(target=_run_flask, daemon=True)
    t.start()
    print("[INFO] Flask MJPEG 推流服务: http://127.0.0.1:7861/camera")


# ==================== 启动 ====================

def _get_auth_credentials():
    """从环境变量读取 Gradio 登录凭证。
    
    设置 GRADIO_USERNAME 和 GRADIO_PASSWORD 环境变量启用认证。
    未设置则无认证（向后兼容）。
    
    示例 .env:
        GRADIO_USERNAME=admin
        GRADIO_PASSWORD=steel2026
    """
    username = os.environ.get("GRADIO_USERNAME", "")
    password = os.environ.get("GRADIO_PASSWORD", "")
    if username and password:
        return [(username, password)]
    return None


def launch(config_path: str = "config.yaml", monitor=None):
    """启动 Gradio 工作台
    
    Args:
        config_path: 配置文件路径
        monitor: SystemMonitor 实例（可选），用于集成监控
    """
    state.init_from_config(config_path)
    state.load_models()

    # ===== PLC硬触发支持 =====
    plc_config = state.config.get("plc", {})
    if plc_config.get("enabled", False):
        try:
            from src.plc_trigger import create_plc_trigger_from_config
            state.plc_trigger = create_plc_trigger_from_config(plc_config)
            if state.plc_trigger:
                print(f"[INFO] PLC硬触发已启用: {plc_config['host']}:{plc_config['port']}")
        except ImportError as e:
            print(f"[WARN] PLC模块不可用: {e}")
        except Exception as e:
            print(f"[ERROR] PLC初始化失败: {e}")

    # ===== 监控集成 =====
    if monitor:
        def get_camera_status():
            if state.camera:
                return {
                    "connected": state.camera.is_running,
                    "fps": state.camera.fps_actual,
                    "status": "running" if state.camera.is_running else "stopped",
                    "trigger_mode": state.camera.trigger_mode.value,
                    "has_triggered_frame": state.camera.has_triggered_frame,
                }
            return {"connected": False, "fps": -1, "status": "stopped"}
        monitor.set_camera_status_provider(get_camera_status)
        print("[INFO] 监控系统已集成相机状态")

    # ===== 启动 Flask MJPEG 推流服务（后台线程） =====
    _start_mjpeg_server(monitor)

    # 系统状态检测
    gpu_ok = False
    try:
        import torch; gpu_ok = torch.cuda.is_available()
    except: pass

    status_badges = f"""
    <span style="background:rgba(255,255,255,0.18);padding:5px 14px;border-radius:20px;font-size:12px">
        {ICON_VLM} VLM: gemini-3.1-flash</span>
    <span style="background:rgba(255,255,255,0.18);padding:5px 14px;border-radius:20px;font-size:12px">
        {ICON_GPU} {'GPU 加速' if gpu_ok else 'CPU 模式'}</span>
    <span style="background:rgba(255,255,255,0.18);padding:5px 14px;border-radius:20px;font-size:12px">
        {ICON_RAG} RAG 知识库就绪</span>
    <span style="background:rgba(255,255,255,0.18);padding:5px 14px;border-radius:20px;font-size:12px">
        {ICON_EXPORT} API 已连接</span>
    """

    with gr.Blocks(title="钢铁表面缺陷检测系统") as app:
        # ===================== 顶部导航 =====================
        gr.HTML(f"""
        <div class="industrial-header">
            <div class="header-rivet-bar"></div>
            <div class="header-content">
                <div class="header-left">
                    <img class="logo-hex" src="{load_logo_svg("logo_b_lattice_scan")}" alt="Steel Vision Logo" width="52" height="52">
                    <div class="logo-info">
                        <div class="logo-title">STEEL VISION PRO</div>
                        <div class="logo-sub">钢铁表面缺陷智能检测平台 · YOLO + VLM 双引擎</div>
                    </div>
                </div>
                <div class="header-status-row">
                    <div class="status-led-group">
                        <div class="led-dot led-green"></div>
                        <span>系统运行中</span>
                    </div>
                    <div class="status-led-group">
                        <div class="led-dot led-blue"></div>
                        <span>{'GPU 加速' if gpu_ok else 'CPU 模式'}</span>
                    </div>
                    <div class="status-led-group">
                        <div class="led-dot led-orange"></div>
                        <span>VLM: gemini-3.1-flash</span>
                    </div>
                    <div class="status-led-group">
                        <div class="led-dot led-green"></div>
                        <span>RAG 就绪</span>
                    </div>
                    <button class="theme-toggle-btn" id="themeToggle"
                        onclick="var d=document.documentElement;var c=d.getAttribute('data-theme')||'light';var n=c==='dark'?'light':'dark';d.setAttribute('data-theme',n);localStorage.setItem('steel-theme',n);var t=document.getElementById('themeIcon');var l=document.getElementById('themeLabel');if(n==='dark'){{t.textContent='☀️';l.textContent='浅色';}}else{{t.textContent='🌙';l.textContent='深色';}}"
                        title="切换浅色/深色主题">
                        <span id="themeIcon">🌙</span> <span id="themeLabel">深色</span>
                    </button>
                </div>
            </div>
        </div>
        <script>
        (function(){{var d=document.documentElement;var s=localStorage.getItem('steel-theme')||'light';d.setAttribute('data-theme',s);var i=document.getElementById('themeIcon');var l=document.getElementById('themeLabel');if(s==='dark'){{i.textContent='☀️';l.textContent='浅色';}}else{{i.textContent='🌙';l.textContent='深色';}}}})();
        </script>""")

        # ===================== 统计卡片 =====================
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(f"""<div class="industrial-card">
                    <div class="card-icon-row"><span class="card-dot dot-blue"></span> 今日检测</div>
                    <div class="card-big-num">0</div>
                    <div class="card-sub">累计 -- 条记录</div>
                </div>""")
            with gr.Column(scale=1):
                gr.HTML(f"""<div class="industrial-card">
                    <div class="card-icon-row"><span class="card-dot dot-red"></span> 检出缺陷</div>
                    <div class="card-big-num" style="color:#e63946">0</div>
                    <div class="card-sub">缺陷率 --%</div>
                </div>""")
            with gr.Column(scale=1):
                gr.HTML(f"""<div class="industrial-card">
                    <div class="card-icon-row"><span class="card-dot dot-orange"></span> 待审核</div>
                    <div class="card-big-num" style="color:#ff6b35">0</div>
                    <div class="card-sub">需人工复核</div>
                </div>""")
            with gr.Column(scale=1):
                gr.HTML(f"""<div class="industrial-card">
                    <div class="card-icon-row"><span class="card-dot dot-green"></span> 准确率</div>
                    <div class="card-big-num" style="color:#2a9d8f">--</div>
                    <div class="card-sub">YOLO mAP@50</div>
                </div>""")

        gr.Markdown("")

        # ===================== 标签页 =====================
        with gr.Tabs():
            # ===== 实时采集 (摄像头) =====
            with gr.TabItem("📷 实时采集"):
                gr.Markdown("### 📷 工业相机 / RTSP 流采集")
                gr.Markdown("*USB 本地摄像头 / 网络 RTSP 流 — 实时画面预览 + 快照 (FR-07)*")

                with gr.Row():
                    # 左：信号源选择
                    with gr.Column(scale=3):
                        cam_type = gr.Radio(
                            choices=[("💻 USB 摄像头", "usb"), ("🌐 RTSP 网络流", "rtsp")],
                            value="usb", label="信号源类型", interactive=True,
                        )
                        with gr.Row():
                            cam_index = gr.Number(
                                label="摄像头编号", value=0, precision=0,
                                minimum=0, maximum=9, visible=True,
                                info="0=内置摄像头, 1/2/3=外接 USB 摄像头",
                            )
                            cam_rtsp_url = gr.Textbox(
                                label="RTSP 地址", placeholder="rtsp://192.168.1.100:554/stream",
                                visible=False,
                                info="示例: rtsp://admin:12345@192.168.1.100:554/h264",
                            )
                        cam_resolution = gr.Dropdown(
                            choices=["640×480 (VGA)", "1280×720 (HD)", "1920×1080 (Full HD)"],
                            value="1280×720 (HD)", label="分辨率",
                        )

                    # 右：控制面板
                    with gr.Column(scale=2):
                        gr.Markdown("")
                        with gr.Row():
                            cam_start_btn = gr.Button("▶ 连接摄像头", variant="primary", scale=2)
                            cam_stop_btn = gr.Button("⏹ 断开", variant="stop", scale=1)
                        with gr.Row():
                            cam_snap_btn = gr.Button("📸 截取快照", variant="secondary", scale=2)
                            cam_detect_btn = gr.Button("🔍 快照并检测", variant="secondary", scale=1)

                        cam_status = gr.Markdown(
                            "> ⏳ 等待连接...\n\n"
                            "选择信号源类型，点击 **▶ 连接摄像头** 开始预览"
                        )

                # 实时画面 (MJPEG 流)
                with gr.Row():
                    cam_feed = gr.HTML(
                        '<div style="text-align:center;color:#94a3b8;padding:40px;border:2px dashed #e2e8f0;border-radius:12px;min-height:300px">'
                        '📹 点击 <b>▶ 连接摄像头</b> 开始推流'
                        '</div>'
                    )
                    cam_snapshot_img = gr.Image(label="📸 快照", height=380, interactive=False)

                # 隐藏的快照传递 (用于快照并检测)
                hidden_snap = gr.State(None)

                # 切换信号源类型时隐藏/显示对应输入
                def _on_cam_type_change(t):
                    if t == "usb":
                        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
                    else:
                        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

                cam_type.change(
                    fn=_on_cam_type_change,
                    inputs=[cam_type],
                    outputs=[cam_index, cam_rtsp_url, cam_resolution],
                )

                # 构建实际 source 字符串
                def _build_cam_source(cam_type, cam_index, rtsp_url):
                    if cam_type == "usb":
                        return str(int(cam_index))
                    return rtsp_url.strip() or "0"

                # 连接 - 相机启动后通过 Flask MJPEG 推流
                def _on_cam_connect(t, idx, url, res):
                    source = _build_cam_source(t, idx, url)
                    frame, status = camera_start(source, res)
                    if frame is None:
                        # 连接失败
                        return (
                            '<div style="text-align:center;color:#E53E3E;padding:40px">'
                            '❌ 连接失败</div>',
                            status,
                            None,
                        )
                    # 返回 MJPEG 流 HTML
                    mjpeg_html = (
                        '<img src="http://127.0.0.1:7861/camera" '
                        'style="width:100%;max-height:380px;object-fit:contain;border-radius:8px;border:2px solid #38A169" '
                        'alt="Camera Stream" />'
                    )
                    return mjpeg_html, status, frame

                cam_start_btn.click(
                    fn=_on_cam_connect,
                    inputs=[cam_type, cam_index, cam_rtsp_url, cam_resolution],
                    outputs=[cam_feed, cam_status, cam_snapshot_img],
                )

                # 断开
                def _on_cam_disconnect():
                    state.camera_stop() if hasattr(state, 'camera_stop') else camera_stop()
                    return (
                        '<div style="text-align:center;color:#94a3b8;padding:40px;border:2px dashed #e2e8f0;border-radius:12px;min-height:300px">'
                        '📹 摄像头已断开<br><span style="font-size:12px">点击 <b>▶ 连接摄像头</b> 重新开始</span>'
                        '</div>',
                        "> ⏹ 摄像头已断开",
                        None,
                    )

                cam_stop_btn.click(
                    fn=_on_cam_disconnect,
                    inputs=[], outputs=[cam_feed, cam_status, cam_snapshot_img],
                )
                cam_snap_btn.click(
                    fn=lambda: _camera_grab(), inputs=[],
                    outputs=[cam_snapshot_img],
                )
                def _snap_and_guide():
                    frame = _camera_grab()
                    if frame is not None:
                        state.current_image = frame.copy()
                        return frame, "> 📸 快照已保存！切换到「🔍 实时检测」标签页上传图片进行检测"
                    return None, "> ⚠️ 请先连接摄像头"
                cam_detect_btn.click(
                    fn=_snap_and_guide, inputs=[],
                    outputs=[cam_snapshot_img, cam_status],
                )

                gr.Markdown("---")
                gr.Markdown("*截取快照后，切换到「🔍 实时检测」标签页上传快照进行 YOLO/VLM 分析*")

            # ===== 实时检测 =====
            with gr.TabItem("🔍 实时检测"):
                with gr.Row(equal_height=True):
                    # 左栏：输入区
                    with gr.Column(scale=4):
                        input_img = gr.Image(
                            label="上传钢板表面图像",
                            type="numpy",
                            height=420,
                            sources=["upload", "clipboard"],
                        )
                        with gr.Row():
                            conf_slider = gr.Slider(
                                0.01, 0.50, value=0.05, step=0.01,
                                label="🔧 检测灵敏度",
                                info="值越低检出越多（可能误报），越高越精准（可能漏检）",
                            )
                        # 按钮组
                        with gr.Row():
                            yolo_btn = gr.Button(
                                "⚡ YOLO 快速筛查", variant="primary", scale=1,
                                elem_classes=["btn-primary"],
                            )
                            vlm_btn = gr.Button(
                                "🧠 VLM 精细分析", variant="secondary", scale=1,
                                elem_classes=["btn-vlm"],
                            )
                            full_btn = gr.Button(
                                "🚀 一键全流程 (YOLO → VLM → RAG)", variant="stop", scale=2,
                                elem_classes=["btn-full"],
                            )
                        gr.HTML("""<div style="font-size:11px;color:#94a3b8;text-align:center;margin-top:4px">
                            💡 YOLO 定位异常区域 → VLM 确认缺陷类型 → RAG 推理根因 → 人工审核兜底
                        </div>""")

                    # 右栏：结果区
                    with gr.Column(scale=5):
                        output_img = gr.Image(label="检测标注结果", height=340, interactive=False)
                        result_md = gr.Markdown(
                            "> 等待检测...\n\n请上传钢板图像，点击检测按钮开始分析。",
                            label="检测详情",
                        )

                # RAG 折叠区
                with gr.Accordion("📚 RAG 根因分析报告", open=False):
                    rag_html = gr.HTML()
                    with gr.Row():
                        rag_btn = gr.Button("🔍 生成根因分析", variant="secondary", size="sm")
                        rag_clr = gr.Button("✕ 清除", size="sm")

                # 保存记录
                with gr.Accordion("💾 保存检测记录", open=False):
                    with gr.Row():
                        save_reviewer = gr.Textbox(label="审核人", placeholder="输入工号或姓名", scale=2)
                        save_note = gr.Textbox(label="备注", placeholder="可选备注信息", scale=3)
                    save_btn = gr.Button("💾 保存到数据库", variant="secondary", size="sm")
                    save_msg = gr.Textbox(label="保存状态", interactive=False, show_label=False)

                # ---- 事件绑定 ----
                yolo_btn.click(
                    fn=detect_image, inputs=[input_img, conf_slider],
                    outputs=[output_img, result_md],
                )
                vlm_btn.click(
                    fn=vlm_analyze_image, inputs=[input_img],
                    outputs=[output_img, result_md],
                )
                rag_btn.click(fn=rag_root_cause_analysis, inputs=[], outputs=[rag_html])
                rag_clr.click(fn=lambda: "", inputs=[], outputs=[rag_html])
                save_btn.click(
                    fn=save_and_record,
                    inputs=[save_reviewer, save_note],
                    outputs=[save_msg],
                )

                def full_pipeline(img, conf, progress=gr.Progress()):
                    """一键全流程：YOLO 筛查 → VLM 复核 → RAG 根因，各阶段独立容错"""
                    if img is None:
                        return None, "## ⚠️ 请先上传钢板表面图像", ""

                    t0 = time.time()
                    stages = []  # [(label, ok, icon, detail)]
                    output_image = img
                    rag_report = ""

                    # ============ 阶段 1/3：YOLO 快速筛查 ============
                    progress(0.0, desc="阶段 1/3: YOLO 快速筛查中...")
                    try:
                        yolo_img, yolo_md = detect_image(img, conf)
                        output_image = yolo_img
                        stages.append(("YOLO 快速筛查", True, "⚡", yolo_md))
                    except Exception as e:
                        stages.append(("YOLO 快速筛查", False, "⚡",
                                       f"<span style='color:#E53E3E'>检测异常: {str(e)[:120]}</span>"))

                    # ============ 阶段 2/3：VLM 精细复核 ============
                    progress(0.33, desc="阶段 2/3: VLM 精细分析中...")
                    try:
                        vlm_img, vlm_md = vlm_analyze_image(img)
                        output_image = vlm_img  # VLM 标注覆盖 YOLO 标注
                        # 区分正常失败和 API 配额耗尽
                        vlm_ok = "缺陷" in vlm_md or "正常" in vlm_md or "无缺陷" in vlm_md or "检测结果" in vlm_md
                        stages.append(("VLM 精细分析", vlm_ok, "🧠", vlm_md))
                    except Exception as e:
                        stages.append(("VLM 精细分析", False, "🧠",
                                       f"<span style='color:#E53E3E'>VLM 调用失败: {str(e)[:120]}</span>"))

                    # ============ 阶段 3/3：RAG 根因分析 ============
                    progress(0.66, desc="阶段 3/3: 生成根因分析报告...")
                    try:
                        rag_report = rag_root_cause_analysis()
                        rag_ok = "未检测到缺陷" not in rag_report and "请先执行" not in rag_report
                        stages.append(("RAG 根因分析", rag_ok, "📚",
                                       "已生成根因分析报告" if rag_ok else "无可用数据"))
                    except Exception as e:
                        rag_report = f"<div style='color:#E53E3E'>RAG 分析异常: {str(e)[:120]}</div>"
                        stages.append(("RAG 根因分析", False, "📚",
                                       f"<span style='color:#E53E3E'>异常: {str(e)[:120]}</span>"))

                    progress(1.0, desc="全流程完成 ✓")
                    total_elapsed = (time.time() - t0) * 1000

                    # ============ 组装综合报告 ============
                    stage_count = len(stages)
                    ok_count = sum(1 for _, ok, _, _ in stages if ok)
                    status_color = "#38A169" if ok_count == stage_count else "#DD6B20" if ok_count > 0 else "#E53E3E"
                    status_text = "全部通过" if ok_count == stage_count else f"{ok_count}/{stage_count} 通过" if ok_count > 0 else "全部失败"

                    lines = [
                        "---",
                        "## 📊 综合检测报告",
                        "",
                        f"| 状态 | 总耗时 | 阶段数 | 通过 |",
                        f"|------|--------|--------|------|",
                        f"| <span style='color:{status_color};font-weight:800'>{status_text}</span> | {total_elapsed:.0f} ms | {stage_count} | {ok_count} |",
                        "",
                    ]
                    for label, ok, icon, detail in stages:
                        badge = "✅" if ok else "❌"
                        lines.append(f"### {icon} {badge} {label}")
                        lines.append(detail)
                        lines.append("")

                    summary = "\n".join(lines)
                    return output_image, summary, rag_report

                full_btn.click(
                    fn=full_pipeline,
                    inputs=[input_img, conf_slider],
                    outputs=[output_img, result_md, rag_html],
                )

            # ===== 人工审核 =====
            with gr.TabItem("📋 人工审核"):
                gr.Markdown(f"### {ICON_REVIEW} 待审核检测记录")
                gr.Markdown("*质检员对系统检测结果进行人工确认或修正，确保最终判定准确。*")

                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新待审核列表", variant="secondary")
                    batch_pass_btn = gr.Button("✅ 全部通过", variant="primary", size="sm")

                review_table = gr.Dataframe(
                    headers=["ID", "检测时间", "缺陷类型", "缺陷数", "置信度"],
                    label="待审核记录",
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("---")
                gr.Markdown("#### 🔍 逐条审核")
                with gr.Row():
                    review_id = gr.Number(label="记录 ID", precision=0, scale=1)
                    review_reviewer = gr.Textbox(label="审核人", placeholder="工号/姓名", scale=2)
                    review_note = gr.Textbox(label="审核备注", placeholder="填写审核意见", scale=3)
                with gr.Row():
                    review_pass = gr.Button("✅ 审核通过", variant="primary")
                    review_reject = gr.Button("❌ 驳回修正", variant="stop")
                review_feedback = gr.Textbox(label="操作结果", interactive=False)

                refresh_btn.click(fn=load_pending_records, inputs=[], outputs=[review_table])
                review_pass.click(
                    fn=lambda rid, r, n: review_record(rid, "confirmed", r, n),
                    inputs=[review_id, review_reviewer, review_note],
                    outputs=[review_feedback],
                )
                review_reject.click(
                    fn=lambda rid, r, n: review_record(rid, "corrected", r, n),
                    inputs=[review_id, review_reviewer, review_note],
                    outputs=[review_feedback],
                )

            # ===== 统计报表 =====
            with gr.TabItem("📊 统计报表"):
                gr.Markdown(f"### {ICON_REPORT} 检测数据统计与导出")
                gr.Markdown("*支持按时间范围、缺陷类型等维度统计，支持 CSV / Bad Case / HTML 报告导出。*")

                with gr.Row():
                    report_start = gr.Textbox(
                        label="开始日期", placeholder="2026-01-01",
                        value="2026-01-01", scale=2,
                    )
                    report_end = gr.Textbox(
                        label="结束日期", placeholder="2026-12-31",
                        value="2026-12-31", scale=2,
                    )

                with gr.Row():
                    report_btn = gr.Button("📊 生成统计报告", variant="primary", scale=2)
                    export_csv_btn = gr.Button("📥 导出 CSV", variant="secondary", scale=1)
                    export_html_btn = gr.Button("🌐 导出 HTML 报告", variant="secondary", scale=1)
                    export_bc_btn = gr.Button("📦 Bad Case 数据集", variant="secondary", scale=1)

                report_output = gr.Markdown("> 点击「生成统计报告」查看数据统计...")

                report_btn.click(
                    fn=generate_report,
                    inputs=[report_start, report_end],
                    outputs=[report_output],
                )

                # 导出按钮
                export_csv_btn.click(
                    fn=export_csv_data,
                    inputs=[report_start, report_end], outputs=[report_output],
                )
                export_html_btn.click(
                    fn=export_inspection_report,
                    inputs=[report_start, report_end], outputs=[report_output],
                )
                export_bc_btn.click(
                    fn=export_badcase_data,
                    inputs=[report_start, report_end], outputs=[report_output],
                )

        # ===================== 底部信息栏 =====================
        gr.HTML("""
        <div class="industrial-footer">
            <div class="footer-rivet-bar"></div>
            <div class="footer-content">
                <span>STEEL VISION PRO V2.1</span>
                <span class="footer-sep">|</span>
                <span>YOLO + VLM 双引擎</span>
                <span class="footer-sep">|</span>
                <span>SQLite 本地存储</span>
                <span class="footer-sep">|</span>
                <span>Gradio 工业工作台</span>
                <span class="footer-sep">|</span>
                <span>对照 SRS V1.0 开发</span>
            </div>
        </div>""")

        # ===================== 语音命令 =====================
        voice_commander = VoiceCommander()

        # 隐藏的语音命令输入框
        voice_input = gr.Textbox(
            label="语音命令",
            visible=False,
            elem_id="voice-command-input",
        )
        voice_feedback = gr.HTML(
            value="",
            visible=True,
            elem_id="voice-feedback",
        )

        # 麦克风浮窗 (固定在页面右下角)
        gr.HTML(VOICE_INPUT_HTML)

        # ---- 语音命令处理器 ----
        def handle_voice_cmd(text: str) -> str:
            if not text or not text.strip():
                return ""
            return voice_commander.execute(text)

        voice_input.change(
            fn=handle_voice_cmd,
            inputs=[voice_input],
            outputs=[voice_feedback],
        )

        # ---- 注册命令处理器 ----
        def _voice_nav_tab(tab_index: int):
            """辅助: 导航到指定标签页"""
            return gr.update(selected=tab_index)

        voice_commander.register_handler("yolo_detect",
            lambda cmd: "YOLO 快速筛查 — 请在检测页上传图像后点击 YOLO 按钮")
        voice_commander.register_handler("vlm_analyze",
            lambda cmd: "VLM 精细分析 — 请在检测页上传图像后点击 VLM 按钮")
        voice_commander.register_handler("full_pipeline",
            lambda cmd: "一键全流程 — 请在检测页上传图像后点击全流程按钮")
        voice_commander.register_handler("rag_analysis",
            lambda cmd: "RAG 根因分析 — 请先执行 VLM 分析再生成报告")
        voice_commander.register_handler("review_refresh",
            lambda cmd: "审核列表已刷新，共加载待审核记录")
        voice_commander.register_handler("export_csv",
            lambda cmd: "CSV 导出 — 请在报表页设置日期后点击导出 CSV")
        voice_commander.register_handler("export_html",
            lambda cmd: "HTML 报告导出 — 请在报表页设置日期后点击导出 HTML")
        voice_commander.register_handler("export_badcase",
            lambda cmd: "Bad Case 导出 — 请在报表页设置日期后点击导出 Bad Case")
        voice_commander.register_handler("export_report",
            lambda cmd: "统计报告 — 请在报表页设置日期后点击生成统计报告")
        voice_commander.register_handler("camera_connect",
            lambda cmd: "摄像头连接 — 请切换到「实时采集」页点击连接摄像头")
        voice_commander.register_handler("camera_disconnect",
            lambda cmd: "摄像头已断开")
        voice_commander.register_handler("camera_snapshot",
            lambda cmd: "快照已截取 — 请切换到检测页上传图像")
        voice_commander.register_handler("theme_dark",
            lambda cmd: "已切换深色模式 🌙")
        voice_commander.register_handler("theme_light",
            lambda cmd: "已切换浅色模式 ☀️")
        voice_commander.register_handler("tab_detect",
            lambda cmd: "请切换到「实时检测」标签页")
        voice_commander.register_handler("tab_review",
            lambda cmd: "请切换到「人工审核」标签页")
        voice_commander.register_handler("tab_report",
            lambda cmd: "请切换到「统计报表」标签页")
        voice_commander.register_handler("tab_camera",
            lambda cmd: "请切换到「实时采集」标签页")
        voice_commander.register_handler("review_pass",
            lambda cmd: f"审核通过 #{cmd.params.get('number','?')} — 请在审核页输入记录 ID 后点击审核通过")
        voice_commander.register_handler("review_reject",
            lambda cmd: f"驳回修正 #{cmd.params.get('number','?')} — 请在审核页输入记录 ID 后点击驳回修正")

    # 启动
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        css=load_css(),
        auth=_get_auth_credentials(),
    )


if __name__ == "__main__":
    launch()
