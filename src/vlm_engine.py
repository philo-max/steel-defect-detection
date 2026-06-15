"""
VLM 检测引擎 - 支持云端API + 本地离线分析回退。
优先: 阿里通义千问 (国内) > Google Gemini > 本地离线分析
"""

import json
import os
import time
import base64
import socket
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from .base_detector import BaseDetector, InferenceResult, DetectionResult
from .utils.bbox import bbox_iou

# 支持的 API 提供商配置 (国内优先)
_PROVIDERS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "default_model": "qwen-vl-max",
        "name": "阿里通义千问",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
        "name": "Google Gemini",
    },
}

_VALID_DEFECT_CLASSES = {
    "crack", "scratch", "scale", "indentation", "blister",
    "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"
}
_VALID_SEVERITIES = {"minor", "moderate", "severe"}


def _detect_provider(api_base=None):
    """自动检测可用的 VLM 提供商，国内优先"""
    if api_base:
        key = (os.getenv("DASHSCOPE_API_KEY") or os.getenv("GEMINI_API_KEY") or
               os.getenv("VLM_API_KEY") or "")
        return ("custom", key, api_base, "自定义")

    # 1. 优先阿里通义千问（国内访问稳定）
    for name in ["qwen", "gemini"]:
        cfg = _PROVIDERS[name]
        key = os.getenv(cfg["env_key"], "").strip()
        if key:
            return (name, key, cfg["base_url"], cfg["name"])

    return ("none", "", "", "none")


def _quick_connectivity_check(host: str, port: int = 443, timeout: float = 3.0) -> bool:
    """快速检测网络连通性"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


# ============================================================
# 本地离线分析引擎 (无需 API Key，完全离线运行)
# ============================================================

class LocalAnalyzer:
    """基于图像处理的本地离线缺陷分析器"""

    # 缺陷中文说明
    _DEFECT_INFO = {
        "crazing":          {"cn": "裂纹(龟裂)",   "desc": "表面网状细小裂纹，通常由热应力或不当冷却引起"},
        "scratches":        {"cn": "划痕",         "desc": "细长条状机械损伤，沿轧制方向或随机分布"},
        "patches":          {"cn": "斑块",         "desc": "色差明显的区域状缺陷，可能由氧化不均或污染造成"},
        "pitted_surface":   {"cn": "麻点/凹坑",     "desc": "密集分布的小凹坑，通常由轧辊表面粗糙或腐蚀引起"},
        "inclusion":        {"cn": "夹杂物",       "desc": "非金属夹杂物嵌入表面，呈暗色不规则斑点"},
        "rolled-in_scale":  {"cn": "轧制氧化皮",    "desc": "氧化皮被压入表面形成的暗色不规则斑块"},
    }

    def analyze(self, image: np.ndarray) -> InferenceResult:
        start = time.perf_counter()

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape[:2]

            detections = []

            # 1. 边缘异常检测 (裂纹/划痕)
            edge_detections = self._detect_edges(gray, (h, w))
            detections.extend(edge_detections)

            # 2. 纹理异常检测 (斑块/麻点/氧化皮)
            texture_detections = self._detect_texture(gray, (h, w))
            detections.extend(texture_detections)

            # 3. 暗色区域检测 (夹杂/氧化皮)
            dark_detections = self._detect_dark_regions(gray, (h, w))
            detections.extend(dark_detections)

            # 4. 全局亮度异常检测
            brightness_detections = self._detect_brightness_anomaly(gray, (h, w))
            detections.extend(brightness_detections)

            # 去重合并
            detections = self._merge_detections(detections)

            elapsed = (time.perf_counter() - start) * 1000

            return InferenceResult(
                detections=detections,
                inference_time_ms=elapsed,
                image_shape=image.shape[:2],
                raw_output={
                    "mode": "local",
                    "method": "image_processing",
                    "vlm_raw_response": {
                        "detections": [
                            {
                                "class_name": d.class_name,
                                "confidence": d.confidence,
                                "bbox_description": self._DEFECT_INFO.get(
                                    d.class_name, {}
                                ).get("desc", ""),
                                "severity": "moderate",
                            }
                            for d in detections
                        ]
                    }
                },
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return InferenceResult(inference_time_ms=elapsed, error=f"本地分析异常: {e}")

    def _detect_edges(self, gray: np.ndarray, shape: tuple) -> list:
        """边缘检测 - 找裂纹/划痕"""
        h, w = shape
        detections = []

        # 自适应 Canny 边缘检测
        median_val = np.median(gray)
        lower = int(max(0, 0.66 * median_val))
        upper = int(min(255, 1.33 * median_val))
        edges = cv2.Canny(gray, lower, upper)

        # 霍夫线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=30, maxLineGap=10)

        if lines is not None and len(lines) > 0:
            # 按长度分类
            long_lines = []
            short_lines = []
            for line in lines:
                length = np.linalg.norm([line[0][2] - line[0][0], line[0][3] - line[0][1]])
                if length > 60:
                    long_lines.append(line)
                else:
                    short_lines.append(line)

            # 长线 -> 划痕
            if len(long_lines) >= 3:
                xs = [l[0][0] for l in long_lines] + [l[0][2] for l in long_lines]
                ys = [l[0][1] for l in long_lines] + [l[0][3] for l in long_lines]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                padding = 0.03
                detections.append(DetectionResult(
                    bbox=[max(0, x1 / w - padding), max(0, y1 / h - padding),
                          min(1, x2 / w + padding), min(1, y2 / h + padding)],
                    class_name="scratches",
                    confidence=min(0.92, 0.55 + len(long_lines) * 0.04),
                ))

            # 短线簇 -> 裂纹
            if len(short_lines) >= 5:
                xs = [l[0][0] for l in short_lines] + [l[0][2] for l in short_lines]
                ys = [l[0][1] for l in short_lines] + [l[0][3] for l in short_lines]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                padding = 0.03
                detections.append(DetectionResult(
                    bbox=[max(0, x1 / w - padding), max(0, y1 / h - padding),
                          min(1, x2 / w + padding), min(1, y2 / h + padding)],
                    class_name="crazing",
                    confidence=min(0.90, 0.55 + len(short_lines) * 0.03),
                ))

        return detections

    def _detect_texture(self, gray: np.ndarray, shape: tuple) -> list:
        """纹理分析 - 找斑块/麻点/氧化皮"""
        h, w = shape
        detections = []

        # 局部标准差 (纹理异常区域)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        std_diff = cv2.absdiff(gray, blurred)
        std_diff = cv2.GaussianBlur(std_diff, (5, 5), 0)

        # 自适应阈值找异常区域
        _, binary = cv2.threshold(std_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学操作去噪
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = (w * h) * 0.0005  # 至少 0.05% 面积
        texture_regions = [c for c in contours if cv2.contourArea(c) > min_area]

        if len(texture_regions) > 0:
            for contour in texture_regions[:6]:
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cw * ch
                total_area = w * h
                ratio = cw / max(ch, 1)

                if area < total_area * 0.02:
                    # 小区域
                    if ratio < 0.5 or ratio > 2.0:
                        def_type = "scratches"
                    else:
                        def_type = "pitted_surface"
                elif area < total_area * 0.08:
                    def_type = "patches"
                else:
                    def_type = "rolled-in_scale"

                detections.append(DetectionResult(
                    bbox=[max(0, x / w - 0.01), max(0, y / h - 0.01),
                          min(1, (x + cw) / w + 0.01), min(1, (y + ch) / h + 0.01)],
                    class_name=def_type,
                    confidence=0.72,
                ))

        return detections

    def _detect_dark_regions(self, gray: np.ndarray, shape: tuple) -> list:
        """暗色区域检测 - 找夹杂/氧化皮"""
        h, w = shape
        detections = []

        # 自适应阈值找暗色区域
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 形态学处理
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = (w * h) * 0.001
        dark_regions = [c for c in contours if cv2.contourArea(c) > min_area]

        if len(dark_regions) > 0:
            for contour in dark_regions[:5]:
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cw * ch
                total_area = w * h

                if area < total_area * 0.02:
                    def_type = "inclusion"
                else:
                    def_type = "rolled-in_scale"

                detections.append(DetectionResult(
                    bbox=[max(0, x / w - 0.01), max(0, y / h - 0.01),
                          min(1, (x + cw) / w + 0.01), min(1, (y + ch) / h + 0.01)],
                    class_name=def_type,
                    confidence=0.68,
                ))

        return detections

    def _detect_brightness_anomaly(self, gray: np.ndarray, shape: tuple) -> list:
        """全局亮度异常检测 - 大面积异常区域"""
        h, w = shape
        detections = []

        mean_val = np.mean(gray)
        std_val = np.std(gray)

        # 亮度过低或过高 -> 可能有氧化皮或斑块
        if mean_val < 80 or mean_val > 180:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_pixels = cv2.countNonZero(binary)
            ratio = white_pixels / (w * h)

            if ratio > 0.3 and ratio < 0.7:
                detections.append(DetectionResult(
                    bbox=[0.05, 0.05, 0.95, 0.95],
                    class_name="patches",
                    confidence=0.60,
                ))

        return detections

    def _merge_detections(self, detections: list) -> list:
        """合并重叠的检测框"""
        if not detections:
            return []

        # 按置信度排序
        detections.sort(key=lambda d: d.confidence, reverse=True)

        merged = []
        used = set()

        for i, det_i in enumerate(detections):
            if i in used:
                continue
            merged.append(det_i)
            for j, det_j in enumerate(detections):
                if j <= i or j in used:
                    continue
                iou = bbox_iou(det_i.bbox, det_j.bbox)
                if iou > 0.5 and det_i.class_name == det_j.class_name:
                    used.add(j)

        return merged



# ============================================================
# VLM 检测器 (云端API + 本地回退)
# ============================================================

class VLMDetector(BaseDetector):
    """视觉大模型缺陷检测器 (云端API + 本地离线分析)"""

    SYSTEM_PROMPT = """你是一名资深的钢铁表面质量检测工程师。
仔细检查钢板表面图像，找出所有可能的质量缺陷。

## 缺陷类型
1. 裂纹 (crazing)：深色锯齿状网状裂缝，通常由热应力引起
2. 划痕 (scratches)：浅色细长条痕，沿轧制方向或随机分布
3. 氧化皮 (rolled-in_scale)：不规则暗色斑块，氧化皮被压入表面
4. 夹杂 (inclusion)：非金属夹杂物，暗色不规则斑点
5. 斑块 (patches)：色差明显的大面积区域，由氧化不均或污染造成
6. 麻点 (pitted_surface)：密集分布的小凹坑，由轧辊表面粗糙或腐蚀引起

## 返回格式
严格返回 JSON，不要其他文字：
{"detections": [{"class_name": "crazing|scratches|rolled-in_scale|inclusion|patches|pitted_surface", "confidence": 0.92, "bbox_description": "位置描述", "severity": "minor|moderate|severe"}]}
无缺陷返回 {"detections": []}"""

    def __init__(self, api_base=None, model=None, max_tokens=2048, temperature=0.1,
                 timeout=8, max_retries=0, jpeg_quality=85, max_image_size=1024):
        super().__init__(name="vlm")
        self._provider, self.api_key, self.api_base, self.provider_name = _detect_provider(api_base)

        if model:
            self.model = model
        elif self._provider in _PROVIDERS:
            self.model = _PROVIDERS[self._provider]["default_model"]
        else:
            self.model = "qwen-vl-max"

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.jpeg_quality = jpeg_quality
        self.max_image_size = max_image_size
        self._client = None
        self._local_analyzer = LocalAnalyzer()
        self._mode = "unknown"  # "api" | "local" | "unknown"
        self._mode_reason = ""

    def load_model(self, model_path=None, **kwargs):
        """初始化 API 客户端或本地分析器"""
        if self.api_key:
            # 快速连通性检测 - 仅对 Gemini 做 (国内可能被墙)
            if self._provider == "gemini":
                reachable = _quick_connectivity_check("generativelanguage.googleapis.com", timeout=3.0)
                if not reachable:
                    self._mode = "local"
                    self._mode_reason = "Gemini API 不可达 (国内网络限制)，已自动切换为本地离线分析"
                    logger.info(self._mode_reason)
                    self._warm = True
                    return

            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
                self._mode = "api"
                self._mode_reason = f"云端API: {self.provider_name} / {self.model}"
                logger.info(f"VLM 就绪: {self._mode_reason}")
            except Exception as e:
                self._mode = "local"
                self._mode_reason = f"API 客户端初始化失败: {e}"
                logger.warning(self._mode_reason)
                logger.info("VLM 回退到本地离线分析模式")
        else:
            self._mode = "local"
            self._mode_reason = "未配置 API Key，使用本地离线分析"
            logger.info(self._mode_reason)

        self._warm = True

    def detect(self, image: np.ndarray) -> InferenceResult:
        """对单张图像执行检测 (API优先，失败回退本地)"""
        start = time.perf_counter()

        if self._mode == "api" and self._client is not None:
            result = self._detect_via_api(image, start)
            if result.error is None:
                return result
            # API 失败，仅本次回退本地，不永久切换模式
            logger.warning(f"VLM API 失败: {result.error}，本次回退到本地分析")

        # 本地离线分析
        result = self._detect_local(image, start)
        return result

    def _detect_via_api(self, image: np.ndarray, start: float) -> InferenceResult:
        """通过云 API 检测"""
        try:
            img_base64 = self._image_to_base64(image)
            messages = self._build_messages(img_base64)
            response = self._call_api(messages)

            raw_text = response.choices[0].message.content or "{}"
            parsed = self._parse_response(raw_text)
            validated = self._validate_response(parsed)

            elapsed = (time.perf_counter() - start) * 1000
            detections = self._build_detections(validated)

            return InferenceResult(
                detections=detections,
                inference_time_ms=elapsed,
                image_shape=image.shape[:2],
                raw_output={
                    "vlm_raw_response": validated,
                    "provider": self._provider,
                    "model": self.model,
                    "mode": "api",
                },
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            error_msg = f"{type(e).__name__}: {e}"
            return InferenceResult(inference_time_ms=elapsed, error=error_msg)

    def _detect_local(self, image: np.ndarray, start: float) -> InferenceResult:
        """本地离线分析"""
        result = self._local_analyzer.analyze(image)
        if result.raw_output is None:
            result.raw_output = {}
        result.raw_output["mode"] = "local"
        result.raw_output["provider"] = "本地离线分析"
        result.raw_output["mode_reason"] = self._mode_reason
        return result

    def _call_api(self, messages):
        """调用 API (单次调用，不重试)"""
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def _build_messages(self, img_base64):
        user_content = [
            {"type": "text", "text": "请仔细检查这张钢板表面图像，找出所有缺陷。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
        ]

        if self._provider == "gemini":
            return [{"role": "user", "content": [{"type": "text", "text": self.SYSTEM_PROMPT}, *user_content]}]

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _image_to_base64(self, image: np.ndarray) -> str:
        h, w = image.shape[:2]
        max_dim = self.max_image_size
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        if image.ndim == 2:
            pil_img = Image.fromarray(image, mode="L").convert("RGB")
        else:
            pil_img = Image.fromarray(image[..., ::-1], mode="RGB")

        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _parse_response(self, text: str) -> dict:
        if not text or not text.strip():
            return {"detections": []}

        text = text.strip()
        original = text

        # 提取 JSON 代码块
        for marker in ["```json", "```"]:
            if marker in text:
                parts = text.split(marker)
                if len(parts) > 1:
                    text = parts[1].split("```")[0].strip()
                    break

        # 提取 JSON 对象
        candidates = []
        start_idx = text.find("{")
        while start_idx != -1:
            brace_count = 0
            for i in range(start_idx, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        candidates.append(text[start_idx:i + 1])
                        break
            start_idx = text.find("{", start_idx + 1)

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        try:
            fixed = self._repair_truncated_json(text)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        for t in [text, original]:
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                pass

        logger.warning(f"VLM响应解析失败: {original[:200]}")
        return {"detections": [], "raw": original, "parse_error": True}

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        start = text.find("{")
        if start == -1:
            return "{}"
        end = len(text) - 1
        while end >= start and text[end] not in "}]\",":
            end -= 1
        if end < start:
            return "{}"
        text = text[start:end + 1]
        if text.rstrip().endswith(","):
            text = text.rstrip()[:-1]
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        text += "]" * max(0, open_brackets)
        text += "}" * max(0, open_braces)
        return text

    def _validate_response(self, parsed: dict) -> dict:
        if not isinstance(parsed, dict):
            return {"detections": []}

        detections = parsed.get("detections")
        if detections is None:
            for key in ["detection", "results", "result", "defects", "objects"]:
                if key in parsed:
                    detections = parsed[key]
                    break

        if not isinstance(detections, list):
            return {"detections": []}

        validated = []
        for item in detections:
            if not isinstance(item, dict):
                continue

            class_name = str(item.get("class_name", "unknown")).lower().strip()
            if class_name not in _VALID_DEFECT_CLASSES:
                matched = self._fuzzy_match_class(class_name)
                class_name = matched if matched else "unknown"

            confidence = item.get("confidence", 0.8)
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (ValueError, TypeError):
                confidence = 0.8

            severity = str(item.get("severity", "moderate")).lower()
            if severity not in _VALID_SEVERITIES:
                severity = "moderate"

            validated.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox_description": str(item.get("bbox_description", "")),
                "severity": severity,
            })

        return {"detections": validated}

    @staticmethod
    def _fuzzy_match_class(class_name: str) -> Optional[str]:
        mapping = {
            "crazing": ["crack", "裂纹", "裂缝", "裂", "cracks", "crazing", "龟裂"],
            "scratches": ["scratch", "划痕", "划伤", "擦痕", "scratches"],
            "rolled-in_scale": ["scale", "氧化皮", "氧化", "锈", "rust", "鳞片", "rolled-in_scale"],
            "inclusion": ["inclusion", "夹杂", "夹杂物", "inclusions"],
            "patches": ["patches", "斑块", "斑", "patch", "色斑", "色差"],
            "pitted_surface": ["pitted", "麻点", "凹坑", "坑", "pit", "pits", "pitted_surface"],
        }
        for canonical, aliases in mapping.items():
            if any(alias in class_name for alias in aliases):
                return canonical
        return None

    def _build_detections(self, parsed: dict) -> list[DetectionResult]:
        detections = []
        for item in parsed.get("detections", []):
            base_conf = float(item.get("confidence", 0.8))
            severity = item.get("severity", "moderate")
            severity_boost = {"minor": -0.05, "moderate": 0.0, "severe": 0.05}
            adjusted_conf = max(0.0, min(1.0, base_conf + severity_boost.get(severity, 0.0)))
            detections.append(DetectionResult(
                bbox=[0.0, 0.0, 1.0, 1.0],
                class_name=item.get("class_name", "unknown"),
                confidence=adjusted_conf,
            ))
        return detections

    @property
    def mode_info(self) -> str:
        """返回当前分析模式信息"""
        return self._mode_reason or ("云端API" if self._mode == "api" else "本地离线分析")