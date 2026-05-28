"""
VLM 检测引擎 - 基于视觉大模型的钢铁表面缺陷精细分析。

支持: Google Gemini (免费) / 阿里 Qwen-VL / 任意 OpenAI 兼容接口
用于 YOLO 低置信度/未知类别的复核分析。
"""

import json
import os
import time
from typing import Optional
import base64
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
from openai import OpenAI

from .base_detector import BaseDetector, InferenceResult, DetectionResult

# 支持的 API 提供商配置
_PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "default_model": "qwen-vl-max",
    },
}


def _detect_provider(api_base: Optional[str] = None) -> tuple[str, str, str, str]:
    """自动检测可用的 VLM 提供商 -> (provider_name, api_key, base_url, model)"""
    # 优先级1: VLM_API_KEY (自定义 OpenAI 兼容接口)
    vlm_key = os.getenv("VLM_API_KEY", "")
    if vlm_key:
        vlm_base = os.getenv("VLM_BASE_URL", "")
        vlm_model = os.getenv("VLM_MODEL", "gemini-2.5-flash")
        return ("custom", vlm_key, vlm_base, vlm_model)

    if api_base:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
        return ("custom", api_key, api_base, "gemini-2.5-flash")

    # 优先级2: Gemini 免费
    for name, cfg in _PROVIDERS.items():
        key = os.getenv(cfg["env_key"], "")
        if key:
            return (name, key, cfg["base_url"], cfg["default_model"])

    return ("none", "", "", "")


class VLMDetector(BaseDetector):
    """视觉大模型缺陷检测器 (多提供商)
    
    注意: SYSTEM_PROMPT 约 3KB (~750 tokens)，每次 API 调用都会发送。
    按 Gemini Flash 定价 (~$0.075/1M input tokens)，单次调用 prompt 成本约 $0.00006。
    高频场景建议配合 FastScreener 预筛选减少 VLM 调用次数。
    """

    # 优化后的钢铁缺陷检测提示词 (高灵敏度版 v2)
    SYSTEM_PROMPT = """你是钢铁表面缺陷检测 AI，任务是在钢板图像中精确定位并描述所有可见缺陷。

## 检测指令

1. 扫描整张图像，视距覆盖全局和局部细节
2. 钢板正常表面为均匀灰色金属纹理，任何偏离此特征的都是缺陷
3. 特别关注：细线、斑块、色差、凹陷、凸起、反光异常
4. 评估每个缺陷的严重程度: minor(轻微)、moderate(中等)、severe(严重)

## 七类缺陷定义

| 类型 | 视觉特征 | 典型形态 |
|------|---------|---------|
| rust 锈蚀 | 红棕色/橙黄色/黄褐色区域 | 不规则斑块或点状分布，颜色明显偏暖 |
| scratch 划痕 | 细长浅色/深色线状 | 直线或微弯，宽度<2mm |
| crack 裂纹 | 深色不规则裂缝 | 锯齿边缘，宽度不一 |
| scale 氧化皮 | 暗色不规则斑块 | 边缘模糊，颜色深灰/黑 |
| indentation 压痕 | 圆形/椭圆凹陷 | 边缘清晰，局部变暗 |
| blister 气泡 | 圆形凸起 | 中心亮、边缘暗的圆斑 |
| patches 斑块 | 大面积不均匀 | 颜色深浅交替区域 |

rust 锈蚀识别要点：
- 颜色是关键信号：铁锈呈红棕色、橙黄色、黄褐色，与灰色钢面形成强烈对比
- 可呈片状、点状、沿边缘蔓延状
- 常见于钢板边缘、受潮区域、存放时间长的钢材表面
- 即使是轻微泛黄的氧化迹象也应标注为 rust

## 坐标描述规范

bbox_description 必须包含位置和形态：
- 位置：左上角/右上角/中央/左下角/右下角/顶部/底部/左侧/右侧
- 形态：水平/垂直/斜向，大致尺寸（小/中/大/细长）
- 示例："右侧中部，垂直细长白色划痕，长约图像高度30%"

## 输出 (仅 JSON，无 Markdown)

{"detections":[{"class_name":"rust","confidence":0.95,"bbox_description":"右下角大面积红棕色锈蚀斑块","severity":"severe"},{"class_name":"scratch","confidence":0.88,"bbox_description":"左上角斜向浅色划痕","severity":"minor"}]}

无缺陷时返回: {"detections":[]}"""

    def __init__(
        self,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        timeout: int = 30,
        max_retries: int = 3,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(name="vlm")
        self._provider, self.api_key, self.api_base, auto_model = _detect_provider(api_base)

        # 用户指定模型 > 自动检测模型 > 默认
        if model:
            self.model = model
        elif auto_model:
            self.model = auto_model
        elif self._provider in _PROVIDERS:
            self.model = _PROVIDERS[self._provider]["default_model"]
        else:
            self.model = "gemini-2.5-flash"

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[OpenAI] = None
        self._system_prompt = system_prompt or self.SYSTEM_PROMPT

    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """初始化 API 客户端"""
        if not self.api_key:
            raise ValueError(
                "请设置 VLM API Key 环境变量:\n"
                "  Gemini (免费): GEMINI_API_KEY\n"
                "  阿里百炼:     DASHSCOPE_API_KEY"
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        print(f"[INFO] VLM 引擎就绪: {self._provider} / {self.model}")
        self._warm = True

    def detect(self, image: np.ndarray) -> InferenceResult:
        """对单张图像执行 VLM 检测 (含图像增强)"""
        if self._client is None:
            return InferenceResult(error="VLM 客户端未初始化，请先调用 load_model()")

        start = time.perf_counter()

        try:
            # 图像预处理：增强对比度，让缺陷更明显
            enhanced = self._enhance_for_defects(image)
            img_base64 = self._image_to_base64(enhanced)

            # 使用 system prompt + user image
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请仔细检查这张钢板表面图像，找出所有缺陷。"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                        ],
                    },
                ],
                max_tokens=self.max_tokens,
                temperature=0.1,  # 微调温度提高灵敏度 (0.0 太保守)
            )

            raw_text = response.choices[0].message.content or "{}"
            parsed = self._parse_response(raw_text)

        except Exception as e:
            return InferenceResult(
                inference_time_ms=self._measure_time(start),
                error=f"VLM API 调用异常: {e}",
            )

        elapsed = self._measure_time(start)
        detections = self._build_detections(parsed)

        return InferenceResult(
            detections=detections,
            inference_time_ms=elapsed,
            image_shape=image.shape[:2],
            raw_output={"vlm_raw_response": parsed},
        )

    def _enhance_for_defects(self, image: np.ndarray) -> np.ndarray:
        """图像预处理：CLAHE 自适应直方图均衡化 + 锐化，增强缺陷可见性"""
        # 转灰度做 CLAHE 增强
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # 合并回 BGR：用增强后的灰度替换 V 通道
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = enhanced_gray  # 替换亮度通道
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 轻微锐化
        kernel = np.array([[-0.5, -0.5, -0.5],
                           [-0.5,  5.0, -0.5],
                           [-0.5, -0.5, -0.5]], dtype=np.float32)
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        return enhanced

    def _image_to_base64(self, image: np.ndarray) -> str:
        """将 numpy 图像编码为 base64 JPEG"""
        if image.shape[-1] == 3:
            pil_img = Image.fromarray(image[..., ::-1])  # BGR -> RGB
        else:
            pil_img = Image.fromarray(image)

        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()

    def _parse_response(self, text: str) -> dict:
        """解析 VLM 返回的 JSON (增强容错)"""
        text = text.strip()
        # 提取 JSON 代码块
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # 预处理：修复常见的 JSON 格式问题
        text = self._sanitize_json_text(text)

        # 尝试多种方式提取 JSON 对象
        for extractor in [
            lambda t: t[t.find("{"):t.rfind("}")+1],  # 提取最外层 {...}
            lambda t: t,
        ]:
            try:
                extracted = extractor(text)
                return json.loads(extracted)
            except (json.JSONDecodeError, ValueError):
                continue

        # 最后尝试：修复截断的 JSON
        try:
            fixed = self._repair_truncated_json(text)
            return json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            pass

        return {"detections": [], "raw": text}

    @staticmethod
    def _sanitize_json_text(text: str) -> str:
        """修复 VLM 常见的 JSON 格式问题"""
        import re
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 策略: 将 JSON 字符串值内部的换行替换为空格
        # 保护引号内的内容，把换行替换掉
        result = []
        in_string = False
        escape_next = False
        for ch in text:
            if escape_next:
                result.append(ch)
                escape_next = False
                continue
            if ch == '\\':
                result.append(ch)
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                result.append(ch)
                continue
            if in_string and ch == '\n':
                result.append(' ')  # 字符串内换行→空格
                continue
            result.append(ch)
        text = ''.join(result)

        # 1. 修复被换行打断的键名: "confi\ndence" -> "confidence" (现在不会匹配引号内)
        # 已由上面处理

        # 2. 修复被换行打断的数字: "0\n.98" -> "0.98"
        text = re.sub(r'(\d)\s*\n\s*\.(\d)', r'\1.\2', text)

        # 3. 移除键值分隔符前后的换行
        text = re.sub(r':\s*\n\s*', ': ', text)

        # 4. 移除尾随逗号
        text = re.sub(r',\s*(\]|\})', r'\1', text)

        # 5. 补全缺失的闭合括号
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        while text.count('{') < text.count('}'):
            idx = text.rfind('}')
            text = text[:idx] + text[idx+1:]
        text += ']' * max(0, open_brackets)
        text += '}' * max(0, open_braces)

        # 6. 移除注释
        text = re.sub(r'//[^\n]*', '', text)

        return text

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        """尝试修复被截断的 JSON"""
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1:
            return "{}"
        text = text[start:end+1] if end >= start else text[start:]

        # 补全未闭合的括号
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        # 移除最后不完整的键值对
        last_comma = text.rfind(",")
        if last_comma > text.rfind("}"):
            text = text[:last_comma]
        # 补全括号
        text += "]" * max(0, open_brackets)
        text += "}" * max(0, open_braces)
        return text

    def _build_detections(self, parsed: dict) -> list[DetectionResult]:
        """将 VLM 解析结果转换为标准 DetectionResult"""
        detections = []
        for item in parsed.get("detections", []):
            detections.append(DetectionResult(
                bbox=[0.0, 0.0, 1.0, 1.0],  # VLM 不提供精确 bbox
                class_name=item.get("class_name", "unknown"),
                confidence=float(item.get("confidence", 0.8)),
            ))
        return detections
