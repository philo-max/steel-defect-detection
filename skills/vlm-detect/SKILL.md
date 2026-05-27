---
name: vlm-detect
description: 调用视觉大模型 (Qwen-VL) 对钢铁表面图像进行精细缺陷分析，用于低置信度或未知缺陷类型的复核。
---

# VLM 缺陷分析 Skill

## 功能

调用 Qwen-VL 视觉大模型 API，对钢铁表面图像进行精细化的缺陷分析。在 YOLO 检测置信度较低或遇到未知缺陷类型时，使用此 Skill 进行二次复核。

## 输入

- `image_path` (str): 待分析图像的本地路径

## 输出

返回 JSON 格式的分析结果：

```json
{
  "detections": [
    {
      "class_name": "crack",
      "confidence": 0.88,
      "bbox_description": "图像右上角区域",
      "severity": "severe"
    }
  ]
}
```

## 前置条件

- 设置环境变量 `DASHSCOPE_API_KEY`
- 网络可访问 Qwen-VL API

## 使用方式

```python
from skills.vlm_detect.detect import analyze

result = analyze("path/to/suspicious_image.jpg")
print(result)
```

## 注意事项

- 单次 API 调用约 2-5 秒
- 有重试机制 (默认 3 次)
- 不提供精确 BBox，仅用于定性分析
