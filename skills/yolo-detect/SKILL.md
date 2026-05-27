---
name: yolo-detect
description: 调用 YOLO 模型对钢铁表面图像进行缺陷检测，返回缺陷类型、位置和置信度。
---

# YOLO 缺陷检测 Skill

## 功能

调用 YOLO 目标检测模型，对输入的钢铁表面图像进行实时缺陷检测与分类。

## 输入

- `image_path` (str): 待检测图像的本地路径或 numpy 数组

## 输出

返回 JSON 格式的检测结果：

```json
{
  "detections": [
    {
      "class_name": "crack",
      "confidence": 0.92,
      "bbox": [0.1, 0.2, 0.5, 0.6]
    }
  ],
  "inference_time_ms": 45.2,
  "defect_count": 1
}
```

## 支持的缺陷类型

- crack (裂纹)
- scratch (划痕)
- scale (氧化皮)
- indentation (压痕)
- blister (气泡)

## 使用方式

```python
from skills.yolo_detect.detect import detect

result = detect("path/to/steel_image.jpg")
print(result)
```

## 注意事项

- 模型需预先下载到 `models/weights/` 目录
- 首次推理会自动预热
- 单张推理延迟 < 200ms (RTX 4060)
