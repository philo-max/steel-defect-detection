"""
缺陷严重度评估工具 — YOLO/VLM 检测结果的分级逻辑。

严重度分级:
  A = 严重 (面积大或置信度极高)
  B = 中等偏重
  C = 轻度
  D = 轻微/可忽略
"""


def compute_severity(bbox: list[float], confidence: float) -> str:
    """
    根据边界框面积占比和置信度计算严重度等级。

    Args:
        bbox: 归一化边界框 [x1, y1, x2, y2]
        confidence: 检测置信度 [0, 1]

    Returns:
        "A" | "B" | "C" | "D"
    """
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    score = area * 0.6 + confidence * 0.4

    if score >= 0.45 or confidence >= 0.85:
        return "A"
    elif score >= 0.25 or confidence >= 0.70:
        return "B"
    elif score >= 0.10 or confidence >= 0.40:
        return "C"
    else:
        return "D"


def compute_overall_status(severity_index: float) -> str:
    """
    根据严重度指数计算总体判定。

    Args:
        severity_index: 严重度指数 (0-100)

    Returns:
        "Pass" | "Marginal" | "Fail"
    """
    if severity_index > 75:
        return "Fail"
    elif severity_index > 30:
        return "Marginal"
    else:
        return "Pass"


def compute_severity_index(detections: list[dict], defect_density: float) -> int:
    """
    根据检测列表和缺陷密度计算综合严重度指数。

    Args:
        detections: 检测结果列表，每项必须包含 severity 字段
        defect_density: 缺陷面积密度百分比

    Returns:
        严重度指数 (0-100)
    """
    if not detections:
        return 4

    max_sev = "D"
    for d in detections:
        sev = d.get("severity", "D")
        if sev == "A":
            max_sev = "A"
        elif sev == "B" and max_sev not in ("A",):
            max_sev = "B"
        elif sev == "C" and max_sev not in ("A", "B"):
            max_sev = "C"

    base_score = {"A": 90, "B": 65, "C": 35, "D": 10}[max_sev]
    return min(100, int(base_score + defect_density * 1.5))


def compute_defect_density(detections: list[dict]) -> float:
    """
    根据检测框计算缺陷面积密度。

    Args:
        detections: 检测结果列表，每项必须包含 bbox [y1, x1, y2, x2] (百分比 0-100)

    Returns:
        缺陷密度百分比 (0-100)
    """
    if not detections:
        return 0.0
    density = sum(
        (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) / 10000.0
        for d in detections
    ) * 100.0
    return min(100.0, float(density))
