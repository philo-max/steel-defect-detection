"""
快速筛查模块 — 两阶段检测的第一级。

功能：
1. 频域分析（FFT）检测纹理异常
2. 统计特征（局部均值/方差偏离度）
3. 边缘密度评估（Canny + Laplacian）
4. 轻量级综合评分，快速过滤正常帧

原理：
    钢铁正常表面为均匀灰色金属纹理。任何缺陷（划痕、裂纹、氧化皮等）
    都会在频域、统计量、边缘密度上产生异常信号。
    本模块综合三种信号，给出 0-1 异常分数，低于阈值的帧直接跳过 YOLO。

性能目标：单帧 < 5ms（CPU），过滤率 > 90%（正常帧直接跳过）

使用方式:
    from src.fast_screener import FastScreener
    screener = FastScreener()
    score, is_anomaly = screener.screen(frame)
    if is_anomaly:
        result = yolo.detect(frame)
"""

import time
import threading
from typing import Tuple

import cv2
import numpy as np


class FastScreener:
    """
    轻量级异常筛查器。

    使用三种互补信号综合判断：
    1. FFT 频域能量分布 — 缺陷会引入高频分量
    2. 局部标准差 — 缺陷区域方差显著偏离全局
    3. 边缘密度 — 裂纹、划痕产生密集边缘
    """

    def __init__(
        self,
        fft_threshold: float = 0.35,        # 频域异常阈值
        std_threshold: float = 0.30,         # 标准差异常阈值
        edge_threshold: float = 0.40,        # 边缘密度阈值
        combo_threshold: float = 0.45,       # 综合分数阈值
        resize_to: int = 256,                # 筛查时缩放到的尺寸（加速）
        grid_size: int = 4,                  # 统计分析的网格数 (4x4)
    ):
        self.fft_threshold = fft_threshold
        self.std_threshold = std_threshold
        self.edge_threshold = edge_threshold
        self.combo_threshold = combo_threshold
        self.resize_to = resize_to
        self.grid_size = grid_size

        # 滑动统计（用于自适应阈值）
        self._score_history: list[float] = []  # 最近 100 帧的分数
        self._max_history: int = 100
        self._adaptive_enabled: bool = True
        self._history_lock = threading.Lock()   # 保护 _score_history 线程安全

    # ==================== 主入口 ====================

    def screen(self, image: np.ndarray) -> Tuple[float, bool]:
        """
        筛查单帧图像。

        Args:
            image: BGR 图像 (H, W, 3)

        Returns:
            (anomaly_score, is_anomaly) — score 越高越可能异常
        """
        start = time.perf_counter()

        # 缩放加速
        h, w = image.shape[:2]
        if max(h, w) > self.resize_to:
            scale = self.resize_to / max(h, w)
            small = cv2.resize(image, (int(w * scale), int(h * scale)))
        else:
            small = image

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # 三个信号
        fft_score = self._fft_analyze(gray)
        std_score = self._std_analyze(gray)
        edge_score = self._edge_analyze(gray)

        # 加权综合
        combo = 0.35 * fft_score + 0.30 * std_score + 0.35 * edge_score

        # 自适应阈值更新
        with self._history_lock:
            self._score_history.append(combo)
            if len(self._score_history) > self._max_history:
                self._score_history.pop(0)

        # 动态阈值（锁内计算，避免 _compute_dynamic_threshold 读取 _score_history 时竞态）
        if self._adaptive_enabled:
            with self._history_lock:
                if len(self._score_history) >= 30:
                    dynamic_thresh = self._compute_dynamic_threshold()
                else:
                    dynamic_thresh = self.combo_threshold
        else:
            dynamic_thresh = self.combo_threshold

        is_anomaly = combo > dynamic_thresh

        elapsed = (time.perf_counter() - start) * 1000
        if elapsed > 10:
            print(f"[FastScreener] 警告: 筛查耗时 {elapsed:.1f}ms (目标 <5ms)")

        return combo, is_anomaly

    def screen_batch(self, images: list[np.ndarray]) -> list[Tuple[float, bool]]:
        """批量筛查"""
        return [self.screen(img) for img in images]

    # ==================== 信号分析 ====================

    def _fft_analyze(self, gray: np.ndarray) -> float:
        """
        频域分析 — FFT 高频能量占比。

        原理：正常钢板表面纹理均匀，FFT 能量集中在低频。
              缺陷（裂纹、划痕）引入陡峭边缘 → 高频能量显著增加。
        """
        # 2D FFT
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # 计算低频/高频分界半径
        radius_low = min(cy, cx) * 0.15  # 低频区域半径

        # 创建低频掩码
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        low_mask = dist <= radius_low

        low_energy = np.sum(magnitude[low_mask])
        total_energy = np.sum(magnitude) + 1e-10

        # 高频能量占比 = 1 - 低频能量占比
        high_ratio = 1.0 - (low_energy / total_energy)

        # 归一化到 [0,1]（高频比例通常 0.3-0.7）
        return float(np.clip((high_ratio - 0.3) / 0.5, 0.0, 1.0))

    def _std_analyze(self, gray: np.ndarray) -> float:
        """
        局部标准差分析 — 检测纹理突变。

        原理：将图像分成 grid × grid 的网格，计算每格标准差。
              缺陷区域的局部标准差与全局中位数偏差显著。
        """
        h, w = gray.shape
        gh, gw = h // self.grid_size, w // self.grid_size

        stds = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch = gray[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
                stds.append(float(np.std(patch)))

        stds = np.array(stds)

        # 异常程度 = max(std) / median(std) - 1
        median_std = np.median(stds)
        if median_std < 0.1:
            return 0.0

        deviation = (np.max(stds) / median_std) - 1.0
        # 归一化：偏差 > 2.0 的视为明确异常
        return float(np.clip(deviation / 3.0, 0.0, 1.0))

    def _edge_analyze(self, gray: np.ndarray) -> float:
        """
        边缘密度分析 — 检测密集边缘信号。

        原理：正常钢板表面平滑，Canny 边缘稀疏。
              划痕、裂纹、氧化皮产生密集的边缘像素。
        """
        # Laplacian 方差（模糊度评估）
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(lap.var())

        # Canny 边缘
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size

        # 综合：边缘比例高 + Laplacian 方差大的可能是缺陷
        lap_norm = np.clip(lap_var / 500.0, 0.0, 1.0)
        edge_norm = np.clip(edge_ratio / 0.08, 0.0, 1.0)

        return 0.6 * edge_norm + 0.4 * lap_norm

    # ==================== 自适应阈值 ====================

    def _compute_dynamic_threshold(self) -> float:
        """
        基于历史分数的动态阈值。

        使用 3-sigma 规则：mean + 2.5 * std，适应不同产线环境。
        """
        history = np.array(self._score_history[-50:])
        mean = np.mean(history)
        std = np.std(history)

        dynamic = mean + 2.5 * std
        # 限制阈值范围 [0.35, 0.70]，防止极端情况
        return float(np.clip(dynamic, 0.35, 0.70))

    # ==================== 统计接口 ====================

    @property
    def stats(self) -> dict:
        """返回当前筛查统计"""
        if not self._score_history:
            return {"mean_score": 0, "recent_filter_rate": 0, "samples": 0}

        recent = self._score_history[-50:]
        filter_rate = sum(1 for s in recent if s < self.combo_threshold) / len(recent)
        return {
            "mean_score": round(float(np.mean(recent)), 4),
            "recent_filter_rate": round(filter_rate * 100, 1),
            "samples": len(self._score_history),
        }

    def reset(self):
        """重置历史统计"""
        self._score_history.clear()


# ==================== 筛查模式枚举 ====================

class ScreeningMode:
    """筛查模式"""
    BYPASS = "bypass"          # 跳过筛查，直接 YOLO
    STRICT = "strict"           # 严格模式：高灵敏度，不漏检（稍微多跑 YOLO）
    BALANCED = "balanced"       # 平衡模式：默认
    AGGRESSIVE = "aggressive"   # 激进模式：最大化过滤率（可能轻微增加漏检）


def create_screener(mode: str = "balanced", **kwargs) -> FastScreener:
    """
    根据模式创建筛查器。

    Args:
        mode: "bypass" | "strict" | "balanced" | "aggressive"
    """
    presets = {
        "strict": {
            "fft_threshold": 0.25,
            "std_threshold": 0.20,
            "edge_threshold": 0.30,
            "combo_threshold": 0.35,
        },
        "balanced": {
            "fft_threshold": 0.35,
            "std_threshold": 0.30,
            "edge_threshold": 0.40,
            "combo_threshold": 0.45,
        },
        "aggressive": {
            "fft_threshold": 0.45,
            "std_threshold": 0.40,
            "edge_threshold": 0.50,
            "combo_threshold": 0.55,
        },
    }

    preset = presets.get(mode, presets["balanced"])
    preset.update(kwargs)
    return FastScreener(**preset)


# ==================== 自检 ====================

if __name__ == "__main__":
    import sys

    # 快速自检：正常图像 vs 缺陷图像
    print("FastScreener 自检\n" + "=" * 50)

    screener = create_screener("balanced")

    # 模拟正常钢板图像（均匀灰色）
    normal = np.ones((256, 256), dtype=np.uint8) * 128
    normal = normal + np.random.normal(0, 3, (256, 256)).astype(np.uint8)
    normal_bgr = cv2.cvtColor(normal, cv2.COLOR_GRAY2BGR)

    # 模拟缺陷图像（带划痕）
    defect = normal.copy()
    cv2.line(defect, (50, 100), (200, 110), 30, 2)  # 深色划痕
    cv2.line(defect, (80, 150), (220, 155), 20, 2)
    defect_bgr = cv2.cvtColor(defect, cv2.COLOR_GRAY2BGR)

    score_n, is_a_n = screener.screen(normal_bgr)
    score_d, is_a_d = screener.screen(defect_bgr)

    print(f"正常钢板: score={score_n:.3f}, anomaly={is_a_n} {'✅ 正确过滤' if not is_a_n else '❌ 误报'}")
    print(f"缺陷钢板: score={score_d:.3f}, anomaly={is_a_d} {'✅ 正确检出' if is_a_d else '❌ 漏检'}")

    # 批量测试
    import time
    t0 = time.perf_counter()
    for _ in range(200):
        screener.screen(normal_bgr)
    elapsed = (time.perf_counter() - t0) * 1000 / 200
    print(f"\n平均筛查耗时: {elapsed:.2f}ms/帧 (目标 <5ms)")

    sys.exit(0 if is_a_d and not is_a_n else 1)
