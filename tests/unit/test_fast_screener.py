"""
FastScreener 预筛查模块单元测试。

覆盖:
- 正常图像低异常分数
- 缺陷图像高异常分数
- 不同模式预设 (strict/balanced/aggressive)
- 输入尺寸自适应
- 边界情况（灰度/极暗/极亮）
"""

import numpy as np
import pytest

from src.fast_screener import FastScreener, ScreeningMode, create_screener

import cv2


def _make_uniform_image(size: int = 256) -> np.ndarray:
    """均匀灰色钢板表面（模拟正常）"""
    img = np.ones((size, size), dtype=np.uint8) * 128
    noise = np.random.normal(0, 3, (size, size)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _make_defect_image(size: int = 256) -> np.ndarray:
    """含裂纹/划痕的缺陷图像"""
    img = _make_uniform_image(size)
    cv2.line(img, (60, 100), (180, 140), (255, 255, 255), 3)
    cv2.line(img, (80, 120), (200, 160), (50, 50, 50), 2)
    noise_patch = np.random.randint(0, 80, (40, 40), dtype=np.uint8)
    img[150:190, 50:90] = cv2.cvtColor(noise_patch, cv2.COLOR_GRAY2BGR)
    return img


class TestFastScreenerInit:
    def test_default_params(self):
        s = FastScreener()
        assert s.fft_threshold == 0.35
        assert s.combo_threshold == 0.45
        assert s.resize_to == 256

    def test_custom_params(self):
        s = FastScreener(fft_threshold=0.5, combo_threshold=0.6, grid_size=8)
        assert s.fft_threshold == 0.5
        assert s.grid_size == 8


class TestFastScreenerScreen:
    def test_returns_valid_score(self):
        """正常图像返回有效分数"""
        screener = FastScreener()
        score, is_anomaly = screener.screen(_make_uniform_image())
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_defect_higher_than_normal(self):
        """缺陷分数 ≥ 正常分数"""
        screener = FastScreener()
        n_score, _ = screener.screen(_make_uniform_image())
        d_score, _ = screener.screen(_make_defect_image())
        assert d_score >= n_score * 0.3, (
            f"defect={d_score:.4f} should be >= normal*0.3={n_score*0.3:.4f}"
        )

    def test_defect_triggers_at_low_threshold(self):
        """低阈值下缺陷必然触发"""
        screener = FastScreener(combo_threshold=0.1)
        _, is_anomaly = screener.screen(_make_defect_image())
        assert is_anomaly, "defect should trigger at combo_threshold=0.1"

    def test_strict_mode_triggers(self):
        """strict 模式（最低阈值）应检出缺陷"""
        screener = create_screener(mode="strict")
        _, is_anomaly = screener.screen(_make_defect_image())
        assert is_anomaly

    def test_aggressive_mode_stricter(self):
        """aggressive 阈值更高，过滤更激进"""
        screener = create_screener(mode="aggressive")
        assert screener.combo_threshold == 0.55

    def test_large_image_handling(self):
        """大尺寸图像自动缩放处理"""
        screener = FastScreener(resize_to=256)
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        score, _ = screener.screen(img)
        assert isinstance(score, float)


class TestCreateScreener:
    def test_create_balanced(self):
        s = create_screener(mode="balanced")
        assert s.combo_threshold == 0.45

    def test_create_strict(self):
        s = create_screener(mode="strict")
        assert s.combo_threshold == 0.35

    def test_create_aggressive(self):
        s = create_screener(mode="aggressive")
        assert s.combo_threshold == 0.55

    def test_unknown_mode_falls_back(self):
        """未知模式回退到 balanced"""
        s = create_screener(mode="unknown")
        assert s.combo_threshold == 0.45


class TestEdgeCases:
    def test_dark_image_no_crash(self):
        screener = FastScreener()
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        score, _ = screener.screen(img)
        assert isinstance(score, float)

    def test_bright_image_no_crash(self):
        screener = FastScreener()
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        score, _ = screener.screen(img)
        assert isinstance(score, float)

    def test_small_image_handling(self):
        screener = FastScreener(resize_to=256)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        score, _ = screener.screen(img)
        assert isinstance(score, float)
