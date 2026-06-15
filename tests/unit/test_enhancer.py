"""
图像增强模块单元测试。
"""

import numpy as np
import pytest

from src.image_enhancer import ImageEnhancer, enhance_for_defect_detection


@pytest.fixture
def sample_image():
    """生成一张模拟钢板表面的灰色渐变图像"""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    # 添加渐变背景
    for i in range(200):
        val = int(128 + 30 * np.sin(i / 20.0))
        img[i, :, :] = [val, val, val]
    # 添加模拟缺陷 (暗色条纹)
    img[80:90, 100:200, :] = [40, 40, 40]
    # 添加模拟瑕疵 (暗色斑点)
    img[140:145, 50:60, :] = [30, 30, 30]
    img[150:155, 230:240, :] = [35, 35, 35]
    return img


class TestImageEnhancer:
    def test_default_construction(self):
        enhancer = ImageEnhancer()
        assert enhancer.enable_clahe is True
        assert enhancer.enable_sharpen is True
        assert enhancer.enable_denoise is True

    def test_disabled_flags(self):
        enhancer = ImageEnhancer(
            enable_clahe=False,
            enable_sharpen=False,
            enable_denoise=False,
        )
        assert enhancer.enable_clahe is False

    def test_enhance_returns_same_shape(self, sample_image):
        enhancer = ImageEnhancer()
        result = enhancer.enhance(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_enhance_none_image(self):
        enhancer = ImageEnhancer()
        assert enhancer.enhance(None) is None

    def test_enhance_empty_image(self):
        enhancer = ImageEnhancer()
        img = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        result = enhancer.enhance(img)
        assert result.size == 0

    def test_quick_enhance(self, sample_image):
        enhancer = ImageEnhancer()
        result = enhancer.quick_enhance(sample_image)
        assert result.shape == sample_image.shape

    def test_enhance_preserves_value_range(self, sample_image):
        enhancer = ImageEnhancer()
        result = enhancer.enhance(sample_image)
        assert result.min() >= 0
        assert result.max() <= 255


class TestEnhanceForDefectDetection:
    def test_standard_mode(self, sample_image):
        result = enhance_for_defect_detection(sample_image, mode="standard")
        assert result.shape == sample_image.shape

    def test_quick_mode(self, sample_image):
        result = enhance_for_defect_detection(sample_image, mode="quick")
        assert result.shape == sample_image.shape

    def test_aggressive_mode(self, sample_image):
        result = enhance_for_defect_detection(sample_image, mode="aggressive")
        assert result.shape == sample_image.shape

    def test_default_mode(self, sample_image):
        result = enhance_for_defect_detection(sample_image)
        assert result.shape == sample_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
