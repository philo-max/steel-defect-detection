"""
图像增强预处理模块 - 提升缺陷检测准确率

功能:
1. CLAHE 自适应直方图均衡化
2. 缺陷区域对比度增强
3. 噪声抑制与边缘锐化
4. 多尺度增强融合
"""

import cv2
import numpy as np


class ImageEnhancer:
    """钢铁表面缺陷图像增强器"""

    def __init__(
        self,
        enable_clahe: bool = True,
        enable_sharpen: bool = True,
        enable_denoise: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: tuple = (8, 8),
        sharpen_strength: float = 1.0,
    ):
        self.enable_clahe = enable_clahe
        self.enable_sharpen = enable_sharpen
        self.enable_denoise = enable_denoise
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.sharpen_strength = sharpen_strength

        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size,
        )

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        对输入图像执行完整的增强流程

        Args:
            image: BGR 格式的 numpy 图像 (H, W, 3)

        Returns:
            增强后的图像
        """
        if image is None or image.size == 0:
            return image

        enhanced = image.copy()

        # 1. 去噪 (保护边缘)
        if self.enable_denoise:
            enhanced = self._denoise(enhanced)

        # 2. CLAHE 增强 (LAB 色彩空间，保护色调)
        if self.enable_clahe:
            enhanced = self._apply_clahe(enhanced)

        # 3. 锐化 (增强缺陷边缘)
        if self.enable_sharpen:
            enhanced = self._sharpen(enhanced)

        # 4. 多尺度融合 (保留原图细节 + 增强后的对比度)
        enhanced = self._multi_scale_fusion(image, enhanced)

        return enhanced

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """非局部均值去噪，保护边缘"""
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=3,        # 亮度分量滤波强度
            hColor=3,   # 色度分量滤波强度
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """在 LAB 色彩空间应用 CLAHE，增强亮度对比度"""
        # 转换到 LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 对 L 通道应用 CLAHE
        l_enhanced = self._clahe.apply(l)

        # 合并通道
        lab_enhanced = cv2.merge([l_enhanced, a, b])

        # 转换回 BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Unsharp Mask 锐化"""
        # 高斯模糊
        blurred = cv2.GaussianBlur(image, (0, 0), 3)

        # Unsharp mask: original + strength * (original - blurred)
        sharpened = cv2.addWeighted(
            image, 1.0 + self.sharpen_strength,
            blurred, -self.sharpen_strength,
            0,
        )

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _multi_scale_fusion(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        alpha: float = 0.7,
    ) -> np.ndarray:
        """
        多尺度融合：保留原图细节 + 增强后的对比度

        Args:
            original: 原图
            enhanced: 增强后的图像
            alpha: 增强图像的权重 (0-1)
        """
        # 简单加权融合
        fused = cv2.addWeighted(original, 1 - alpha, enhanced, alpha, 0)

        # 进一步：提取高频细节 (原图) + 低频对比度 (增强图)
        # 使用拉普拉斯金字塔融合
        try:
            # 高斯金字塔
            g0_orig = original.astype(np.float32) / 255.0
            g0_enh = enhanced.astype(np.float32) / 255.0

            # 一层拉普拉斯
            g1_orig = cv2.pyrDown(g0_orig)
            g1_enh = cv2.pyrDown(g0_enh)

            l0_orig = g0_orig - cv2.pyrUp(g1_orig, dstsize=(original.shape[1], original.shape[0]))
            l0_enh = g0_enh - cv2.pyrUp(g1_enh, dstsize=(enhanced.shape[1], enhanced.shape[0]))

            # 融合：高频用原图 (保留细节)，低频用增强图 (提升对比度)
            l0_fused = l0_orig * 0.6 + l0_enh * 0.4
            g1_fused = g1_enh  # 低频用增强图

            # 重建
            fused_pyr = (cv2.pyrUp(g1_fused, dstsize=(original.shape[1], original.shape[0])) + l0_fused)
            fused_pyr = np.clip(fused_pyr * 255, 0, 255).astype(np.uint8)

            return fused_pyr

        except Exception:
            # 金字塔融合失败，回退到简单加权
            return fused

    def quick_enhance(self, image: np.ndarray) -> np.ndarray:
        """快速增强模式 (用于实时检测)"""
        if image is None or image.size == 0:
            return image

        # LAB CLAHE only (fastest)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self._clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def enhance_for_defect_detection(image: np.ndarray, mode: str = "standard") -> np.ndarray:
    """
    便捷的缺陷检测图像增强函数

    Args:
        image: 输入图像 (BGR)
        mode: "standard" (标准) | "quick" (快速) | "aggressive" (激进增强)

    Returns:
        增强后的图像
    """
    if mode == "quick":
        enhancer = ImageEnhancer(
            enable_clahe=True,
            enable_sharpen=False,
            enable_denoise=False,
        )
        return enhancer.quick_enhance(image)

    elif mode == "aggressive":
        enhancer = ImageEnhancer(
            enable_clahe=True,
            enable_sharpen=True,
            enable_denoise=True,
            clahe_clip_limit=3.0,
            sharpen_strength=1.5,
        )
        return enhancer.enhance(image)

    else:  # standard
        enhancer = ImageEnhancer(
            enable_clahe=True,
            enable_sharpen=True,
            enable_denoise=True,
        )
        return enhancer.enhance(image)
