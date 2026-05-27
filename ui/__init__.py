"""UI 模块 - 主题样式、Logo 与辅助函数"""

import base64
from pathlib import Path


def load_css() -> str:
    """加载工业精装主义主题 CSS 文件内容，供 Gradio css= 参数使用"""
    css_path = Path(__file__).parent / "industrial.css"
    return css_path.read_text(encoding="utf-8")


def load_logo_svg(name: str = "logo_b_lattice_scan") -> str:
    """加载 Logo SVG 文件，返回 base64 data URI（适合 <img src> 使用）"""
    logo_path = Path(__file__).parent / f"{name}.svg"
    svg_bytes = logo_path.read_bytes()
    b64 = base64.b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def load_logo_svg_inline(name: str = "logo_b_lattice_scan") -> str:
    """加载 Logo SVG 文件，返回原始 SVG 字符串（适合直接嵌入 HTML）"""
    logo_path = Path(__file__).parent / f"{name}.svg"
    return logo_path.read_text(encoding="utf-8")
