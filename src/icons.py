"""
SVG 矢量图标库 - 钢铁表面缺陷检测系统。

所有图标为 24x24 viewBox，通过 stroke="currentColor" 继承文字颜色。
使用方法: from icons import ICON_XXX; gr.HTML(ICON_XXX)
"""

# --- 导航/标题图标 ---

ICON_SEARCH = """<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="#1a73e8" stroke-width="2" style="vertical-align:middle;margin-right:4px"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>"""

ICON_CHART_UP = """<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:2px"><path d="M23 6l-9.5 9.5-5-5L1 18"/><path d="M17 6h6v6"/></svg>"""

ICON_BAR_CHART = """<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:2px"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>"""

ICON_CHECK_CIRCLE = """<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:2px"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>"""

# --- 操作按钮图标 ---

ICON_CAMERA = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:2px"><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><circle cx="12" cy="13" r="4"/></svg>"""

ICON_PLAY = """<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" style="vertical-align:middle;margin-right:2px"><path d="M8 5v14l11-7z"/></svg>"""

ICON_SAVE = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:2px"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><path d="M17 21v-8H7v8M7 3v5h8"/></svg>"""

ICON_REFRESH = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:2px"><path d="M23 4v6h-6M1 20v-6h6"/><path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/></svg>"""

ICON_EDIT = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:middle;margin-right:2px"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>"""

# --- 状态图标 ---

ICON_SUCCESS = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#16a34a" stroke-width="2.5" style="vertical-align:middle;margin-right:4px"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>"""

ICON_INFO = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#1a73e8" stroke-width="2" style="vertical-align:middle;margin-right:4px"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>"""

ICON_WARN = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#d97706" stroke-width="2" style="vertical-align:middle;margin-right:4px"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><path d="M12 9v4M12 17h.01"/></svg>"""

# --- 产品 Logo (组合图标) ---

ICON_LOGO = """<svg viewBox="0 0 40 40" width="32" height="32" fill="none" style="vertical-align:middle;margin-right:8px">
  <rect x="2" y="2" width="36" height="36" rx="8" fill="#1a237e"/>
  <circle cx="16" cy="16" r="8" stroke="#64b5f6" stroke-width="2.5" fill="none"/>
  <path d="M30 30l-6-6" stroke="#64b5f6" stroke-width="2.5" stroke-linecap="round"/>
  <rect x="8" y="26" width="6" height="10" rx="1" fill="#64b5f6" opacity="0.6"/>
  <rect x="17" y="22" width="6" height="14" rx="1" fill="#64b5f6" opacity="0.8"/>
  <rect x="26" y="18" width="6" height="18" rx="1" fill="#64b5f6"/>
</svg>"""


def icon_button(label: str, svg_icon: str) -> str:
    """生成带 SVG 图标的按钮 HTML (用于 gr.HTML 嵌入)"""
    return f"""<div style="display:inline-flex;align-items:center;gap:6px;padding:8px 16px;
        background:#1a73e8;color:white;border-radius:6px;cursor:pointer;
        font-size:14px;font-weight:500;border:none;transition:background 0.2s">
        {svg_icon}{label}</div>"""
