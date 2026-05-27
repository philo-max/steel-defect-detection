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

# --- 行业专业图标 (替换 emoji) ---

# 缺陷检测 - 放大镜+方框
ICON_DETECT = """<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/><rect x="3" y="3" width="5" height="5" rx="0.5" fill="currentColor" opacity="0.15"/></svg>"""

# 审核 - 剪贴板+勾
ICON_REVIEW = """<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2"/><rect x="8" y="2" width="8" height="4" rx="1"/><path d="M9 14l2 2 4-4"/></svg>"""

# 报表/统计
ICON_REPORT = """<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>"""

# 摄像头/采集
ICON_CAMERA_TAB = """<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><circle cx="12" cy="13" r="4"/><circle cx="12" cy="13" r="1.5" fill="currentColor"/></svg>"""

# YOLO 神经网络
ICON_YOLO = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="2"/><circle cx="5" cy="12" r="2"/><circle cx="19" cy="12" r="2"/><circle cx="12" cy="19" r="2"/><line x1="12" y1="7" x2="5" y2="10"/><line x1="12" y1="7" x2="19" y2="10"/><line x1="5" y1="14" x2="12" y2="17"/><line x1="19" y1="14" x2="12" y2="17"/></svg>"""

# VLM / AI 大脑
ICON_VLM = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a4 4 0 00-4 4c0 1.5.8 2.8 2 3.5V12a4 4 0 104 0V9.5c1.2-.7 2-2 2-3.5a4 4 0 00-4-4z"/><path d="M8 15a4 4 0 008 0"/><circle cx="9" cy="4" r="0.5" fill="currentColor"/><circle cx="15" cy="4" r="0.5" fill="currentColor"/></svg>"""

# 全流程/火箭
ICON_FULL = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 15l-3-3m0 0l3-3m-3 3h12M4 21h16a1 1 0 001-1V4a1 1 0 00-1-1H4a1 1 0 00-1 1v16a1 1 0 001 1z"/></svg>"""

# 合格/通过
ICON_PASS = """<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/></svg>"""

# 告警/缺陷
ICON_ALERT = """<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><circle cx="12" cy="17" r="0.5" fill="currentColor"/></svg>"""

# 数据库
ICON_DATABASE = """<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>"""

# 完成/确认
ICON_CONFIRMED = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#16a34a" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="M9 12l2 2 4-4"/></svg>"""

# 驳回
ICON_REJECT = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#dc2626" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>"""

# 刷新
ICON_REFRESH = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 4v6h-6M1 20v-6h6"/><path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/></svg>"""

# 保存
ICON_SAVE = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><path d="M17 21v-8H7v8M7 3v5h8"/></svg>"""

# 知识库/RAG
ICON_RAG = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/><line x1="8" y1="7" x2="16" y2="7"/><line x1="8" y1="11" x2="14" y2="11"/></svg>"""

# 导出
ICON_EXPORT = """<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>"""

# 准确率
ICON_PRECISION = """<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2" fill="currentColor"/></svg>"""

# 灵敏度
ICON_SENSITIVITY = """<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="1" fill="currentColor"/><circle cx="12" cy="12" r="4" opacity="0.6"/><circle cx="12" cy="12" r="7" opacity="0.3"/></svg>"""

# 系统/工业
ICON_SYSTEM = """<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>"""

# 坐标
ICON_COORD = """<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="2" x2="12" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><polyline points="9 8 12 5 15 8"/><polyline points="15 16 12 19 9 16"/></svg>"""

# CPU/GPU
ICON_GPU = """<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/></svg>"""

# Logo
ICON_LOGO = """<svg viewBox="0 0 40 40" width="32" height="32" fill="none">
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
