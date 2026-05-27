"""
语音命令处理器 — 钢铁缺陷检测系统。

将自然语言指令映射为系统操作，支持：
- 检测控制: "检测这张图" / "YOLO筛查" / "VLM分析" / "一键全流程"
- 审核操作: "通过第3条" / "驳回记录5" / "刷新待审核"
- 报表导出: "导出今天CSV" / "生成HTML报告" / "导出Bad Case"
- 相机控制: "连接摄像头" / "断开摄像头" / "截取快照"
- 主题切换: "切换深色模式" / "浅色主题"
- 页面导航: "打开检测页" / "去审核页" / "报表页面"

用法:
    from src.voice_commander import VoiceCommander
    vc = VoiceCommander()
    result = vc.execute("YOLO 快速筛查")
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class VoiceCommand:
    """解析后的语音命令"""
    action: str         # 目标操作名
    params: dict = field(default_factory=dict)  # 参数
    raw_text: str = ""  # 原始语音文本
    confidence: float = 1.0


# ==================== 命令路由表 ====================

COMMAND_ROUTES: list[tuple[str, list[str], str]] = [
    # (action_key, [触发短语...], 描述)

    # ---- 检测控制 ----
    ("yolo_detect",  ["YOLO筛查", "YOLO检测", "快速筛查", "快速检测", "YOLO"], "YOLO 快速筛查"),
    ("vlm_analyze",  ["VLM分析", "精细分析", "大模型分析", "V L M", "VLM"], "VLM 精细分析"),
    ("full_pipeline", ["一键全流程", "全流程", "完整检测", "一键检测", "全面分析"], "一键全流程"),
    ("rag_analysis", ["根因分析", "RAG分析", "原因分析", "R A G"], "RAG 根因分析"),

    # ---- 审核操作 ----
    ("review_pass",  ["通过", "审核通过", "确认", "标记通过"]),
    ("review_reject", ["驳回", "拒绝", "修正", "标记误检"]),
    ("review_refresh", ["刷新审核", "刷新列表", "刷新待审核"]),

    # ---- 报表导出 ----
    ("export_report", ["生成报告", "统计报告", "生成报表"]),
    ("export_csv",   ["导出CSV", "CSV导出", "导出表格", "C S V"]),
    ("export_html",  ["导出HTML", "HTML报告", "网页报告", "H T M L"]),
    ("export_badcase", ["导出BadCase", "Bad Case", "错误样本", "导出缺陷样本"]),

    # ---- 相机控制 ----
    ("camera_connect", ["连接摄像头", "开启摄像头", "打开相机", "启动相机"]),
    ("camera_disconnect", ["断开摄像头", "关闭相机", "停止相机"]),
    ("camera_snapshot", ["截取快照", "拍照", "截图", "快照"]),

    # ---- 主题切换 ----
    ("theme_dark",  ["深色模式", "夜间模式", "暗色主题", "切换深色"]),
    ("theme_light", ["浅色模式", "日间模式", "亮色主题", "切换浅色", "浅色主题"]),

    # ---- 页面导航 ----
    ("tab_detect",  ["检测页面", "检测页", "实时检测", "打开检测"]),
    ("tab_review",  ["审核页面", "审核页", "人工审核", "打开审核"]),
    ("tab_report",  ["报表页面", "报表页", "统计报表", "打开报表"]),
    ("tab_camera",  ["相机页面", "采集页面", "实时采集", "打开相机页"]),
]


class VoiceCommander:
    """语音命令解析与执行器"""

    def __init__(self):
        self._handlers: dict[str, Callable] = {}
        self._last_command: Optional[VoiceCommand] = None
        # 构建正则匹配表（预编译加速）
        self._route_map: list[tuple[re.Pattern, str]] = []
        for action, phrases, *_ in COMMAND_ROUTES:
            for phrase in phrases:
                # 模糊匹配：允许中间插入"一下""帮我""请"等语气词
                pattern = re.compile(
                    re.escape(phrase).replace(r"\ ", r"\s*"),
                    re.IGNORECASE,
                )
                self._route_map.append((pattern, action))

    def register_handler(self, action: str, handler: Callable) -> None:
        """注册操作处理器"""
        self._handlers[action] = handler

    def parse(self, text: str) -> Optional[VoiceCommand]:
        """解析语音文本为命令"""
        text = text.strip()
        if not text:
            return None

        # 数字提取（用于审核/标记操作）
        numbers = re.findall(r"\d+", text)
        params = {}
        if numbers:
            params["number"] = int(numbers[0])

        # 日期提取
        date_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", text)
        if date_match:
            params["date"] = date_match.group(1)
        else:
            # "今天"/"昨天"
            if "今天" in text:
                from datetime import date
                params["date"] = date.today().isoformat()
            elif "昨天" in text:
                from datetime import date, timedelta
                params["date"] = (date.today() - timedelta(days=1)).isoformat()

        # 匹配路由
        best_match = None
        best_len = 0
        for pattern, action in self._route_map:
            match = pattern.search(text)
            if match and len(match.group()) > best_len:
                best_match = action
                best_len = len(match.group())

        if best_match:
            cmd = VoiceCommand(action=best_match, params=params, raw_text=text)
            self._last_command = cmd
            return cmd

        return None

    def execute(self, text: str) -> str:
        """解析并执行语音命令，返回反馈消息"""
        cmd = self.parse(text)
        if cmd is None:
            return f'<span style="color:#999">未识别命令: "{text}"</span>'

        handler = self._handlers.get(cmd.action)
        if handler is None:
            return (f'<span style="color:#DD6B20">命令已识别 [{cmd.action}]'
                    f'，但未绑定处理器</span>')

        try:
            result = handler(cmd)
            return f'<span style="color:#38A169">✅ {result}</span>'
        except Exception as e:
            return f'<span style="color:#E53E3E">❌ 执行失败: {str(e)[:80]}</span>'

    @property
    def last_command(self) -> Optional[VoiceCommand]:
        return self._last_command

    @staticmethod
    def describe_commands() -> str:
        """生成命令帮助文本"""
        lines = ["| 功能 | 可以这样说 |", "|------|-----------|"]
        seen = set()
        for action, phrases, *rest in COMMAND_ROUTES:
            desc = rest[0] if rest else action
            if action not in seen:
                seen.add(action)
                examples = " / ".join(f'"{p}"' for p in phrases[:3])
                lines.append(f"| {desc} | {examples} |")
        return "\n".join(lines)


# ==================== 语音 UI 组件 (HTML + JS) ====================

VOICE_INPUT_HTML = """
<div id="voice-control" style="position:fixed;bottom:24px;right:24px;z-index:9999;
    display:flex;flex-direction:column;align-items:flex-end;gap:8px">

  <!-- 语音反馈气泡 -->
  <div id="voice-bubble" style="display:none;
    background:var(--bg-card,#1e1e1e);color:var(--text-primary,#e8e8e8);
    padding:10px 18px;border-radius:16px;font-size:14px;max-width:320px;
    box-shadow:0 4px 20px rgba(0,0,0,0.3);border:1px solid var(--border-color,#333);
    transition:all 0.2s">
    <span id="voice-text">🎤 正在听...</span>
  </div>

  <!-- 麦克风按钮 -->
  <button id="mic-btn" title="语音命令 (点击开始说话)"
    style="width:56px;height:56px;border-radius:50%;border:2px solid #ff6b35;
    background:var(--bg-card,#1e1e1e);cursor:pointer;font-size:24px;
    box-shadow:0 4px 16px rgba(255,107,53,0.25);
    transition:all 0.2s;display:flex;align-items:center;justify-content:center"
    onmousedown="startVoice()" onmouseup="stopVoice()"
    onmouseleave="stopVoice()"
    ontouchstart="startVoice()" ontouchend="stopVoice()">
    🎤
  </button>
</div>

<script>
let recognition = null;
let voiceInput = null;

function initVoice() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        document.getElementById('mic-btn').style.opacity = '0.3';
        document.getElementById('mic-btn').title = '浏览器不支持语音识别';
        return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = 'zh-CN';
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.maxAlternatives = 1;

    // 找到 Gradio 的隐藏 Textbox 用于传递文本
    // 通过 data-testid 或遍历找到 voice_command_input
    setTimeout(() => {
        const inputs = document.querySelectorAll('textarea');
        for (const inp of inputs) {
            const label = inp.closest('.block')?.querySelector('label');
            if (label && label.textContent.includes('语音命令')) {
                voiceInput = inp;
                break;
            }
        }
        // fallback: 查找所有隐藏的 textarea
        if (!voiceInput) {
            const hidden = document.querySelectorAll('textarea[data-testid]');
            for (const h of hidden) {
                if (h.closest('[style*="display:none"]') || h.offsetParent === null) {
                    voiceInput = h;
                }
            }
        }
    }, 2000);

    recognition.onresult = (event) => {
        let text = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            text += event.results[i][0].transcript;
        }
        document.getElementById('voice-text').textContent = '🎤 ' + text;

        if (event.results[event.results.length-1].isFinal) {
            document.getElementById('voice-text').textContent = '⏳ 执行中: ' + text;
            // 填入隐藏的 Gradio Textbox 并触发 change
            if (voiceInput) {
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeInputValueSetter.call(voiceInput, text);
                voiceInput.dispatchEvent(new Event('input', { bubbles: true }));
                voiceInput.dispatchEvent(new Event('change', { bubbles: true }));
                // 尝试触发提交 (回车)
                voiceInput.dispatchEvent(new KeyboardEvent('keydown', {
                    key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true
                }));
            }
            setTimeout(() => {
                document.getElementById('voice-bubble').style.display = 'none';
            }, 2000);
        }
    };

    recognition.onerror = (event) => {
        document.getElementById('voice-text').textContent = '❌ ' + event.error;
        setTimeout(() => {
            document.getElementById('voice-bubble').style.display = 'none';
        }, 2000);
    };

    recognition.onend = () => {
        document.getElementById('mic-btn').style.background = 'var(--bg-card,#1e1e1e)';
    };
}

function startVoice() {
    if (!recognition) return;
    document.getElementById('voice-bubble').style.display = 'block';
    document.getElementById('voice-text').textContent = '🎤 正在听...';
    document.getElementById('mic-btn').style.background = '#ff6b35';
    document.getElementById('mic-btn').style.color = '#fff';
    try { recognition.start(); } catch(e) {}
}

function stopVoice() {
    if (!recognition) return;
    try { recognition.stop(); } catch(e) {}
}

// 初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initVoice);
} else {
    initVoice();
}
</script>
"""
