"""
语音命令模块单元测试。
"""

import pytest
from src.voice_commander import VoiceCommander, VoiceCommand, COMMAND_ROUTES


class TestVoiceCommanderParse:
    def setup_method(self):
        self.vc = VoiceCommander()

    def test_parse_yolo(self):
        cmd = self.vc.parse("YOLO 快速筛查")
        assert cmd is not None
        assert cmd.action == "yolo_detect"

    def test_parse_vlm(self):
        cmd = self.vc.parse("VLM 精细分析")
        assert cmd is not None
        assert cmd.action == "vlm_analyze"

    def test_parse_full_pipeline(self):
        cmd = self.vc.parse("一键全流程")
        assert cmd is not None
        assert cmd.action == "full_pipeline"

    def test_parse_rag(self):
        cmd = self.vc.parse("帮我做一下根因分析")
        assert cmd is not None
        assert cmd.action == "rag_analysis"

    def test_parse_fuzzy(self):
        """模糊匹配：语气词不影响识别"""
        cmd = self.vc.parse("请帮我做YOLO检测")
        assert cmd is not None
        assert cmd.action == "yolo_detect"

    def test_parse_theme_dark(self):
        cmd = self.vc.parse("切换到深色模式")
        assert cmd is not None
        assert cmd.action == "theme_dark"

    def test_parse_theme_light(self):
        cmd = self.vc.parse("我要浅色主题")
        assert cmd is not None
        assert cmd.action == "theme_light"

    def test_parse_export_csv(self):
        cmd = self.vc.parse("导出CSV")
        assert cmd is not None
        assert cmd.action == "export_csv"

    def test_parse_export_badcase(self):
        cmd = self.vc.parse("导出Bad Case数据集")
        assert cmd is not None
        assert cmd.action == "export_badcase"

    def test_parse_camera_connect(self):
        cmd = self.vc.parse("连接摄像头")
        assert cmd is not None
        assert cmd.action == "camera_connect"

    def test_parse_camera_disconnect(self):
        cmd = self.vc.parse("关闭相机")
        assert cmd is not None
        assert cmd.action == "camera_disconnect"

    def test_parse_review_pass(self):
        cmd = self.vc.parse("审核通过")
        assert cmd is not None
        assert cmd.action == "review_pass"

    def test_parse_review_reject(self):
        cmd = self.vc.parse("标记误检")
        assert cmd is not None
        assert cmd.action == "review_reject"

    def test_parse_navigation(self):
        for text, action in [
            ("打开检测页", "tab_detect"),
            ("去审核页", "tab_review"),
            ("打开报表", "tab_report"),
        ]:
            cmd = self.vc.parse(text)
            assert cmd is not None, f"Failed: {text}"
            assert cmd.action == action, f"{text} -> {cmd.action} != {action}"

    def test_parse_number_extraction(self):
        cmd = self.vc.parse("通过第3条记录")
        assert cmd is not None
        assert cmd.params.get("number") == 3

    def test_parse_empty(self):
        assert self.vc.parse("") is None
        assert self.vc.parse("   ") is None

    def test_parse_unknown(self):
        assert self.vc.parse("今天天气怎么样") is None


class TestVoiceCommanderExecute:
    def setup_method(self):
        self.vc = VoiceCommander()

    def test_execute_registered_handler(self):
        self.vc.register_handler("yolo_detect", lambda cmd: "YOLO started")
        result = self.vc.execute("YOLO检测")
        assert "YOLO started" in result

    def test_execute_unregistered_handler(self):
        result = self.vc.execute("YOLO检测")
        assert "未绑定" in result

    def test_execute_unknown_command(self):
        result = self.vc.execute("放音乐")
        assert "未识别" in result

    def test_describe_commands(self):
        desc = VoiceCommander.describe_commands()
        assert "YOLO" in desc
        assert "VLM" in desc
        assert "导出" in desc


class TestVoiceCommandDataclass:
    def test_create(self):
        cmd = VoiceCommand(action="test", params={"n": 1}, raw_text="hello")
        assert cmd.action == "test"
        assert cmd.params["n"] == 1
