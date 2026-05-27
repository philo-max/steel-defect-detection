"""
系统监控模块单元测试。

覆盖:
- SystemMonitor 初始化与配置
- 指标采集（GPU/CPU/内存/磁盘）
- 健康检查 JSON 输出
- 相机状态集成
- AlertEngine 告警规则评估
"""

import json

import pytest

from src.monitor import (
    Alert, AlertEngine, AlertLevel, AlertRule, LogNotification, SystemMonitor,
)


class TestAlertRule:
    def test_create_gt_rule(self):
        rule = AlertRule("mem", "memory.usage", 80.0, 95.0)
        assert rule.comparison == "gt"

    def test_create_lt_rule(self):
        rule = AlertRule("fps", "perf.fps", 10.0, 5.0, comparison="lt")
        assert rule.comparison == "lt"


class TestAlertLevel:
    def test_values(self):
        assert AlertLevel.WARNING.value == "WARNING"
        assert AlertLevel.CRITICAL.value == "CRITICAL"


class TestAlert:
    def test_default_not_acknowledged(self):
        a = Alert("t", AlertLevel.WARNING, "r", "k", 1.0, 2.0, "m")
        assert not a.acknowledged


class TestSystemMonitorInit:
    def test_default(self):
        m = SystemMonitor()
        assert m.enabled is True
        assert m.alert_engine is not None

    def test_with_config(self):
        m = SystemMonitor({"check_interval": 30, "enabled": False})
        assert m.check_interval == 30
        assert m.enabled is False

    def test_alert_engine_has_rules_with_config(self):
        m = SystemMonitor({"alert_rules": {}})
        assert len(m.alert_engine.rules) >= 5


class TestSystemMonitorCollect:
    def test_structure(self):
        m = SystemMonitor()
        metrics = m.collect_metrics()
        for key in ("timestamp", "gpu", "cpu", "memory", "disk"):
            assert key in metrics, f"Missing {key}"

    def test_cpu_usage_valid(self):
        m = SystemMonitor()
        metrics = m.collect_metrics()
        assert 0.0 <= metrics["cpu"]["usage"] <= 100.0

    def test_memory_usage_valid(self):
        m = SystemMonitor()
        metrics = m.collect_metrics()
        assert 0.0 <= metrics["memory"]["usage"] <= 100.0


class TestSystemMonitorHealth:
    def test_health_json(self):
        m = SystemMonitor()
        m.collect_metrics()
        data = json.loads(m.get_health_json())
        assert "status" in data

    def test_health_dict(self):
        m = SystemMonitor()
        m.collect_metrics()
        data = m.get_health_dict()
        assert "status" in data

    def test_with_camera_connected(self):
        m = SystemMonitor()
        m.set_camera_status_provider(lambda: {"connected": True, "fps": 30, "status": "running"})
        metrics = m.collect_metrics()
        assert "camera" in metrics
        assert metrics["camera"]["connected"] is True

    def test_with_camera_disconnected(self):
        m = SystemMonitor()
        m.set_camera_status_provider(lambda: {"connected": False, "fps": -1})
        metrics = m.collect_metrics()
        assert "camera" in metrics
        assert metrics["camera"]["connected"] is False


class TestAlertEngineEvaluate:
    def test_warning_triggered(self):
        engine = AlertEngine()
        engine.rules = [AlertRule("t", "cpu.usage", 80, 95)]
        alerts = engine.evaluate({"cpu": {"usage": 85.0}})
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WARNING

    def test_critical_triggered(self):
        engine = AlertEngine()
        engine.rules = [AlertRule("t", "cpu.usage", 80, 95)]
        alerts = engine.evaluate({"cpu": {"usage": 97.0}})
        assert alerts[0].level == AlertLevel.CRITICAL

    def test_normal_no_alert(self):
        engine = AlertEngine()
        engine.rules = [AlertRule("t", "cpu.usage", 80, 95)]
        alerts = engine.evaluate({"cpu": {"usage": 30.0}})
        assert len(alerts) == 0

    def test_lt_comparison(self):
        engine = AlertEngine()
        engine.rules = [AlertRule("f", "p.fps", 10, 5, comparison="lt")]
        alerts = engine.evaluate({"p": {"fps": 3.0}})
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.CRITICAL

    def test_add_rule(self):
        engine = AlertEngine()
        rule = AlertRule("c", "d.u", 90, 98)
        engine.add_rule(rule)
        assert rule in engine.rules


class TestLogNotification:
    def test_send_ok(self):
        alert = Alert("t", AlertLevel.WARNING, "r", "k", 85.0, 80.0, "test")
        assert LogNotification().send(alert) is True
