"""
系统监控与告警模块

功能：
1. 系统健康检查（GPU显存/CPU/内存/磁盘/推理延迟/相机连接状态）
2. 告警规则引擎（阈值配置、多级告警：WARNING/CRITICAL）
3. 告警通知（日志告警 + 可扩展的通知接口，预留钉钉/微信/邮件）
4. 健康状态RESTful接口（GET /health 返回JSON状态）

设计原则：
- 所有监控指标通过统一接口采集
- 告警规则可配置，支持多级阈值
- 通知渠道可扩展，默认日志告警
- 健康检查结果以JSON格式输出，便于集成
"""

import json
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)


# ==================== 告警级别 ====================

class AlertLevel(Enum):
    """告警级别"""
    INFO = "INFO"           # 信息
    WARNING = "WARNING"     # 警告
    CRITICAL = "CRITICAL"   # 严重


# ==================== 数据模型 ====================

@dataclass
class AlertRule:
    """告警规则"""
    name: str                          # 规则名称
    metric_key: str                    # 监控指标键名
    warning_threshold: float           # 警告阈值
    critical_threshold: float          # 严重阈值
    comparison: str = "gt"             # 比较方式: gt(大于) | lt(小于)
    description: str = ""              # 规则描述
    enabled: bool = True               # 是否启用


@dataclass
class Alert:
    """告警记录"""
    timestamp: str                     # 告警时间
    level: AlertLevel                  # 告警级别
    rule_name: str                     # 规则名称
    metric_key: str                    # 指标键名
    current_value: float               # 当前值
    threshold: float                   # 阈值
    message: str                       # 告警消息
    acknowledged: bool = False         # 是否已确认


@dataclass
class HealthStatus:
    """系统健康状态"""
    status: str = "healthy"            # healthy | warning | critical
    timestamp: str = ""                # 检查时间
    components: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)


# ==================== 通知接口 ====================

class NotificationChannel:
    """通知渠道基类"""
    
    def send(self, alert: Alert) -> bool:
        """发送告警通知"""
        raise NotImplementedError


class LogNotification(NotificationChannel):
    """日志通知渠道（默认）"""
    
    def send(self, alert: Alert) -> bool:
        level_map = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.ERROR,
        }
        log_level = level_map.get(alert.level, logging.WARNING)
        logger.log(log_level, f"[{alert.level.value}] {alert.message}")
        return True


class EmailNotification(NotificationChannel):
    """邮件通知渠道 (SMTP)"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.smtp_host = config.get("smtp_host") or config.get("smtp_server", "")
        self.smtp_port = config.get("smtp_port", 465)
        self.sender = config.get("sender") or config.get("username", "")
        self.password = config.get("password", "")
        self.receivers = config.get("receivers", [])
        self.use_ssl = config.get("use_ssl", True)

    def send(self, alert: Alert) -> bool:
        if not self.enabled or not self.smtp_host or not self.receivers:
            return False
        try:
            import smtplib
            from email.mime.text import MIMEText

            level_emoji = {"INFO": "[INFO]", "WARNING": "[WARN]", "CRITICAL": "[CRIT]"}
            prefix = level_emoji.get(alert.level.value, "[ALERT]")

            msg = MIMEText(
                f"告警级别: {alert.level.value}\n"
                f"告警时间: {alert.timestamp}\n"
                f"告警消息: {alert.message}\n"
                f"指标详情: {json.dumps(alert.metrics, ensure_ascii=False, default=str)}",
                "plain", "utf-8",
            )
            msg["Subject"] = f"{prefix} 钢铁缺陷检测系统告警"
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.receivers)

            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=10)
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10)
                server.starttls()
            server.login(self.sender, self.password)
            server.sendmail(self.sender, self.receivers, msg.as_string())
            server.quit()
            return True
        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
            return False


class DingTalkNotification(NotificationChannel):
    """钉钉机器人通知渠道"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.webhook = config.get("webhook", "")
        self.secret = config.get("secret", "")

    def send(self, alert: Alert) -> bool:
        if not self.enabled or not self.webhook:
            return False
        try:
            import urllib.request
            import hmac
            import hashlib
            import base64
            import urllib.parse

            url = self.webhook
            if self.secret:
                timestamp = str(round(time.time() * 1000))
                string_to_sign = f"{timestamp}\n{self.secret}"
                hmac_code = hmac.new(
                    self.secret.encode(), string_to_sign.encode(), hashlib.sha256
                ).digest()
                sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
                url = f"{url}&timestamp={timestamp}&sign={sign}"

            level_tags = {"INFO": "信息", "WARNING": "警告", "CRITICAL": "严重"}
            level_text = level_tags.get(alert.level.value, "告警")

            payload = json.dumps({
                "msgtype": "markdown",
                "markdown": {
                    "title": f"缺陷检测系统-{level_text}",
                    "text": (
                        f"### {level_text}告警\n"
                        f"- **时间**: {alert.timestamp}\n"
                        f"- **级别**: {alert.level.value}\n"
                        f"- **消息**: {alert.message}\n"
                    ),
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                return result.get("errcode") == 0
        except Exception as e:
            logger.error(f"钉钉通知发送失败: {e}")
            return False


class WeChatNotification(NotificationChannel):
    """企业微信机器人通知渠道"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.webhook = config.get("webhook", "")

    def send(self, alert: Alert) -> bool:
        if not self.enabled or not self.webhook:
            return False
        try:
            import urllib.request

            level_tags = {"INFO": "信息", "WARNING": "警告", "CRITICAL": "严重"}
            level_text = level_tags.get(alert.level.value, "告警")

            payload = json.dumps({
                "msgtype": "markdown",
                "markdown": {
                    "content": (
                        f"## 缺陷检测系统 - {level_text}告警\n"
                        f"> **时间**: {alert.timestamp}\n"
                        f"> **级别**: <font color=\"warning\">{alert.level.value}</font>\n"
                        f"> **消息**: {alert.message}\n"
                    ),
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                self.webhook, data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                return result.get("errcode") == 0
        except Exception as e:
            logger.error(f"企业微信通知发送失败: {e}")
            return False


# ==================== 告警引擎 ====================

class AlertEngine:
    """告警规则引擎
    
    负责：
    1. 管理告警规则
    2. 评估监控指标
    3. 触发告警
    4. 分发通知
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化告警引擎
        
        Args:
            config: 告警配置字典，包含 alert_rules 和 notifications
        """
        self.rules: List[AlertRule] = []
        self.alerts: List[Alert] = []
        self.channels: List[NotificationChannel] = []
        self._lock = threading.Lock()
        
        # 默认添加日志通知渠道
        self.add_channel(LogNotification())
        
        # 从配置加载
        if config:
            self.load_config(config)
    
    def load_config(self, config: Dict[str, Any]) -> None:
        """从配置字典加载告警规则和通知渠道"""
        # 加载告警规则
        alert_rules = config.get("alert_rules", {})
        self._load_rules(alert_rules)
        
        # 加载通知渠道
        notifications = config.get("notifications", {})
        self._load_notifications(notifications)
    
    def _load_rules(self, rules_config: Dict[str, Any]) -> None:
        """加载告警规则"""
        rule_definitions = [
            ("gpu_memory", "gpu.memory_usage", 90, 95, "gt", "GPU内存使用率过高"),
            ("gpu_temperature", "gpu.temperature", 85, 90, "gt", "GPU温度过高"),
            ("cpu_usage", "cpu.usage", 90, 95, "gt", "CPU使用率过高"),
            ("cpu_temperature", "cpu.temperature", 80, 85, "gt", "CPU温度过高"),
            ("memory_usage", "memory.usage", 90, 95, "gt", "内存使用率过高"),
            ("disk_usage", "disk.usage", 90, 95, "gt", "磁盘使用率过高"),
            ("inference_delay", "performance.inference_delay", 1000, 2000, "gt", "推理延迟过高"),
            ("camera_fps", "performance.camera_fps", 10, 5, "lt", "相机帧率过低"),
            ("defect_rate", "business.defect_rate", 10, 20, "gt", "缺陷率异常偏高"),
            ("false_positive_rate", "business.false_positive_rate", 5, 10, "gt", "误检率过高"),
        ]
        
        for name, metric_key, warn_thresh, crit_thresh, comp, desc in rule_definitions:
            # 从配置中读取自定义阈值（如果存在）
            config_value = rules_config.get(name)
            if isinstance(config_value, (int, float)):
                warn_thresh = config_value
                crit_thresh = config_value * 1.1  # 严重阈值比警告高10%
            
            self.add_rule(AlertRule(
                name=name,
                metric_key=metric_key,
                warning_threshold=warn_thresh,
                critical_threshold=crit_thresh,
                comparison=comp,
                description=desc,
            ))
    
    def _load_notifications(self, notifications_config: Dict[str, Any]) -> None:
        """加载通知渠道"""
        # 邮件通知
        email_config = notifications_config.get("email", {})
        if email_config.get("enabled", False):
            self.add_channel(EmailNotification(email_config))
        
        # 钉钉通知
        dingtalk_config = notifications_config.get("dingtalk", {})
        if dingtalk_config.get("enabled", False):
            self.add_channel(DingTalkNotification(dingtalk_config))
        
        # 微信通知
        wechat_config = notifications_config.get("wechat", {})
        if wechat_config.get("enabled", False):
            self.add_channel(WeChatNotification(wechat_config))
    
    def add_rule(self, rule: AlertRule) -> None:
        """添加告警规则"""
        self.rules.append(rule)
        logger.debug(f"添加告警规则: {rule.name}")
    
    def add_channel(self, channel: NotificationChannel) -> None:
        """添加通知渠道"""
        self.channels.append(channel)
    
    def evaluate(self, metrics: Dict[str, Any]) -> List[Alert]:
        """
        评估所有告警规则
        
        Args:
            metrics: 监控指标字典，支持嵌套键（如 "gpu.memory_usage"）
            
        Returns:
            List[Alert]: 触发的告警列表
        """
        triggered = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # 获取指标值（支持嵌套键）
            value = self._get_nested_value(metrics, rule.metric_key)
            if value is None:
                continue
            
            # 判断告警级别
            level = self._evaluate_rule(rule, value)
            if level is not None:
                alert = Alert(
                    timestamp=datetime.now().isoformat(),
                    level=level,
                    rule_name=rule.name,
                    metric_key=rule.metric_key,
                    current_value=value,
                    threshold=(
                        rule.critical_threshold if level == AlertLevel.CRITICAL
                        else rule.warning_threshold
                    ),
                    message=f"{rule.description}: 当前值={value}, 阈值={rule.warning_threshold}",
                )
                triggered.append(alert)
                
                with self._lock:
                    self.alerts.append(alert)
                
                # 分发通知
                self._dispatch(alert)
        
        return triggered
    
    def _evaluate_rule(self, rule: AlertRule, value: float) -> Optional[AlertLevel]:
        """评估单条规则"""
        if rule.comparison == "gt":
            if value >= rule.critical_threshold:
                return AlertLevel.CRITICAL
            elif value >= rule.warning_threshold:
                return AlertLevel.WARNING
        elif rule.comparison == "lt":
            if value <= rule.critical_threshold:
                return AlertLevel.CRITICAL
            elif value <= rule.warning_threshold:
                return AlertLevel.WARNING
        
        return None
    
    def _dispatch(self, alert: Alert) -> None:
        """分发告警到所有通知渠道"""
        for channel in self.channels:
            try:
                channel.send(alert)
            except Exception as e:
                logger.error(f"通知渠道发送失败: {e}")
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Optional[float]:
        """从嵌套字典中获取值，如 "gpu.memory_usage" -> data["gpu"]["memory_usage"]"""
        keys = key.split(".")
        current = data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        try:
            return float(current)
        except (TypeError, ValueError):
            return None
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近的告警记录"""
        with self._lock:
            return [asdict(a) for a in self.alerts[-limit:]]
    
    def acknowledge_alert(self, index: int) -> bool:
        """确认告警"""
        with self._lock:
            if 0 <= index < len(self.alerts):
                self.alerts[index].acknowledged = True
                return True
        return False


# ==================== 系统健康检查 ====================

class SystemMonitor:
    """系统监控器
    
    负责：
    1. 定期采集系统健康指标
    2. 评估告警规则
    3. 提供健康状态查询接口
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化系统监控器
        
        Args:
            config: 监控配置字典
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.check_interval = self.config.get("check_interval", 60)
        
        # 告警引擎
        self.alert_engine = AlertEngine(config)
        
        # 状态
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_metrics: Dict[str, Any] = {}
        self._latest_health: HealthStatus = HealthStatus()
        
        # 外部状态引用（由调用方注入）
        self._camera_status_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self._inference_stats_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self._plc_status_provider: Optional[Callable[[], Dict[str, Any]]] = None
    
    def set_camera_status_provider(self, provider: Callable[[], Dict[str, Any]]) -> None:
        """设置相机状态提供者"""
        self._camera_status_provider = provider
    
    def set_inference_stats_provider(self, provider: Callable[[], Dict[str, Any]]) -> None:
        """设置推理统计提供者"""
        self._inference_stats_provider = provider
    
    def set_plc_status_provider(self, provider: Callable[[], Dict[str, Any]]) -> None:
        """设置PLC状态提供者"""
        self._plc_status_provider = provider
    
    def collect_metrics(self) -> Dict[str, Any]:
        """采集所有系统指标"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "gpu": self._collect_gpu_metrics(),
            "cpu": self._collect_cpu_metrics(),
            "memory": self._collect_memory_metrics(),
            "disk": self._collect_disk_metrics(),
            "network": self._collect_network_metrics(),
            "performance": self._collect_performance_metrics(),
            "camera": self._collect_camera_metrics(),
            "plc": self._collect_plc_metrics(),
        }
        
        with self._lock:
            self._latest_metrics = metrics
        
        return metrics
    
    def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """采集GPU指标"""
        gpu_info = {"available": False, "count": 0}
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["count"] = torch.cuda.device_count()
                
                # 获取第一个GPU的详细信息
                if gpu_info["count"] > 0:
                    # GPU内存
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    reserved = torch.cuda.memory_reserved(0)
                    allocated = torch.cuda.memory_allocated(0)
                    
                    gpu_info["memory_total_mb"] = round(total_memory / (1024 ** 2), 1)
                    gpu_info["memory_allocated_mb"] = round(allocated / (1024 ** 2), 1)
                    gpu_info["memory_reserved_mb"] = round(reserved / (1024 ** 2), 1)
                    gpu_info["memory_usage"] = round(allocated / total_memory * 100, 1)
                    
                    # GPU名称
                    gpu_info["name"] = torch.cuda.get_device_name(0)
                    
                    # GPU温度（如果可用）
                    try:
                        import subprocess
                        result = subprocess.run(
                            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5
                        )
                        if result.returncode == 0:
                            gpu_info["temperature"] = float(result.stdout.strip().split("\n")[0])
                    except Exception:
                        gpu_info["temperature"] = -1
        except ImportError:
            pass
        
        return gpu_info
    
    def _collect_cpu_metrics(self) -> Dict[str, Any]:
        """采集CPU指标"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        metrics = {
            "usage": cpu_percent,
            "count": cpu_count,
            "frequency_mhz": round(cpu_freq.current, 1) if cpu_freq else 0,
        }
        
        # CPU温度（Windows）
        try:
            if platform.system() == "Windows":
                import subprocess
                result = subprocess.run(
                    ["wmic", "/namespace:\\\\root\\wmi", "PATH", "MSAcpi_ThermalZoneTemperature", "get", "CurrentTemperature"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) >= 2:
                        temp_k = float(lines[1].strip())
                        metrics["temperature"] = round(temp_k / 10 - 273.15, 1)
                    else:
                        metrics["temperature"] = -1
                else:
                    metrics["temperature"] = -1
            else:
                # Linux: 尝试读取 /sys/class/thermal/
                try:
                    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                        metrics["temperature"] = round(int(f.read().strip()) / 1000, 1)
                except Exception:
                    metrics["temperature"] = -1
        except Exception:
            metrics["temperature"] = -1
        
        return metrics
    
    def _collect_memory_metrics(self) -> Dict[str, Any]:
        """采集内存指标"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "total_gb": round(mem.total / (1024 ** 3), 1),
            "available_gb": round(mem.available / (1024 ** 3), 1),
            "used_gb": round(mem.used / (1024 ** 3), 1),
            "usage": round(mem.percent, 1),
            "swap_total_gb": round(swap.total / (1024 ** 3), 1),
            "swap_usage": round(swap.percent, 1),
        }
    
    def _collect_disk_metrics(self) -> Dict[str, Any]:
        """采集磁盘指标"""
        disks = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks.append({
                    "mountpoint": partition.mountpoint,
                    "total_gb": round(usage.total / (1024 ** 3), 1),
                    "used_gb": round(usage.used / (1024 ** 3), 1),
                    "free_gb": round(usage.free / (1024 ** 3), 1),
                    "usage": round(usage.percent, 1),
                })
            except PermissionError:
                continue
        
        # 汇总：取使用率最高的磁盘
        max_usage = max((d["usage"] for d in disks), default=0)
        
        return {
            "usage": max_usage,
            "partitions": disks,
        }
    
    def _collect_network_metrics(self) -> Dict[str, Any]:
        """采集网络指标"""
        net_io = psutil.net_io_counters()
        return {
            "bytes_sent_mb": round(net_io.bytes_sent / (1024 ** 2), 1),
            "bytes_recv_mb": round(net_io.bytes_recv / (1024 ** 2), 1),
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
        }
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """采集性能指标"""
        metrics = {
            "inference_delay": -1,   # ms，-1表示未采集
            "camera_fps": -1,
            "queue_length": -1,
        }
        
        # 从外部提供者获取推理统计
        if self._inference_stats_provider:
            try:
                stats = self._inference_stats_provider()
                metrics.update(stats)
            except Exception as e:
                logger.debug(f"获取推理统计失败: {e}")
        
        return metrics
    
    def _collect_camera_metrics(self) -> Dict[str, Any]:
        """采集相机指标"""
        metrics = {
            "connected": False,
            "fps": -1,
            "status": "unknown",
        }
        
        if self._camera_status_provider:
            try:
                status = self._camera_status_provider()
                metrics.update(status)
            except Exception as e:
                logger.debug(f"获取相机状态失败: {e}")
        
        return metrics
    
    def _collect_plc_metrics(self) -> Dict[str, Any]:
        """采集PLC指标"""
        metrics = {
            "connected": False,
            "status": "disabled",
        }
        
        if self._plc_status_provider:
            try:
                status = self._plc_status_provider()
                metrics.update(status)
            except Exception as e:
                logger.debug(f"获取PLC状态失败: {e}")
        
        return metrics
    
    def check_health(self) -> HealthStatus:
        """执行完整健康检查"""
        # 采集指标
        metrics = self.collect_metrics()
        
        # 评估告警
        alerts = self.alert_engine.evaluate(metrics)
        
        # 构建健康状态
        health = HealthStatus(
            timestamp=datetime.now().isoformat(),
            components={
                "gpu": metrics.get("gpu", {}),
                "cpu": metrics.get("cpu", {}),
                "memory": metrics.get("memory", {}),
                "disk": metrics.get("disk", {}),
                "network": metrics.get("network", {}),
                "camera": metrics.get("camera", {}),
                "plc": metrics.get("plc", {}),
            },
            performance=metrics.get("performance", {}),
            alerts=[asdict(a) for a in alerts],
        )
        
        # 判断整体状态
        critical_count = sum(1 for a in alerts if a.level == AlertLevel.CRITICAL)
        warning_count = sum(1 for a in alerts if a.level == AlertLevel.WARNING)
        
        if critical_count > 0:
            health.status = "critical"
        elif warning_count > 0:
            health.status = "warning"
        else:
            health.status = "healthy"
        
        with self._lock:
            self._latest_health = health
        
        return health
    
    def get_health_json(self) -> str:
        """获取健康状态JSON字符串"""
        health = self._latest_health
        return json.dumps(asdict(health), ensure_ascii=False, indent=2)
    
    def get_health_dict(self) -> Dict[str, Any]:
        """获取健康状态字典"""
        return asdict(self._latest_health)
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """获取最新指标"""
        with self._lock:
            return self._latest_metrics.copy()
    
    # ---- 后台监控循环 ----
    
    def _monitor_loop(self) -> None:
        """后台监控循环"""
        logger.info(f"系统监控已启动，检查间隔: {self.check_interval}秒")
        
        while self._running:
            try:
                self.check_health()
            except Exception as e:
                logger.error(f"健康检查异常: {e}")
            
            # 等待下一次检查
            time.sleep(self.check_interval)
    
    def start(self) -> bool:
        """启动监控"""
        if not self.enabled:
            logger.info("系统监控已禁用")
            return False
        
        if self._running:
            logger.warning("系统监控已在运行")
            return True
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
        logger.info("系统监控已启动")
        return True
    
    def stop(self) -> None:
        """停止监控"""
        self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
        
        logger.info("系统监控已停止")


# ==================== 工厂函数 ====================

def create_monitor_from_config(config_dict: Dict[str, Any]) -> SystemMonitor:
    """
    从配置字典创建系统监控器
    
    Args:
        config_dict: 配置字典，通常来自config.yaml的monitor段
        
    Returns:
        SystemMonitor: 系统监控器实例
    """
    return SystemMonitor(config_dict)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== 系统监控模块测试 ===\n")
    
    # 测试配置
    test_config = {
        "enabled": True,
        "check_interval": 5,
        "alert_rules": {
            "cpu_usage": 80,
            "memory_usage": 85,
            "disk_usage": 90,
        },
        "notifications": {
            "email": {"enabled": False},
            "dingtalk": {"enabled": False},
            "wechat": {"enabled": False},
        }
    }
    
    # 创建监控器
    monitor = SystemMonitor(test_config)
    
    # 执行一次健康检查
    print("执行健康检查...")
    health = monitor.check_health()
    
    print(f"\n系统状态: {health.status}")
    print(f"检查时间: {health.timestamp}")
    
    print("\n--- 组件状态 ---")
    for name, comp in health.components.items():
        if isinstance(comp, dict):
            status_str = json.dumps(comp, ensure_ascii=False, indent=2)
            print(f"\n[{name}]")
            print(status_str)
    
    print("\n--- 性能指标 ---")
    print(json.dumps(health.performance, ensure_ascii=False, indent=2))
    
    if health.alerts:
        print(f"\n--- 告警 ({len(health.alerts)}条) ---")
        for alert in health.alerts:
            print(f"  [{alert['level']}] {alert['message']}")
    else:
        print("\n--- 无告警 ---")
    
    # 输出完整JSON
    print("\n--- 完整健康状态JSON ---")
    print(monitor.get_health_json())