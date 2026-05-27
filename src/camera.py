"""
图像采集模块 - RTSP 流读取与帧缓存。

支持:
- RTSP 网络摄像头流
- USB/本地摄像头
- 硬触发模式 (PLC 信号)
- 多线程缓冲队列避免帧丢失
- 软件触发 / 连续采集 / PLC硬触发 三种模式
"""

import os
import random
import threading
import time
from collections import deque
from enum import Enum
from typing import Optional, Callable

import cv2
import numpy as np


class TriggerMode(Enum):
    """相机触发模式"""
    SOFTWARE = "software"         # 软件触发（手动调用 snapshot）
    CONTINUOUS = "continuous"     # 连续采集（自由运行）
    HARDWARE = "hardware"         # 硬件触发（PLC 信号）


class CameraCapture:
    """工业相机采集器

    支持三种触发模式：
    - CONTINUOUS: 后台持续采集，适合监控和预览
    - SOFTWARE: 手动触发拍照，适合单次检测
    - HARDWARE: PLC硬触发，适合生产线自动化

    硬触发模式通过 PLCTrigger 模块接收PLC信号，
    每次触发拍照后自动将帧推入触发帧队列，
    并可通过回调通知检测模块处理。
    """

    def __init__(
        self,
        source: str = "0",
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        buffer_size: int = 10,
        trigger_mode: str = "continuous",
        plc_trigger: Optional[object] = None,
    ):
        """
        初始化相机采集器

        Args:
            source: 相机源，数字字符串=USB摄像头编号，RTSP地址=网络流
            width: 图像宽度
            height: 图像高度
            fps: 目标帧率
            buffer_size: 帧缓冲队列大小
            trigger_mode: 触发模式 "continuous" | "software" | "hardware"
            plc_trigger: PLC触发控制器实例（trigger_mode="hardware"时必须提供）
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size

        # 设置触发模式
        self._trigger_mode_map = {
            "continuous": TriggerMode.CONTINUOUS,
            "software": TriggerMode.SOFTWARE,
            "hardware": TriggerMode.HARDWARE,
            "plc_signal": TriggerMode.HARDWARE,
            "plc_hardware": TriggerMode.HARDWARE,
        }
        self.trigger_mode = self._trigger_mode_map.get(
            trigger_mode.lower(), TriggerMode.CONTINUOUS
        )

        self._plc_trigger = plc_trigger
        self._cap: Optional[cv2.VideoCapture] = None
        self._buffer: deque = deque(maxlen=buffer_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame_count = 0
        self._start_time = 0.0

        # 硬触发相关
        self._trigger_frame_buffer: deque = deque(maxlen=10)
        self._trigger_callbacks: list[Callable] = []
        self._plc_monitor_thread: Optional[threading.Thread] = None
        self._plc_monitor_running = False

    def open(self) -> bool:
        """打开摄像头连接"""
        # 判断源类型
        if self.source.isdigit():
            source_id = int(self.source)
        else:
            source_id = self.source

        self._cap = cv2.VideoCapture(source_id)

        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 实际值可能与设置值不同
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return True

    def start(self) -> None:
        """启动后台采集线程"""
        if self._running:
            return
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("摄像头未打开，请先调用 open()")

        self._running = True
        self._start_time = time.perf_counter()

        # 根据触发模式启动不同的采集逻辑
        if self.trigger_mode == TriggerMode.HARDWARE:
            self._start_hardware_trigger()
        else:
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """停止采集"""
        self._running = False
        self._plc_monitor_running = False

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._plc_monitor_thread:
            self._plc_monitor_thread.join(timeout=2.0)
            self._plc_monitor_thread = None

        if self._cap:
            self._cap.release()
            self._cap = None

    def read(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        """从缓冲队列读取一帧

        在硬触发模式下优先读取触发帧队列中的帧；
        在连续/软件触发模式下从采集缓冲中读取。
        """
        # 硬触发模式：优先取触发帧
        if self.trigger_mode == TriggerMode.HARDWARE:
            with self._lock:
                if self._trigger_frame_buffer:
                    return self._trigger_frame_buffer.popleft()
            return None

        # 连续/软件触发模式：从缓冲读
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._buffer:
                    return self._buffer.popleft()
            time.sleep(0.001)
        return None

    def read_triggered_frame(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """阻塞等待PLC触发帧（硬触发模式专用）"""
        if self.trigger_mode != TriggerMode.HARDWARE:
            return self.read(timeout)

        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._trigger_frame_buffer:
                    return self._trigger_frame_buffer.popleft()
            time.sleep(0.005)
        return None

    def register_trigger_callback(self, callback: Callable) -> None:
        """注册硬触发回调

        回调会在每次PLC触发拍照后调用，
        参数为触发时捕获的帧（numpy数组）。
        """
        self._trigger_callbacks.append(callback)

    def snapshot(self) -> Optional[np.ndarray]:
        """软件触发：立即捕获一帧"""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if ret:
            self._frame_count += 1
        return frame if ret else None

    # ---- PLC 硬触发逻辑 ----

    def _start_hardware_trigger(self) -> None:
        """启动PLC硬触发监控"""
        if self._plc_trigger is None:
            raise RuntimeError(
                "trigger_mode='hardware' 但未提供 plc_trigger 实例，"
                "请先初始化 PLCTrigger 并传入 CameraCapture"
            )

        # 将PLC触发信号绑定到拍照动作
        self._plc_trigger.register_callback("trigger", self._on_plc_trigger)

        # 仍然启动采集线程用于保持相机活跃并获取帧
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # 启动PLC监控（如果尚未启动）
        if not self._plc_trigger._running:
            self._plc_trigger.start()

    def _on_plc_trigger(self, trigger_value: int) -> None:
        """PLC触发信号回调：执行拍照 + 注入触发帧"""
        # 从采集缓冲区获取最新帧作为触发帧
        frame = self._grab_latest_frame()
        if frame is not None:
            with self._lock:
                self._trigger_frame_buffer.append(frame)

            # 调用所有注册的触发回调
            for cb in self._trigger_callbacks:
                try:
                    cb(frame)
                except Exception:
                    pass

            # 向PLC发送检测完成信号
            if self._plc_trigger is not None:
                self._plc_trigger.send_detection_complete(success=True)

    def _grab_latest_frame(self) -> Optional[np.ndarray]:
        """从相机直接抓取最新一帧（不经过缓冲队列）"""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if ret:
            self._frame_count += 1
            return frame
        return None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps_actual(self) -> float:
        if self._frame_count > 0:
            elapsed = time.perf_counter() - self._start_time
            return self._frame_count / elapsed if elapsed > 0 else 0
        return 0.0

    @property
    def has_triggered_frame(self) -> bool:
        """是否有待处理的PLC触发帧"""
        with self._lock:
            return len(self._trigger_frame_buffer) > 0

    def _capture_loop(self) -> None:
        """后台采集循环"""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue

            self._frame_count += 1

            with self._lock:
                self._buffer.append(frame)

    def __enter__(self):
        self.open()
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ==================== 线阵相机 ====================


class LineScanCamera:
    """
    线阵相机采集器 — 将逐行扫描数据累积为 2D 帧。

    工作原理：
        线阵相机每次只拍摄一行像素（如 8192×1），随着钢铁带材运动，
        逐行累积形成完整 2D 图像。累积行数由编码器触发或软件设定。

    核心参数:
        line_width: 每行像素宽度（如 8192=8K 线阵）
        frame_height: 拼接成帧的目标高度（累积多少行）
        line_rate_hz: 线扫描频率（受编码器/内部时钟控制）

    使用场景:
        1. 实时模式: 从物理线阵相机 (CameraLink/GigE Vision) 接收行数据
        2. 仿真模式: 从已有 2D 图像按行切分模拟线扫描过程（用于算法验证）

    使用方式:
        # 仿真模式（从 2D 图像模拟）
        cam = LineScanCamera(mode="simulation", source_image="test.jpg")
        cam.open()
        cam.start()
        frame = cam.read()  # 累积完成后返回完整 2D 帧

        # 实时模式（连接物理线阵相机 SDK）
        cam = LineScanCamera(mode="realtime", line_width=8192, frame_height=4096)
        # 需配合厂商 SDK 实现 _grab_line()
    """

    def __init__(
        self,
        line_width: int = 8192,
        frame_height: int = 4096,
        line_rate_hz: float = 20000.0,
        mode: str = "simulation",
        source_image: str = "",
        encoder_resolution_um: float = 50.0,  # 编码器分辨率 (μm/pulse)
        target_resolution_um: float = 50.0,   # 目标分辨率 (μm/pixel)
    ):
        """
        Args:
            line_width: 每行像素宽度
            frame_height: 累积行数（帧高度）
            line_rate_hz: 线扫描频率
            mode: "simulation" | "realtime"
            source_image: 仿真模式下的源图像路径
            encoder_resolution_um: 编码器每脉冲对应的物理距离 (μm)
            target_resolution_um: 目标像素分辨率 (μm/pixel)
        """
        self.line_width = line_width
        self.frame_height = frame_height
        self.line_rate_hz = line_rate_hz
        self.mode = mode
        self.source_image = source_image
        self.encoder_resolution_um = encoder_resolution_um
        self.target_resolution_um = target_resolution_um

        # 内部状态
        self._running = False
        self._frame_buffer: deque = deque(maxlen=5)  # 完整帧缓冲
        self._line_buffer: Optional[np.ndarray] = None  # 正在累积的帧
        self._line_index: int = 0  # 当前累积到第几行
        self._frame_count: int = 0
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._start_time: float = 0.0

        # 仿真模式专用
        self._sim_source_img: Optional[np.ndarray] = None
        self._sim_line_index: int = 0
        self._sim_total_lines: int = 0

    # ==================== 生命周期 ====================

    def open(self) -> bool:
        """初始化相机连接"""
        if self.mode == "simulation":
            return self._open_simulation()
        else:
            return self._open_realtime()

    def _open_simulation(self) -> bool:
        """仿真模式：加载 2D 图像作为线扫描源"""
        if not self.source_image or not os.path.exists(self.source_image):
            print("[LineScan] 仿真模式需要有效的 source_image")
            return False

        self._sim_source_img = cv2.imread(self.source_image)
        if self._sim_source_img is None:
            print(f"[LineScan] 无法加载图像: {self.source_image}")
            return False

        # 将 2D 图像转换为 "线扫描流"
        # 如果图像足够大，按 frame_height 循环扫描
        self._sim_total_lines = self._sim_source_img.shape[0]
        self._sim_line_index = 0

        print(f"[LineScan] 仿真模式就绪: {self._sim_source_img.shape[1]}×{self._sim_total_lines}")
        return True

    def _open_realtime(self) -> bool:
        """实时模式：连接物理线阵相机"""
        # 实际部署时需要对接厂商 SDK（如 Basler, Dalsa, Keyence）
        # 这里提供接口框架，具体 SDK 调用需按实际相机型号实现
        print(f"[LineScan] 实时模式: {self.line_width}×{self.frame_height} @ {self.line_rate_hz}Hz")
        print("[LineScan] 提示: 需对接具体相机厂商 SDK (Basler pylon / Dalsa Sapera / etc.)")
        # 初始化行缓冲
        self._line_buffer = np.zeros(
            (self.frame_height, self.line_width, 3), dtype=np.uint8
        )
        return True

    def start(self) -> None:
        """启动行采集线程"""
        if self._running:
            return

        self._running = True
        self._start_time = time.perf_counter()
        self._line_index = 0
        self._line_buffer = np.zeros(
            (self.frame_height, self.line_width, 3), dtype=np.uint8
        )

        self._thread = threading.Thread(target=self._line_scan_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止采集"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ==================== 行扫描主循环 ====================

    def _line_scan_loop(self) -> None:
        """
        行扫描主循环。

        仿真模式: 从 2D 图像逐行读取 → 累积
        实时模式: 等待相机 SDK 回调 → 接收行数据 → 累积
        """
        while self._running:
            if self.mode == "simulation":
                line_data = self._grab_line_simulation()
            else:
                line_data = self._grab_line_realtime()

            if line_data is None:
                time.sleep(0.0001)  # 100μs 等待
                continue

            # 写入行缓冲
            with self._lock:
                if self._line_buffer is not None and self._line_index < self.frame_height:
                    # 确保行数据宽度匹配
                    actual_width = min(len(line_data), self.line_width)
                    self._line_buffer[self._line_index, :actual_width] = line_data[:actual_width]
                    self._line_index += 1

            # 帧累积完成
            if self._line_index >= self.frame_height:
                self._complete_frame()

            # 控制行速率（仿真模式）
            if self.mode == "simulation":
                line_interval = 1.0 / self.line_rate_hz
                time.sleep(line_interval)

    def _complete_frame(self) -> None:
        """帧累积完成：推入帧缓冲，重置行缓冲"""
        with self._lock:
            if self._line_buffer is not None:
                frame = self._line_buffer.copy()
                self._frame_buffer.append(frame)
                self._frame_count += 1

                # 重置
                self._line_index = 0
                self._line_buffer = np.zeros(
                    (self.frame_height, self.line_width, 3), dtype=np.uint8
                )

    # ==================== 行获取 ====================

    def _grab_line_simulation(self) -> Optional[np.ndarray]:
        """仿真：从 2D 图像读取下一行"""
        if self._sim_source_img is None:
            return None

        line = self._sim_source_img[self._sim_line_index, :].copy()

        # 如果宽度不匹配，缩放
        if line.shape[1] != self.line_width:
            line = cv2.resize(line, (self.line_width, 1))

        # 循环扫描（模拟连续带材）
        self._sim_line_index = (self._sim_line_index + 1) % self._sim_total_lines

        return line

    def _grab_line_realtime(self) -> Optional[np.ndarray]:
        """
        实时：从物理线阵相机获取一行。

        实际部署时需替换为厂商 SDK 调用，例如:
            - Basler: camera.RetrieveResult(...)
            - Dalsa: SapBuffer.GetLine(...)
            - Keyence: 专用 API

        返回: 1D numpy array, shape=(line_width, 3) 或 (line_width,)
        """
        # 占位实现 — 模拟噪声行数据（用于接口测试）
        # 实际部署请替换为真实相机 SDK
        line = np.random.randint(100, 160, (self.line_width, 3), dtype=np.uint8)
        # 模拟偶尔的缺陷行（深色像素）
        if random.random() < 0.02:
            defect_pos = random.randint(0, self.line_width - 50)
            line[defect_pos:defect_pos + 50, :] = 30
        return line

    # ==================== 帧读取 ====================

    def read(self, timeout: float = 10.0) -> Optional[np.ndarray]:
        """
        从帧缓冲读取一个完整 2D 帧。

        会阻塞等待直到有完整帧可用或超时。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame_buffer:
                    return self._frame_buffer.popleft()
            time.sleep(0.01)
        return None

    def read_nonblocking(self) -> Optional[np.ndarray]:
        """非阻塞读取帧"""
        with self._lock:
            return self._frame_buffer.popleft() if self._frame_buffer else None

    # ==================== 属性 ====================

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps_actual(self) -> float:
        """实际帧率 = 行速率 / 帧高度"""
        if self._frame_count > 0:
            elapsed = time.perf_counter() - self._start_time
            return self._frame_count / elapsed if elapsed > 0 else 0
        return 0.0

    @property
    def line_rate_actual(self) -> float:
        """实际行速率"""
        if self._running:
            elapsed = time.perf_counter() - self._start_time
            return self._line_index / elapsed if elapsed > 0 else 0
        return 0.0

    @property
    def progress(self) -> float:
        """当前帧累积进度 (0.0 ~ 1.0)"""
        return self._line_index / self.frame_height if self.frame_height > 0 else 0.0

    # ==================== 编码器参数计算 ====================

    def compute_scan_params(
        self,
        strip_speed_ms: float,
        target_pixel_resolution_um: float = 50.0,
    ) -> dict:
        """
        根据带钢运行速度计算线扫描参数。

        Args:
            strip_speed_ms: 带钢运行速度 (m/s)
            target_pixel_resolution_um: 目标像素分辨率 (μm/pixel)

        Returns:
            dict: 包含 line_rate, frame_height, overlap 等参数
        """
        # 行速率 = 速度 / 分辨率
        # 例: 10 m/s ÷ 50 μm = 10,000,000 μm/s ÷ 50 μm = 200,000 Hz
        speed_ums = strip_speed_ms * 1_000_000  # μm/s
        line_rate = speed_ums / target_pixel_resolution_um

        # 帧高度建议：使帧覆盖约 200mm 的带钢长度
        frame_length_mm = 200
        frame_height = int(frame_length_mm * 1000 / target_pixel_resolution_um)

        # 建议重叠行数（避免帧边界缺陷漏检）
        overlap_lines = int(frame_height * 0.1)

        return {
            "strip_speed_ms": strip_speed_ms,
            "target_resolution_um": target_pixel_resolution_um,
            "required_line_rate_hz": line_rate,
            "suggested_frame_height": frame_height,
            "frame_coverage_mm": frame_length_mm,
            "overlap_lines": overlap_lines,
            "frame_rate_fps": line_rate / (frame_height + overlap_lines),
        }

    def __enter__(self):
        self.open()
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()