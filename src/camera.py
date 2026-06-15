"""
图像采集模块 - RTSP 流读取与帧缓存。

支持:
- RTSP 网络摄像头流
- USB/本地摄像头
- 硬触发模式 (PLC 信号)
- 多线程缓冲队列避免帧丢失
"""

import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


class CameraCapture:
    """工业相机采集器"""

    def __init__(
        self,
        source: str = "0",
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        buffer_size: int = 10,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size

        self._cap: Optional[cv2.VideoCapture] = None
        self._buffer: deque = deque(maxlen=buffer_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame_count = 0
        self._start_time = 0.0

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
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止采集"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None

    def read(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        """从缓冲队列读取一帧"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._buffer:
                    return self._buffer.popleft()
            time.sleep(0.001)
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

    def _capture_loop(self) -> None:
        """后台采集循环"""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
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
