"""
相机模块单元测试。
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.camera import CameraCapture, TriggerMode


class TestTriggerMode:
    def test_enum_values(self):
        assert TriggerMode.SOFTWARE.value == "software"
        assert TriggerMode.CONTINUOUS.value == "continuous"
        assert TriggerMode.HARDWARE.value == "hardware"


class TestCameraCapture:
    @patch("src.camera.cv2")
    def test_init_default_params(self, mock_cv2):
        cam = CameraCapture(source="0")
        assert cam.source == "0"
        assert cam.width == 1920
        assert cam.height == 1080
        assert cam.fps == 30
        assert cam.buffer_size == 10

    @patch("src.camera.cv2")
    def test_init_custom_params(self, mock_cv2):
        cam = CameraCapture(
            source="rtsp://192.168.1.100/stream",
            width=640,
            height=480,
            fps=15,
            buffer_size=5,
            trigger_mode="software",
        )
        assert cam.source == "rtsp://192.168.1.100/stream"
        assert cam.width == 640
        assert cam.trigger_mode == TriggerMode.SOFTWARE

    @patch("src.camera.cv2")
    def test_init_trigger_mode_parsing(self, mock_cv2):
        cam = CameraCapture(trigger_mode="hardware")
        assert cam.trigger_mode == TriggerMode.HARDWARE

        cam2 = CameraCapture(trigger_mode="continuous")
        assert cam2.trigger_mode == TriggerMode.CONTINUOUS

    @patch("src.camera.cv2")
    def test_open_camera_success(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4

        cam = CameraCapture(source="0")
        result = cam.open()
        assert result is True
        mock_cv2.VideoCapture.assert_called_once_with(0)

    @patch("src.camera.cv2")
    def test_open_camera_failure(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4

        cam = CameraCapture(source="0")
        result = cam.open()
        assert result is False

    @patch("src.camera.cv2")
    def test_snapshot_returns_frame(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4

        cam = CameraCapture(source="0", trigger_mode="software")
        cam.open()
        frame = cam.snapshot()
        assert frame is not None
        assert frame.shape == (480, 640, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
