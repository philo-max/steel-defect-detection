"""
TensorRT 推理引擎 — 工业级实时加速后端。

功能：
1. PyTorch → ONNX → TensorRT 一键导出
2. TensorRT Engine 加载与异步推理
3. 兼容 BaseDetector 接口，可无缝替换 YOLODetector
4. FP16/INT8 量化支持
5. 动态 Batch / 静态 Shape 双模式

依赖: tensorrt, pycuda (需手动安装，对应 CUDA 版本)

使用方式:
    from src.tensorrt_engine import TensorRTDetector
    detector = TensorRTDetector(engine_path="model.engine")
    detector.load_model()
    result = detector.detect(image)
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .base_detector import BaseDetector, InferenceResult, DetectionResult

# ==================== TensorRT 延迟导入（允许无 GPU 环境仍可 import 此模块） ====================

_trt_available = False
_cuda_available = False

try:
    import tensorrt as trt  # pyright: ignore[reportMissingImports]
    _trt_available = True
except ImportError:
    trt = None  # type: ignore[assignment]

try:
    import pycuda.driver as cuda  # pyright: ignore[reportMissingImports]
    import pycuda.autoinit  # pyright: ignore[reportMissingImports]  # noqa: F401
    _cuda_available = True
except ImportError:
    cuda = None  # type: ignore[assignment]


# ==================== NEU-DET 6 类标签映射 ====================

NEU_DET_CLASSES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}


# ==================== ONNX 导出 ====================

def export_onnx(
    pt_path: str,
    onnx_path: Optional[str] = None,
    img_size: int = 640,
    batch_size: int = 1,
    opset: int = 12,
    simplify: bool = True,
    half: bool = False,
) -> str:
    """
    将 YOLO .pt 模型导出为 ONNX 格式。

    Args:
        pt_path: PyTorch 权重路径 (.pt)
        onnx_path: 输出 ONNX 路径（默认与 pt 同目录同名）
        img_size: 输入尺寸
        batch_size: 批次大小
        opset: ONNX opset 版本
        simplify: 是否使用 onnx-simplifier 简化图
        half: 是否导出 FP16

    Returns:
        ONNX 文件路径
    """
    from ultralytics import YOLO

    if onnx_path is None:
        onnx_path = str(Path(pt_path).with_suffix(".onnx"))

    model = YOLO(pt_path)
    model.export(
        format="onnx",
        imgsz=img_size,
        batch=batch_size,
        opset=opset,
        simplify=simplify,
        half=half,
        dynamic=False,
    )

    # ultralytics 导出路径可能与指定不同，移动文件
    default_onnx = str(Path(pt_path).with_suffix(".onnx"))
    if default_onnx != onnx_path and os.path.exists(default_onnx):
        import shutil
        shutil.move(default_onnx, onnx_path)

    print(f"[TensorRT] ONNX 导出完成: {onnx_path}")
    return onnx_path


# ==================== TensorRT Engine 构建 ====================

def build_engine(
    onnx_path: str,
    engine_path: Optional[str] = None,
    fp16: bool = True,
    int8: bool = False,
    int8_calib_dir: Optional[str] = None,
    workspace_gb: int = 4,
    max_batch_size: int = 1,
) -> str:
    """
    从 ONNX 构建 TensorRT Engine。

    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出 Engine 路径
        fp16: 启用 FP16 精度
        int8: 启用 INT8 量化（需要校准数据）
        int8_calib_dir: INT8 校准图像目录
        workspace_gb: 最大工作空间 (GB)
        max_batch_size: 最大批次大小

    Returns:
        Engine 文件路径
    """
    if not _trt_available:
        raise ImportError("tensorrt 未安装。pip install tensorrt")

    if engine_path is None:
        engine_path = str(Path(onnx_path).with_suffix(".engine"))

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    # 解析 ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(
                f"  [{parser.get_error(i).code()}] {parser.get_error(i).desc()}"
                for i in range(parser.num_errors)
            )
            raise RuntimeError(f"ONNX 解析失败:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1024 ** 3))

    # 精度配置
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        _set_int8_calibrator(config, network, int8_calib_dir, onnx_path)

    if fp16 and not int8:
        config.set_flag(trt.BuilderFlag.FP16)

    # 序列化 Engine
    print(f"[TensorRT] 正在构建 Engine (fp16={fp16}, int8={int8})...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("TensorRT Engine 构建失败")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"[TensorRT] Engine 构建完成: {engine_path}")
    return engine_path


def _set_int8_calibrator(config, network, calib_dir, onnx_path):
    """设置 INT8 校准器（使用随机数据作为简化回退）"""
    # 简化版：使用 EntropyCalibrator2
    # 生产环境建议使用真实产线图像
    try:
        import cv2
        calib_images = _load_calibration_images(calib_dir, onnx_path)
        if calib_images:
            config.int8_calibrator = _Int8Calibrator(
                calib_images, network, batch_size=1
            )
            print(f"[TensorRT] INT8 校准器就绪 ({len(calib_images)} 张图像)")
        else:
            print("[TensorRT] 警告: 未找到校准图像，使用默认 INT8 策略")
    except Exception as e:
        print(f"[TensorRT] INT8 校准器初始化失败: {e}")


def _load_calibration_images(calib_dir: Optional[str], onnx_path: str) -> list:
    """加载 INT8 校准图像"""
    images = []
    search_dirs = [calib_dir] if calib_dir else []

    # 回退：尝试使用 NEU-DET 训练集
    neu_paths = [
        Path(onnx_path).parent.parent / "data" / "datasets" / "neu_det" / "images" / "train",
        Path("data/datasets/neu_det/images/train"),
    ]
    for p in neu_paths:
        if p.exists():
            search_dirs.append(str(p))
            break

    import cv2
    for d in search_dirs:
        if d and os.path.isdir(d):
            files = list(Path(d).glob("*.jpg")) + list(Path(d).glob("*.png")) + list(Path(d).glob("*.bmp"))
            for f in files[:100]:  # 最多 100 张
                img = cv2.imread(str(f))
                if img is not None:
                    images.append(img)
            if images:
                break

    return images


if _trt_available:

    class _Int8Calibrator(trt.IInt8EntropyCalibrator2):
        """INT8 熵校准器"""

        def __init__(self, images, network, batch_size=1):
            super().__init__()
            self.images = images
            self.batch_size = batch_size
            self._iterator = iter(self.images)

            # 分配校准批次缓存
            input_shape = network.get_input(0).shape
            c, h, w = input_shape[1], input_shape[2], input_shape[3]
            self._calib_batch = np.zeros((batch_size, c, h, w), dtype=np.float32)
            self._device_mem = cuda.mem_alloc(self._calib_batch.nbytes)

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            try:
                import cv2
                img = next(self._iterator)
                img = cv2.resize(img, (640, 640))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
                self._calib_batch[0] = img
                cuda.memcpy_htod(self._device_mem, self._calib_batch)
                return [int(self._device_mem)]
            except StopIteration:
                return None

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache):
            pass

else:
    # 无 TensorRT 时的占位，保持与真实类相同的构造签名
    class _Int8Calibrator:
        def __init__(self, images=None, network=None, batch_size=1):
            pass
        def get_batch_size(self):
            return 1
        def get_batch(self, names):
            return None
        def read_calibration_cache(self):
            return None
        def write_calibration_cache(self, cache):
            pass


# ==================== TensorRT 检测器 ====================

class TensorRTDetector(BaseDetector):
    """TensorRT 推理检测器 — 兼容 BaseDetector 接口"""

    def __init__(
        self,
        engine_path: str = "models/weights/yolov8n.engine",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        class_names: Optional[dict] = None,
    ):
        super().__init__(name="tensorrt")
        self.engine_path = engine_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.class_names = class_names or NEU_DET_CLASSES

        self._engine: Optional[trt.ICudaEngine] = None
        self._context: Optional[trt.IExecutionContext] = None
        self._inputs: list = []
        self._outputs: list = []
        self._bindings: list = []
        self._stream: Optional[cuda.Stream] = None
        self._input_shape: tuple = ()

    # ---- 导出/构建入口 ----

    @staticmethod
    def from_pt(
        pt_path: str,
        engine_path: Optional[str] = None,
        img_size: int = 640,
        fp16: bool = True,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> "TensorRTDetector":
        """
        一键从 .pt 导出 ONNX → 构建 TensorRT Engine → 返回检测器。

        这是最便捷的入口，适合训练后直接部署。
        """
        if engine_path is None:
            engine_path = str(Path(pt_path).with_suffix(".engine"))

        # 检查 Engine 是否已存在且比 PT 新
        pt_mtime = os.path.getmtime(pt_path) if os.path.exists(pt_path) else 0
        engine_exists = os.path.exists(engine_path)
        engine_mtime = os.path.getmtime(engine_path) if engine_exists else 0

        if not engine_exists or engine_mtime < pt_mtime:
            onnx_path = str(Path(pt_path).with_suffix(".tmp.onnx"))
            try:
                export_onnx(pt_path, onnx_path, img_size=img_size)
                build_engine(onnx_path, engine_path, fp16=fp16)
            finally:
                if os.path.exists(onnx_path):
                    os.remove(onnx_path)
        else:
            print(f"[TensorRT] Engine 已是最新，跳过构建: {engine_path}")

        return TensorRTDetector(
            engine_path=engine_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            img_size=img_size,
        )

    # ---- 检测器接口 ----

    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """加载 TensorRT Engine"""
        if not _trt_available or not _cuda_available:
            raise RuntimeError(
                "TensorRT 推理需要: pip install tensorrt pycuda\n"
                "若无 GPU 环境，请使用 YOLODetector (PyTorch 后端)"
            )

        if model_path:
            self.engine_path = model_path

        engine_file = Path(self.engine_path)
        if not engine_file.exists():
            # 尝试从 .pt 自动构建
            pt_path = engine_file.with_suffix(".pt")
            if pt_path.exists():
                print(f"[TensorRT] Engine 未找到，从 {pt_path} 自动构建...")
                TensorRTDetector.from_pt(str(pt_path), str(engine_file), self.img_size)
            else:
                raise FileNotFoundError(f"Engine 文件不存在: {self.engine_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError(f"Engine 加载失败: {self.engine_path}")

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()
        self._allocate_buffers()

        self._warm = False

    def _allocate_buffers(self):
        """分配主机/设备缓冲区"""
        self._inputs.clear()
        self._outputs.clear()
        self._bindings.clear()

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = self._engine.get_tensor_shape(name)
            dtype = self._engine.get_tensor_dtype(name)

            # 处理动态维度（-1 → 1）
            shape = tuple(s if s != -1 else 1 for s in shape)
            size = int(np.prod(shape))
            np_dtype = _trt_dtype_to_numpy(dtype)
            item_size = np.dtype(np_dtype).itemsize

            host_mem = cuda.pagelocked_empty(size, np_dtype)
            device_mem = cuda.mem_alloc(size * item_size)
            self._bindings.append(int(device_mem))

            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._inputs.append({"name": name, "host": host_mem, "device": device_mem, "shape": shape, "dtype": np_dtype})
                self._input_shape = shape
            else:
                self._outputs.append({"name": name, "host": host_mem, "device": device_mem, "shape": shape, "dtype": np_dtype})

    def detect(self, image: np.ndarray) -> InferenceResult:
        """对单张图像执行 TensorRT 推理"""
        if self._engine is None:
            return InferenceResult(error="模型未加载，请先调用 load_model()")

        start = time.perf_counter()

        try:
            # 1. 预处理
            blob = self._preprocess(image)

            # 2. 主机 → 设备
            np.copyto(self._inputs[0]["host"], blob.ravel())
            cuda.memcpy_htod_async(
                self._inputs[0]["device"],
                self._inputs[0]["host"],
                self._stream,
            )

            # 3. 设置输入 Shape（支持动态 batch）
            self._context.set_tensor_address(self._inputs[0]["name"], int(self._inputs[0]["device"]))
            for out in self._outputs:
                self._context.set_tensor_address(out["name"], int(out["device"]))

            # 4. 异步推理
            self._context.execute_async_v3(stream_handle=self._stream.handle)

            # 5. 设备 → 主机
            for out in self._outputs:
                cuda.memcpy_dtoh_async(out["host"], out["device"], self._stream)

            self._stream.synchronize()

        except Exception as e:
            return InferenceResult(
                inference_time_ms=self._measure_time(start),
                error=f"TensorRT 推理异常: {e}",
            )

        elapsed = self._measure_time(start)

        # 6. 后处理
        raw_output = self._outputs[0]["host"].copy()
        detections = self._postprocess(raw_output, image.shape[:2])

        return InferenceResult(
            detections=detections,
            inference_time_ms=elapsed,
            image_shape=image.shape[:2],
            raw_output={"output_shape": list(self._outputs[0]["shape"])},
        )

    # ---- 预处理 / 后处理 ----

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """YOLO 标准预处理：resize → RGB → normalize → NCHW"""
        import cv2
        img = cv2.resize(image, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC → CHW
        return np.expand_dims(img, axis=0)  # [1, 3, H, W]

    def _postprocess(self, raw_output: np.ndarray, orig_shape: tuple) -> list[DetectionResult]:
        """
        YOLO 输出后处理（简化版 NMS）。

        raw_output shape: [1, 84, 8400] 或 [1, 8400, 84]
        其中前 4 列是 bbox，后面是类别 scores。
        """
        # 确定输出格式
        if len(raw_output.shape) == 3:
            raw_output = raw_output.squeeze(0)  # [84, 8400]

        # 转置为 [N, 84]
        if raw_output.shape[0] <= 84:
            raw_output = raw_output.T  # → [8400, 84]

        boxes = raw_output[:, :4]
        scores = raw_output[:, 4:]

        # 找每行的最佳类别
        class_ids = np.argmax(scores, axis=1)
        confs = np.max(scores, axis=1)

        # 置信度过滤
        mask = confs >= self.conf_threshold
        boxes = boxes[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # 转换 cxcywh → xyxy
        boxes_xyxy = self._cxcywh_to_xyxy(boxes)

        # 简单 NMS（按类别分别抑制）
        keep_indices = self._nms_per_class(boxes_xyxy, confs, class_ids)
        boxes_xyxy = boxes_xyxy[keep_indices]
        confs = confs[keep_indices]
        class_ids = class_ids[keep_indices]

        h, w = orig_shape
        detections = []
        for box, conf, cls_id in zip(boxes_xyxy, confs, class_ids):
            detections.append(DetectionResult(
                bbox=[
                    float(np.clip(box[0], 0, self.img_size)) / self.img_size,
                    float(np.clip(box[1], 0, self.img_size)) / self.img_size,
                    float(np.clip(box[2], 0, self.img_size)) / self.img_size,
                    float(np.clip(box[3], 0, self.img_size)) / self.img_size,
                ],
                class_name=self.class_names.get(int(cls_id), f"cls_{cls_id}"),
                confidence=float(conf),
                class_id=int(cls_id),
            ))

        return detections

    @staticmethod
    def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """[cx, cy, w, h] → [x1, y1, x2, y2]"""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    def _nms_per_class(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> list:
        """按类别 NMS（IoU 阈值）"""
        keep = []
        unique_classes = np.unique(class_ids)
        for cls_id in unique_classes:
            cls_mask = class_ids == cls_id
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            # 按置信度降序
            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_indices = cls_indices[order]

            while len(cls_boxes) > 0:
                keep.append(cls_indices[0])
                if len(cls_boxes) == 1:
                    break
                ious = self._compute_iou(cls_boxes[0:1], cls_boxes[1:])[0]
                mask = ious < self.iou_threshold
                cls_boxes = cls_boxes[1:][mask]
                cls_indices = cls_indices[1:][mask]

        return sorted(keep)

    @staticmethod
    def _compute_iou(box_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """计算 IoU (向量化)"""
        x1 = np.maximum(box_a[0, 0], boxes_b[:, 0])
        y1 = np.maximum(box_a[0, 1], boxes_b[:, 1])
        x2 = np.minimum(box_a[0, 2], boxes_b[:, 2])
        y2 = np.minimum(box_a[0, 3], boxes_b[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (box_a[0, 2] - box_a[0, 0]) * (box_a[0, 3] - box_a[0, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a + area_b - inter
        return inter / (union + 1e-6)

    # ---- 清理 ----

    def __del__(self):
        if hasattr(self, "_stream") and self._stream:
            self._stream.synchronize()
        self._engine = None
        self._context = None


# ==================== 工具函数 ====================

def _trt_dtype_to_numpy(dtype) -> np.dtype:
    """TRT dtype → numpy dtype"""
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL: np.bool_,
    }
    return mapping.get(dtype, np.float32)


def pt_to_trt(
    pt_path: str,
    output_dir: Optional[str] = None,
    img_size: int = 640,
    fp16: bool = True,
) -> str:
    """
    一键转换 PyTorch → ONNX → TensorRT Engine。

    CLI 友好入口，可直接 `python -m src.tensorrt_engine` 调用。
    """
    pt = Path(pt_path)
    if not pt.exists():
        raise FileNotFoundError(f"模型文件不存在: {pt_path}")

    out = Path(output_dir) if output_dir else pt.parent
    out.mkdir(parents=True, exist_ok=True)

    engine_name = pt.stem + (".fp16" if fp16 else ".fp32") + ".engine"
    engine_path = out / engine_name

    detector = TensorRTDetector.from_pt(
        str(pt), str(engine_path), img_size=img_size, fp16=fp16
    )
    detector.load_model()
    print(f"[TensorRT] 转换完成: {engine_path}")
    return str(engine_path)


# ==================== CLI 入口 ====================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch → TensorRT 转换工具")
    parser.add_argument("pt_path", help="PyTorch 权重路径 (.pt)")
    parser.add_argument("--output", "-o", default=None, help="输出 Engine 目录")
    parser.add_argument("--img-size", type=int, default=640, help="输入尺寸")
    parser.add_argument("--fp16", action="store_true", default=True, help="启用 FP16")
    parser.add_argument("--fp32", action="store_true", help="使用 FP32 (覆盖 --fp16)")
    args = parser.parse_args()

    use_fp16 = not args.fp32
    pt_to_trt(args.pt_path, args.output, args.img_size, use_fp16)
