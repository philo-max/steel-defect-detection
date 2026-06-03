"""
命令行工具 - 支持单张检测、批量导出和运维命令。

用法:
    python cli.py detect --image path/to/image.jpg
    python cli.py export
    python cli.py verify
    python cli.py status
    python cli.py switch-model path/to/model.pt
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import yaml
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()


def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==================== 检测 ====================

def run_detect(config_path: str, image_path: str | None):
    """单张图像检测"""
    if image_path is None:
        print("请使用 --image 参数指定图像路径")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    cfg = _load_config(config_path)
    yolo_cfg = cfg.get("yolo", {})
    clahe_cfg = yolo_cfg.get("clahe", {})

    from src.detection_engine import YOLODetector
    detector = YOLODetector(
        model_path=yolo_cfg.get("model_path", "models/weights/yolov8n.pt"),
        conf_threshold=yolo_cfg.get("conf_threshold", 0.25),
        iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
        img_size=yolo_cfg.get("img_size", 640),
        device=yolo_cfg.get("device", "cuda:0"),
        half=yolo_cfg.get("half", False),
        clahe_enabled=clahe_cfg.get("enabled", False),
        clahe_clip_limit=clahe_cfg.get("clip_limit", 2.0),
        clahe_tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
    )

    try:
        detector.load_model()
    except FileNotFoundError:
        print("[FAIL] YOLO 模型文件未找到，请先放置模型权重到 models/weights/")
        return

    result = detector.detect(image)

    print(f"\n{'='*50}")
    print(f"检测完成 | 耗时: {result.inference_time_ms:.1f}ms")
    print(f"缺陷数量: {result.defect_count}")

    if result.detections:
        print("\n检测结果:")
        for i, det in enumerate(result.detections, 1):
            print(f"  {i}. {det.class_name} (置信度: {det.confidence:.2f})")
            print(f"     位置: [{det.bbox[0]:.3f}, {det.bbox[1]:.3f}, {det.bbox[2]:.3f}, {det.bbox[3]:.3f}]")
    else:
        print("未检测到缺陷")

    if result.error:
        print(f"\n错误: {result.error}")
    print(f"{'='*50}\n")


# ==================== 导出 ====================

def run_export(config_path: str):
    """导出检测数据"""
    cfg = _load_config(config_path)
    from src.db_manager import DBManager
    from src.exporter import Exporter
    db = DBManager(cfg.get("database", {}).get("path", "data/inspection.db"))
    exporter = Exporter(db)

    csv_path = exporter.export_csv()
    html_path = exporter.export_html_report()
    badcase_dir = exporter.export_badcase()

    print(f"[OK] CSV 导出: {csv_path}")
    print(f"[OK] HTML 报告: {html_path}")
    print(f"[OK] Bad Case: {badcase_dir}")


# ==================== 环境验证 ====================

def run_verify(config_path: str):
    """验证系统环境：模型、数据库、依赖、相机"""
    cfg = _load_config(config_path)
    issues = []

    print("钢铁表面缺陷检测系统 - 环境验证")
    print("=" * 50)

    # 1. Python 版本
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 8):
        print(f"[OK] Python {py_ver}")
    else:
        print(f"[FAIL] Python {py_ver} (需要 >= 3.8)")
        issues.append("Python 版本过低")

    # 2. 核心依赖
    deps = {
        "torch": "PyTorch",
        "ultralytics": "Ultralytics YOLO",
        "cv2": "OpenCV",
        "gradio": "Gradio",
        "openai": "OpenAI SDK",
        "pymodbus": "pymodbus (PLC)",
        "psutil": "psutil (监控)",
    }
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[WARN] {name} 未安装")
            if module in ("torch", "cv2", "gradio"):
                issues.append(f"{name} 未安装")

    # 3. CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[OK] CUDA: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("[WARN] CUDA 不可用，将使用 CPU 推理")
    except ImportError:
        print("[WARN] PyTorch 未安装，无法检查 CUDA")

    # 4. YOLO 模型
    yolo_cfg = cfg.get("yolo", {})
    model_path = Path(yolo_cfg.get("model_path", "yolov8n.pt"))
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1024**2
        print(f"[OK] YOLO 模型: {model_path.name} ({size_mb:.1f} MB)")
    else:
        print(f"[FAIL] YOLO 模型未找到: {model_path}")
        issues.append("YOLO 模型文件缺失")

    # 5. 数据库
    db_path = Path(cfg.get("database", {}).get("path", "data/inspection.db"))
    if not db_path.is_absolute():
        db_path = Path(__file__).parent / db_path
    if db_path.exists():
        size_mb = db_path.stat().st_size / 1024**2
        print(f"[OK] 数据库: {db_path} ({size_mb:.2f} MB)")
    else:
        print(f"[INFO] 数据库将自动创建: {db_path}")

    # 6. VLM API Key & 延迟测算
    vlm_cfg = cfg.get("vlm", {})
    if vlm_cfg.get("enabled", True):
        from src.vlm_engine import _detect_provider
        provider, api_key, api_base, model = _detect_provider()
        if provider != "none" and api_base:
            print(f"[OK] VLM API 节点已配置: 提供商={provider}, 接口地址={api_base}")
            
            import urllib.request
            import time
            from urllib.parse import urlparse
            from urllib.error import HTTPError
            
            try:
                parsed_url = urlparse(api_base)
                domain = parsed_url.netloc or parsed_url.path.split('/')[0]
                
                print(f"[INFO] 正在测试 VLM API 节点网络连接时延 ({domain})...")
                latencies = []
                success_count = 0
                for _ in range(3):
                    start_t = time.perf_counter()
                    try:
                        req = urllib.request.Request(
                            api_base,
                            method="GET" if provider == "gemini" else "HEAD"
                        )
                        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
                        with urllib.request.urlopen(req, timeout=5) as response:
                            response.read(1)
                        latency = (time.perf_counter() - start_t) * 1000
                        latencies.append(latency)
                        success_count += 1
                    except HTTPError:
                        # 收到 HTTP 状态码错误也算作成功的网络往返时间测量
                        latency = (time.perf_counter() - start_t) * 1000
                        latencies.append(latency)
                        success_count += 1
                    except Exception:
                        pass
                
                if success_count > 0:
                    avg_latency = sum(latencies) / len(latencies)
                    print(f"[OK] VLM API 平均连接延迟: {avg_latency:.1f}ms (成功率: {success_count}/3)")
                    if avg_latency > 1000:
                        print(f"[WARN] 延迟较高 (>1000ms)，在线质检复核可能存在响应卡顿")
                else:
                    print(f"[FAIL] 无法连接到 VLM API 节点 ({domain})，请求超时或解析失败")
                    issues.append(f"VLM API 节点连接失败 ({domain})")
            except Exception as test_err:
                print(f"[WARN] 网络延迟测试出现异常: {test_err}")
        else:
            print("[WARN] 未检测到有效的 VLM API Key，VLM 复核不可用")

    # 7. 相机
    cam_cfg = cfg.get("camera", {})
    source = cam_cfg.get("source", "0")
    try:
        cap = cv2.VideoCapture(source if source != "0" else 0)
        if cap.isOpened():
            print(f"[OK] 相机连接: {source}")
            cap.release()
        else:
            print(f"[WARN] 相机无法连接: {source}")
    except Exception as e:
        print(f"[WARN] 相机检测失败: {e}")

    print("=" * 50)
    if issues:
        print(f"发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    else:
        print("环境验证通过!")
        return 0


# ==================== 状态查看 ====================

def run_status(config_path: str):
    """检查各模块运行状态"""
    cfg = _load_config(config_path)

    print("钢铁表面缺陷检测系统 - 模块状态")
    print("=" * 50)

    # YOLO
    yolo_cfg = cfg.get("yolo", {})
    model_path = yolo_cfg.get("model_path", "yolov8n.pt")
    print(f"  YOLO 模型:  {model_path}")
    print(f"  置信度阈值: {yolo_cfg.get('conf_threshold', 0.25)}")
    print(f"  IOU 阈值:   {yolo_cfg.get('iou_threshold', 0.45)}")
    print(f"  输入尺寸:   {yolo_cfg.get('img_size', 640)}")
    print(f"  设备:       {yolo_cfg.get('device', 'auto')}")

    # VLM
    vlm_cfg = cfg.get("vlm", {})
    print(f"\n  VLM 启用:   {vlm_cfg.get('enabled', True)}")
    print(f"  VLM 模型:   {vlm_cfg.get('model') or '自动选择'}")
    print(f"  超时:       {vlm_cfg.get('timeout', 60)}s")

    # 相机
    cam_cfg = cfg.get("camera", {})
    print(f"\n  相机源:     {cam_cfg.get('source', '0')}")
    print(f"  分辨率:     {cam_cfg.get('width', 1920)}x{cam_cfg.get('height', 1080)}")
    print(f"  帧率:       {cam_cfg.get('fps', 30)} FPS")
    print(f"  触发模式:   {cam_cfg.get('trigger_mode', 'continuous')}")

    # PLC
    plc_cfg = cfg.get("plc", {})
    print(f"\n  PLC 启用:   {plc_cfg.get('enabled', False)}")
    if plc_cfg.get("enabled"):
        print(f"  PLC 地址:   {plc_cfg.get('host', '')}:{plc_cfg.get('port', 502)}")

    # 数据库
    db_cfg = cfg.get("database", {})
    db_path = Path(db_cfg.get("path", "data/inspection.db"))
    if not db_path.is_absolute():
        db_path = Path(__file__).parent / db_path
    if db_path.exists():
        from src.db_manager import DBManager
        db = DBManager(str(db_path))
        count = db.count()
        print(f"\n  数据库:     {db_path}")
        print(f"  记录总数:   {count}")
    else:
        print(f"\n  数据库:     未创建")

    # 监控
    mon_cfg = cfg.get("monitor", {})
    print(f"\n  监控启用:   {mon_cfg.get('enabled', True)}")
    print(f"  检查间隔:   {mon_cfg.get('check_interval', 60)}s")

    # Gradio
    gradio_cfg = cfg.get("gradio", {})
    print(f"\n  Web 地址:   http://{gradio_cfg.get('server_name', '0.0.0.0')}:{gradio_cfg.get('server_port', 7860)}")
    print(f"  认证:       {'已启用' if gradio_cfg.get('auth') else '未启用'}")

    print("=" * 50)


# ==================== 模型切换 ====================

def run_switch_model(config_path: str, new_model_path: str):
    """切换 YOLO 模型权重"""
    cfg = _load_config(config_path)
    yolo_cfg = cfg.get("yolo", {})

    # 验证新模型文件存在
    model_file = Path(new_model_path)
    if not model_file.exists():
        print(f"[FAIL] 模型文件不存在: {new_model_path}")
        return 1

    # 备份当前配置
    old_model = yolo_cfg.get("model_path", "yolov8n.pt")
    backup_path = config_path + ".bak"
    shutil.copy2(config_path, backup_path)
    print(f"[OK] 已备份配置: {backup_path}")

    # 更新配置
    safe_path = new_model_path.replace("\\", "/")
    with open(config_path, encoding="utf-8") as f:
        lines = f.readlines()

    with open(config_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip().startswith("model_path:"):
                f.write(f'  model_path: "{safe_path}"\n')
            else:
                f.write(line)

    print(f"[OK] 模型已切换: {old_model} -> {new_model_path}")

    # 验证新模型可加载
    try:
        from ultralytics import YOLO
        YOLO(new_model_path)
        print(f"[OK] 新模型加载验证通过")
    except Exception as e:
        print(f"[WARN] 新模型加载验证失败: {e}")
        print(f"       配置已更新，请手动确认模型兼容性")

    return 0


# ==================== 入口 ====================

def run_cli(config_path: str):
    """CLI 入口 - 解析子命令并分发"""
    parser = argparse.ArgumentParser(
        description="钢铁表面缺陷检测系统 CLI",
        prog="cli.py",
    )
    parser.add_argument(
        "--config", default=config_path,
        help="配置文件路径",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # detect
    p_detect = subparsers.add_parser("detect", help="单张图像检测")
    p_detect.add_argument("--image", required=True, help="图像路径")

    # export
    subparsers.add_parser("export", help="导出检测数据")

    # verify
    subparsers.add_parser("verify", help="验证系统环境")

    # status
    subparsers.add_parser("status", help="查看模块状态")

    # switch-model
    p_switch = subparsers.add_parser("switch-model", help="切换 YOLO 模型")
    p_switch.add_argument("model_path", help="新模型文件路径")

    args, _ = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        return

    cfg = args.config or config_path

    if args.command == "detect":
        run_detect(cfg, args.image)
    elif args.command == "export":
        run_export(cfg)
    elif args.command == "verify":
        sys.exit(run_verify(cfg))
    elif args.command == "status":
        run_status(cfg)
    elif args.command == "switch-model":
        sys.exit(run_switch_model(cfg, args.model_path))


if __name__ == "__main__":
    run_cli("config.yaml")
