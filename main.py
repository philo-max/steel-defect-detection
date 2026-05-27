"""
主入口 - 钢铁表面缺陷检测系统。

用法:
    python main.py                  # 启动 Gradio 工作台
    python main.py --mode cli       # 命令行模式
    python main.py --mode detect    # 单张图像检测
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = PROJECT_ROOT / config_file
    if not config_file.exists():
        logger.warning(f"配置文件未找到: {config_file}，使用默认配置")
        return {}
    
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def start_monitoring(config: dict):
    """
    启动系统监控
    
    从配置文件读取监控配置，初始化 SystemMonitor 并启动后台监控线程。
    监控在后台运行，不影响主业务流程。
    """
    monitor_config = config.get("monitor", {})
    
    if not monitor_config.get("enabled", True):
        logger.info("系统监控已禁用（config.yaml monitor.enabled=false）")
        return None
    
    try:
        from src.monitor import SystemMonitor
        
        monitor = SystemMonitor(monitor_config)
        
        # 相机状态提供者：从 app.py 中获取 camera 实例的状态
        # 这里先不注入，在 app.py 的 launch() 中通过 setter 注入
        # monitor.set_camera_status_provider(get_camera_status)
        
        if monitor.start():
            logger.info("系统监控已集成启动")
        else:
            logger.warning("系统监控启动失败")
        
        return monitor
    except ImportError as e:
        logger.error(f"监控模块导入失败: {e}")
        return None
    except Exception as e:
        logger.error(f"启动监控异常: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="钢铁表面缺陷检测系统")
    parser.add_argument(
        "--mode",
        choices=["app", "cli", "detect", "export"],
        default="app",
        help="运行模式 (default: app)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="配置文件路径 (default: config.yaml)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="检测模式: 输入图像路径",
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="摄像头源 (default: 0)",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="禁用系统监控",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 启动系统监控（可通过 --no-monitor 禁用）
    monitor = None
    if not args.no_monitor:
        monitor_config = config.get("monitor", {})
        if monitor_config.get("enabled", True):
            try:
                from src.monitor import SystemMonitor
                monitor = SystemMonitor(monitor_config)
                monitor.start()
                logger.info("系统监控已启动")
            except ImportError as e:
                logger.warning(f"监控模块不可用: {e}")
            except Exception as e:
                logger.error(f"监控启动失败: {e}")

    if args.mode == "app":
        from app import launch
        launch(args.config, monitor=monitor)
    elif args.mode == "cli":
        from cli import run_cli
        run_cli(args.config)
    elif args.mode == "detect":
        from cli import run_detect
        run_detect(args.config, args.image)
    elif args.mode == "export":
        from cli import run_export
        run_export(args.config)

    # 清理
    if monitor:
        monitor.stop()


if __name__ == "__main__":
    main()
