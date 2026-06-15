"""
主入口 - 钢铁表面缺陷检测系统。

用法:
    python main.py                  # 启动 Gradio 工作台
    python main.py --mode cli       # 命令行模式
    python main.py --mode detect    # 单张图像检测
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


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
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "app":
        import os
        import uvicorn
        import yaml
        from src.utils.logging_config import setup_logging
        from server import app
        # 初始化日志系统
        log_level = os.getenv("LOG_LEVEL", "INFO")
        setup_logging(log_level=log_level)
        # Load config
        host = "0.0.0.0"
        port = 7860
        try:
            with open(args.config, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            server_cfg = config.get("server", config.get("gradio", {}))
            host = server_cfg.get("host", server_cfg.get("server_name", "0.0.0.0"))
            port = server_cfg.get("port", server_cfg.get("server_port", 7860))
        except Exception as e:
            print(f"[WARN] 无法读取配置文件: {e}")
        # 开发环境下启用自动重载，生产环境禁用
        use_reload = os.getenv("ENV", "development") == "development"
        print(f"[INFO] 正在启动 API & 静态前端服务器: http://{host}:{port}")
        uvicorn.run("server:app", host=host, port=port, reload=use_reload)
    elif args.mode == "cli":
        from cli import run_cli
        run_cli(args.config)
    elif args.mode == "detect":
        from cli import run_detect
        run_detect(args.config, args.image)
    elif args.mode == "export":
        from cli import run_export
        run_export(args.config)
    else:
        print(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()
