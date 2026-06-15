"""
统一日志配置 — 基于 loguru，支持文件轮转和控制台输出。
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> None:
    """配置全局日志系统"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 移除默认 handler
    logger.remove()

    # 控制台输出 (彩色)
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # 文件输出 (按天轮转，保留 30 天)
    logger.add(
        log_path / "steel_defect_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        encoding="utf-8",
    )

    # 错误日志单独文件
    logger.add(
        log_path / "error_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="10 MB",
        retention="90 days",
        encoding="utf-8",
    )

    logger.info(f"日志系统初始化完成，日志目录: {log_path.absolute()}")
