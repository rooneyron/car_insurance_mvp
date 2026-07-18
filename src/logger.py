"""
统一日志配置
使用方式：from src.logger import logger
"""

import logging
import sys
from src.logging_filters import TraceIdFilter


def setup_logging(level: str = "INFO"):
    """
    配置全局日志格式。应在应用启动时调用一次。
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | [%(trace_id)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(TraceIdFilter())

    root = logging.getLogger()
    root.setLevel(log_level)
    if not root.handlers:
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
