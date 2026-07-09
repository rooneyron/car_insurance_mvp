"""
统一日志配置
使用方式：from src.logger import logger
"""

import logging
import sys

def setup_logging(level: str = "INFO"):
    """
    配置全局日志格式。应在应用启动时调用一次。
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    # 避免重复添加 handler
    if not root.handlers:
        root.addHandler(handler)


# 模块级 logger，各模块通过 get_logger 获取自己的 logger
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
