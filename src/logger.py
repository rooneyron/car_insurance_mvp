"""
统一日志配置
使用方式：from src.logger import logger
"""

import logging
import os
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

    # ---------- 传输/SDK 层诊断：openai / httpx / httpcore 三层调到 DEBUG ----------
    # 由外到内三层：openai SDK（请求构造 + 重试决策，如“为何重试/不重试”、Encountered Exception）
    #             → httpx（HTTP 客户端）→ httpcore（连接生命周期：TCP/TLS/收发/断开）。
    # 三层一起开，才能看清“服务端中途断连”“重试与否”这类底层真相。
    # 单独设这几个 logger 的级别，不影响 root=INFO（其余模块日志仍精简）。
    # 环境变量 LLM_HTTP_DEBUG=1 开启（诊断传输层时）；默认 0=关闭，避免刷屏及请求头(含 Authorization)泄密。
    # ⚠ 安全：openai DEBUG 的 “Request options” 可能打印请求头（含 Authorization），日志勿外泄。
    if os.environ.get("LLM_HTTP_DEBUG", "0") == "1":
        logging.getLogger("openai").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
