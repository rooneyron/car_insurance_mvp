"""
日志过滤器：从 contextvars 注入 trace_id
"""
import logging
from src.context import trace_id_var


class TraceIdFilter(logging.Filter):
    def filter(self, record):
        record.trace_id = trace_id_var.get('-')
        return True
