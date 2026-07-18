"""
请求级上下文变量
"""
from contextvars import ContextVar

trace_id_var: ContextVar[str] = ContextVar('trace_id', default='-')


def set_trace_id(tid: str):
    trace_id_var.set(tid)


def get_trace_id() -> str:
    return trace_id_var.get('-')
