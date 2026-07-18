"""
全局状态管理
存储编译后的 LangGraph 编排图，供各模块共享引用。
"""

# 编译后的 StateGraph 编排图
graph = None

# 摘要函数（入口处调用，负责消息压缩）
summarize_fn = None


def set_graph(compiled_graph):
    """设置全局编排图引用"""
    global graph
    graph = compiled_graph


def set_summarize_fn(fn):
    """设置摘要函数引用"""
    global summarize_fn
    summarize_fn = fn
