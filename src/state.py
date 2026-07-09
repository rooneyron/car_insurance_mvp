"""
全局状态管理
存储编译后的 LangGraph 编排图，供各模块共享引用。
"""

# 编译后的 StateGraph 编排图
graph = None


def set_graph(compiled_graph):
    """设置全局编排图引用"""
    global graph
    graph = compiled_graph
