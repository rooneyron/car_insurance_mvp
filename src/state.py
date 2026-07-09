"""
全局状态管理
存储预加载的 Chain 实例，供各模块共享引用。
"""

# 预加载时初始化的 Chain 实例
general_chain = None
agent_sale = None
agent_service = None


def set_chains(general, sale, service):
    """设置全局 Chain 引用"""
    global general_chain, agent_sale, agent_service
    general_chain = general
    agent_sale = sale
    agent_service = service
