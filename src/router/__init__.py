"""
四层漏斗路由器
L0 安全拦截 → L1 关键词 → L2 小模型分类 → L3 大模型复核 → L4 澄清/转人工
"""

from src.router.router import route_message, RouterConfig
from src.router.schemas import RouterResult, RouterState

__all__ = ["route_message", "RouterConfig", "RouterResult", "RouterState"]
