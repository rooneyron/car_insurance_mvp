"""
L0：安全拦截层
纯规则匹配，不调用 LLM。
命中安全红线关键词 → 立即 handoff（转人工）。
"""

import os
import yaml
from typing import Optional
from src.constants import ROUTER_CONFIG_PATH, INTENT_HANDOFF
from src.logger import get_logger

logger = get_logger(__name__)

# ---------- 加载安全关键词 ----------

def _load_safety_keywords() -> list:
    if not os.path.exists(ROUTER_CONFIG_PATH):
        return []
    with open(ROUTER_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("router", {}).get("safety", [])


_safety_keywords: list = _load_safety_keywords()


def reload_safety_keywords():
    """热更新安全关键词（供外部调用）"""
    global _safety_keywords
    _safety_keywords = _load_safety_keywords()


# ---------- 检测函数 ----------

def check_safety(message: str) -> Optional[str]:
    """
    检查消息是否命中安全红线。
    命中返回 "handoff"，未命中返回 None。
    """
    msg_lower = message.lower()
    for kw in _safety_keywords:
        if kw in msg_lower:
            logger.info("[L0] 安全拦截命中: keyword='%s'", kw)
            return INTENT_HANDOFF
    return None
