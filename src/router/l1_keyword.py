"""
L1：关键词层
只拦截「极其明确」的高频请求。
命中 high 词 + 无否定 + 唯一意图 → 直接返回。
其他所有情况 → 透传 L2。
"""

import os
import re
import yaml
from typing import Optional, Tuple
from src.constants import INTENT_SALE, INTENT_SERVICE, ROUTER_CONFIG_PATH, SOURCE_L1_KEYWORD
from src.logger import get_logger

logger = get_logger(__name__)

# ---------- 否定表达检测 ----------

_NEGATION_WORDS = [
    "不想", "不要", "不买", "不续", "不投", "不办",
    "不是", "没有", "没想", "别给", "别给我", "无需",
    "不需要", "用不着", "算了", "取消",
]

# 否定词与关键词的最大距离（字符数）
_NEGATION_DISTANCE = 6


def _has_negation(message: str, keyword: str) -> bool:
    """检测关键词附近是否存在否定表达"""
    for neg in _NEGATION_WORDS:
        if neg not in message:
            continue
        neg_pos = message.find(neg)
        kw_pos = message.find(keyword)
        if kw_pos >= 0 and abs(neg_pos - kw_pos) <= _NEGATION_DISTANCE:
            return True
    return False


# ---------- 加载关键词配置 ----------

def _load_keyword_config() -> dict:
    if not os.path.exists(ROUTER_CONFIG_PATH):
        return {}
    with open(ROUTER_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("router", {}).get("keyword", {})


_keyword_config: dict = _load_keyword_config()


def reload_keyword_config():
    global _keyword_config
    _keyword_config = _load_keyword_config()


# ---------- 检测函数 ----------

def check_keywords(message: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    关键词层检测。
    返回 (intent, source, is_unique) 或 (None, None, False)。

    is_unique: 是否命中唯一业务意图（无歧义、无否定）。
    用于 DST 切换检测：只有 is_unique=True 且 intent != current_task 才认为是 switch。

    规则：
    - 命中 high 词 + 无否定 + 唯一意图 → ("sale"/"service", "l1_keyword", True)
    - 其他所有情况 → (None, None, False)，透传 L2
    """
    high = _keyword_config.get("high", {})
    mid = _keyword_config.get("mid", {})

    high_sales = high.get("sales", [])
    high_after_sales = high.get("after_sales", [])

    mid_sales = mid.get("sales", [])
    mid_after_sales = mid.get("after_sales", [])

    msg = message.lower()

    # ---- 检查 high 词 ----
    hit_sales = False
    hit_after_sales = False
    hit_sales_kw = ""
    hit_after_sales_kw = ""

    for kw in high_sales:
        if kw in msg:
            hit_sales = True
            hit_sales_kw = kw
            break

    for kw in high_after_sales:
        if kw in msg:
            hit_after_sales = True
            hit_after_sales_kw = kw
            break

    # 情况1：同时命中两个意图 → L2
    if hit_sales and hit_after_sales:
        logger.info("[L1] high 多意图命中 (sale + after_sales)，透传 L2")
        return None, None, False

    # 情况2：命中唯一 high 意图
    if hit_sales:
        if _has_negation(message, hit_sales_kw):
            logger.info("[L1] high sale 命中但存在否定表达，透传 L2")
            return None, None, False
        logger.info("[L1] high sale 直接拦截: keyword='%s'", hit_sales_kw)
        return INTENT_SALE, SOURCE_L1_KEYWORD, True

    if hit_after_sales:
        if _has_negation(message, hit_after_sales_kw):
            logger.info("[L1] high after_sales 命中但存在否定表达，透传 L2")
            return None, None, False
        logger.info("[L1] high after_sales 直接拦截: keyword='%s'", hit_after_sales_kw)
        return INTENT_SERVICE, SOURCE_L1_KEYWORD, True

    # ---- 检查 mid 词（命中 → 一律透传 L2）----
    for kw in mid_sales:
        if kw in msg:
            logger.info("[L1] mid sale 命中，透传 L2")
            return None, None, False

    for kw in mid_after_sales:
        if kw in msg:
            logger.info("[L1] mid after_sales 命中，透传 L2")
            return None, None, False

    # 无关键词命中 → L2
    logger.debug("[L1] 无关键词命中，透传 L2")
    return None, None, False
