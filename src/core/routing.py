import sys
import os
# 获取当前文件所在目录的父目录的父目录（即项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import re
import yaml
import os
from typing import Optional, Literal
from route_types import Route, ROUTE_LABELS
from src.logger import get_logger

logger = get_logger(__name__)


# ---------- 1. 加载配置文件（支持热更新） ----------
CONFIG_PATH = "config/config.yaml"

def load_keywords():
    """从 config.yaml 读取关键词列表"""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"配置文件 {CONFIG_PATH} 不存在，请先创建。")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["keywords"]

# 全局加载（后续可以加上文件监听实现热加载，MVP 阶段先手动重启）
keywords = load_keywords()
SALE_KEYWORDS = keywords.get("sale", [])
SERVICE_KEYWORDS = keywords.get("service", [])
FORCE_SERVICE_KEYWORDS = keywords.get("force_service", [])  # 可选

# ---------- 2. 会话状态管理 ----------
session_intent_store = {}

def get_session_intent(session_id: str) -> Optional[Route]:
    return session_intent_store.get(session_id)

def set_session_intent(session_id: str, intent: Optional[Route]):
    if intent is None:
        session_intent_store.pop(session_id, None)
    else:
        session_intent_store[session_id] = intent

# ---------- 3. 关键词扫描（从配置文件读取） ----------
def keyword_scan(message: str) -> Optional[Route]:
    """
    纯规则匹配，不调用大模型。
    返回 'sale' 或 'service'，若都未命中返回 None
    """
    msg_lower = message.lower()
    # 【优先级最高】检查强切换词（如果配置了的话）
    for kw in FORCE_SERVICE_KEYWORDS:
        if kw in msg_lower:
            return Route.SERVICE
    # 先匹配 service（理赔/投诉通常更紧急）
    for kw in SERVICE_KEYWORDS:
        if kw in msg_lower:
            return Route.SERVICE
    for kw in SALE_KEYWORDS:
        if kw in msg_lower:
            return Route.SALE
    return None

# ---------- 4. LLM 分类兜底（模拟版） ----------
def llm_classify(message: str) -> Route:
    """
    目前是模拟逻辑。真实实现会调用 deepseek-chat。
    """
    if "险" in message or "保" in message:
        return Route.SALE
    if "赔" in message or "修" in message or "查" in message:
        return Route.SERVICE
    return Route.GENERAL

# ---------- 5. 路由决策主函数 ----------
def decide_route(session_id: str, message: str) -> Route:
    current_intent = get_session_intent(session_id)
    keyword_hit = keyword_scan(message)

    if keyword_hit is not None:
        set_session_intent(session_id, keyword_hit)
        return keyword_hit

    if current_intent is not None:
        return current_intent

    llm_result = llm_classify(message)
    if llm_result in (Route.SALE, Route.SERVICE):
        set_session_intent(session_id, llm_result)
        return llm_result
    else:
        set_session_intent(session_id, None)
        return Route.GENERAL


# ---------- 6. 自测 ----------
if __name__ == "__main__":
    from src.logger import setup_logging
    setup_logging()
    logger.info(">>> 开始路由逻辑自测（关键词从 config.yaml 加载）...")
    # 每个用例独立 session，避免状态污染
    test_cases = [
        ("你好", Route.GENERAL),
        ("我要续保", Route.SALE),
        ("我车爆胎了",Route.SERVICE),   # 现在爆胎在配置里了
        ("今天天气不错", Route.GENERAL),
        ("报价多少", Route.SALE),
        ("投诉你们", Route.SERVICE),
        ("查一下我的保单", Route.SERVICE),
        ("修车要多少钱", Route.SERVICE), # 同时命中“多少钱”和“修”，service 优先
        ("续保多少钱", Route.SALE),      # 同时命中“续保”和“多少钱”，sale
    ]
    
    all_pass = True
    for idx, (msg, expected) in enumerate(test_cases):

        sid = f"test_user_{idx}"
        result = decide_route(sid, msg)
        status = "✅" if result == expected else "❌"
        logger.info("消息: '%s' -> 路由: %s 预期: %s %s", msg, result, expected, status)
        if result != expected:
            all_pass = False
    
    if all_pass:
        logger.info("🎉 路由逻辑验证通过！配置文件生效。")
    else:
        logger.warning("⚠️ 有测试用例失败，请检查 config.yaml 中的关键词配置。")