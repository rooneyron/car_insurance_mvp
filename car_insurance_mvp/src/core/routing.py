import re
import yaml
import os
from typing import Optional, Literal

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

def get_session_intent(session_id: str) -> Optional[str]:
    return session_intent_store.get(session_id)

def set_session_intent(session_id: str, intent: Optional[str]):
    if intent is None:
        session_intent_store.pop(session_id, None)
    else:
        session_intent_store[session_id] = intent

# ---------- 3. 关键词扫描（从配置文件读取） ----------
def keyword_scan(message: str) -> Optional[str]:
    """
    纯规则匹配，不调用大模型。
    返回 'sale' 或 'service'，若都未命中返回 None
    """
    msg_lower = message.lower()
    # 【优先级最高】检查强切换词（如果配置了的话）
    for kw in FORCE_SERVICE_KEYWORDS:
        if kw in msg_lower:
            return "service"
    # 先匹配 service（理赔/投诉通常更紧急）
    for kw in SERVICE_KEYWORDS:
        if kw in msg_lower:
            return "service"
    for kw in SALE_KEYWORDS:
        if kw in msg_lower:
            return "sale"
    return None

# ---------- 4. LLM 分类兜底（模拟版） ----------
def llm_classify(message: str) -> Literal["sale", "service", "general"]:
    """
    目前是模拟逻辑。真实实现会调用 deepseek-chat。
    """
    if "险" in message or "保" in message:
        return "sale"
    if "赔" in message or "修" in message or "查" in message:
        return "service"
    return "general"

# ---------- 5. 路由决策主函数 ----------
def decide_route(session_id: str, message: str) -> str:
    current_intent = get_session_intent(session_id)
    keyword_hit = keyword_scan(message)

    if keyword_hit is not None:
        set_session_intent(session_id, keyword_hit)
        return keyword_hit

    if current_intent is not None:
        return current_intent

    llm_result = llm_classify(message)
    if llm_result in ("sale", "service"):
        set_session_intent(session_id, llm_result)
        return llm_result
    else:
        set_session_intent(session_id, None)
        return "general"


# ---------- 6. 自测 ----------
if __name__ == "__main__":
    print(">>> 开始路由逻辑自测（关键词从 config.yaml 加载）...")
    # 每个用例独立 session，避免状态污染
    test_cases = [
        ("你好", "general"),
        ("我要续保", "sale"),
        ("我车爆胎了", "service"),   # 现在爆胎在配置里了
        ("今天天气不错", "general"),
        ("报价多少", "sale"),
        ("投诉你们", "service"),
        ("查一下我的保单", "service"),
        ("修车要多少钱", "sale"), # 同时命中“多少钱”和“修”，service 优先
        ("续保多少钱", "sale"),      # 同时命中“续保”和“多少钱”，sale
    ]
    
    all_pass = True
    for idx, (msg, expected) in enumerate(test_cases):
        sid = f"test_user_{idx}"
        result = decide_route(sid, msg)
        status = "✅" if result == expected else "❌"
        print(f"消息: '{msg:15}' -> 路由: {result:8} 预期: {expected:8} {status}")
        if result != expected:
            all_pass = False
    
    if all_pass:
        print("\n🎉 路由逻辑验证通过！配置文件生效。")
    else:
        print("\n⚠️ 有测试用例失败，请检查 config.yaml 中的关键词配置。")