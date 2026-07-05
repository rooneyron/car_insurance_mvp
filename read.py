# 导入正则表达式模块，用于字符串匹配
import re
# 导入 yaml 模块，用于解析 YAML 配置文件
import yaml
# 导入 os 模块，用于文件和路径操作
import os
# 导入类型提示：Optional（可选值）和 Literal（字面量类型）
from typing import Optional, Literal


# ---------- 1. 加载配置文件（支持热更新） ----------
# 定义配置文件的路径（相对于项目根目录）
CONFIG_PATH = "config/config.yaml"

def load_keywords():
    """从 config.yaml 读取关键词列表"""
    # 检查配置文件是否存在，如果不存在则抛出异常
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"配置文件 {CONFIG_PATH} 不存在，请先创建。")
    # 打开并读取 YAML 文件
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)  # 解析 YAML 为 Python 字典
    # 返回 keywords 部分的内容
    return config["keywords"]

# 调用 load_keywords() 加载关键词配置
keywords = load_keywords()
# 从配置中提取 sale 关键词列表
SALE_KEYWORDS = keywords.get("sale", [])
# 从配置中提取 service 关键词列表
SERVICE_KEYWORDS = keywords.get("service", [])
# 从配置中提取强制 service 关键词列表（可选）
FORCE_SERVICE_KEYWORDS = keywords.get("force_service", [])  # 可选


# ---------- 2. 会话状态管理 ----------
# 用字典存储每个会话的当前意图，key 是 session_id，value 是 "sale" 或 "service"
session_intent_store = {}

def get_session_intent(session_id: str) -> Optional[str]:
    """获取指定会话的当前意图，如果不存在则返回 None"""
    return session_intent_store.get(session_id)

def set_session_intent(session_id: str, intent: Optional[str]):
    """设置指定会话的意图"""
    if intent is None:
        # 如果 intent 是 None，从字典中删除该会话
        session_intent_store.pop(session_id, None)
    else:
        # 否则设置该会话的意图
        session_intent_store[session_id] = intent


# ---------- 3. 关键词扫描（从配置文件读取） ----------
def keyword_scan(message: str) -> Optional[str]:
    """
    纯规则匹配，不调用大模型。
    返回 'sale' 或 'service'，若都未命中返回 None
    """
    # 将消息转换为小写，便于不区分大小写的匹配
    msg_lower = message.lower()
    
    # 【优先级最高】检查强切换词（如果配置了的话）
    for kw in FORCE_SERVICE_KEYWORDS:
        if kw in msg_lower:
            # 只要匹配到任何一个强切换词，立即返回 "service"
            return "service"
    
    # 先匹配 service（理赔/投诉通常更紧急）
    for kw in SERVICE_KEYWORDS:
        if kw in msg_lower:
            return "service"
    
    # 再匹配 sale
    for kw in SALE_KEYWORDS:
        if kw in msg_lower:
            return "sale"
    
    # 都没有命中
    return None


# ---------- 4. LLM 分类兜底（模拟版） ----------
def llm_classify(message: str) -> Literal["sale", "service", "general"]:
    """
    目前是模拟逻辑。真实实现会调用 deepseek-chat。
    注意：这是"兜底逻辑"，只在关键词未命中且无会话意图时调用。
    """
    # 如果消息包含"险"或"保"，归类为 sale
    if "险" in message or "保" in message:
        return "sale"
    # 如果消息包含"赔"或"修"或"查"，归类为 service
    if "赔" in message or "修" in message or "查" in message:
        return "service"
    # 否则归类为 general（闲聊）
    return "general"


# ---------- 5. 路由决策主函数 ----------
def decide_route(session_id: str, message: str) -> str:
    """
    路由决策的核心逻辑：
    1. 关键词优先（如果有命中，立即切换）
    2. 如果命中关键词，更新会话意图并返回
    3. 如果未命中关键词，但会话已有意图，复用该意图
    4. 如果首次对话且无关键词，走 LLM 兜底分类
    """
    # 获取当前会话的已有意图
    current_intent = get_session_intent(session_id)
    # 执行关键词扫描
    keyword_hit = keyword_scan(message)

    # 【规则1】关键词优先：只要命中，立即切换/设定意图
    if keyword_hit is not None:
        # 更新会话意图为命中的值
        set_session_intent(session_id, keyword_hit)
        # 返回命中的路由
        return keyword_hit

    # 【规则2】关键词未命中，但当前已有业务意图 -> 复用（不切到 general）
    if current_intent is not None:
        # 直接返回当前意图，不更新
        return current_intent

    # 【规则3】首次对话，且无关键词 -> 调用 LLM 兜底
    llm_result = llm_classify(message)
    if llm_result in ("sale", "service"):
        # 如果 LLM 判为业务，则更新意图（防止下次丢失）
        set_session_intent(session_id, llm_result)
        return llm_result
    else:
        # general 保持 None，不存储
        set_session_intent(session_id, None)
        return "general"


# ---------- 6. 自测 ----------
if __name__ == "__main__":
    # 这个块只在直接运行 routing.py 时执行（不会在 import 时执行）
    print(">>> 开始路由逻辑自测（关键词从 config.yaml 加载）...")
    
    # 定义测试用例列表：(消息, 期望路由)
    test_cases = [
        ("你好", "general"),
        ("我要续保", "sale"),
        ("我车爆胎了", "service"),   # 现在爆胎在配置里了
        ("今天天气不错", "general"),
        ("报价多少", "sale"),
        ("投诉你们", "service"),
        ("查一下我的保单", "service"),
        ("修车要多少钱", "sale"),    # 同时命中"多少钱"和"修"，service 优先
        ("续保多少钱", "sale"),      # 同时命中"续保"和"多少钱"，sale
    ]
    
    all_pass = True
    # 遍历测试用例
    for idx, (msg, expected) in enumerate(test_cases):
        # 每个用例使用独立的 session_id，避免状态污染
        sid = f"test_user_{idx}"
        # 执行路由决策
        result = decide_route(sid, msg)
        # 判断是否与预期一致
        status = "✅" if result == expected else "❌"
        # 打印测试结果
        print(f"消息: '{msg:15}' -> 路由: {result:8} 预期: {expected:8} {status}")
        if result != expected:
            all_pass = False
    
    # 输出最终结果
    if all_pass:
        print("\n🎉 路由逻辑验证通过！配置文件生效。")
    else:
        print("\n⚠️ 有测试用例失败，请检查 config.yaml 中的关键词配置。")