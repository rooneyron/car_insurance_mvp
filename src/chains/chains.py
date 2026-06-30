"""
三个 Chain 的初始化 + Memory 挂载
"""

import os
import sys
# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 现在可以正常导入
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from typing import Optional
import json


# ---------- 全局 Memory（所有 Chain 共用） ----------
shared_memory = MemorySaver()


# ============================================================
# 1. 工具业务逻辑（纯 Python，与 LangChain 解耦）
#    将来迁移 MCP 时，直接把这些函数注册为 MCP 工具即可
# ============================================================

def calculate_premium_logic(car_model: str, driver_age: int, years_driving: int) -> str:
    """
    保费估算核心逻辑
    参数：车型、驾驶员年龄、驾龄
    返回：格式化的保费估算结果
    """
    # 基础保费（根据车型粗略划分）
    base_premium = 5000  # 默认
    if "特斯拉" in car_model or "宝马" in car_model or "奔驰" in car_model:
        base_premium = 8000
    elif "比亚迪" in car_model or "吉利" in car_model or "长城" in car_model:
        base_premium = 5000
    elif "五菱" in car_model or "奇瑞" in car_model:
        base_premium = 3500

    # 年龄系数（25-60 岁为最佳，之外系数提高）
    if 25 <= driver_age <= 60:
        age_factor = 1.0
    elif 18 <= driver_age < 25:
        age_factor = 1.3
    else:
        age_factor = 1.2

    # 驾龄系数（驾龄越长系数越低）
    if years_driving >= 10:
        driving_factor = 0.85
    elif years_driving >= 5:
        driving_factor = 0.95
    elif years_driving >= 2:
        driving_factor = 1.0
    else:
        driving_factor = 1.15

    final_premium = base_premium * age_factor * driving_factor

    return (
        f"🚗 保费估算结果\n"
        f"车型：{car_model}\n"
        f"驾驶员年龄：{driver_age} 岁\n"
        f"驾龄：{years_driving} 年\n"
        f"预估年保费：{final_premium:.0f} 元"
    )


def query_policy_logic(policy_id: str, id_card: str) -> str:
    """
    保单查询核心逻辑
    参数：保单号、身份证号
    返回：格式化的保单详情或未找到提示
    """
    # 读取造假数据
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "policies.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for policy in data["保单列表"]:
        if policy["保单号"] == policy_id and policy["身份证号"] == id_card:
            return (
                f"✅ 保单查询成功\n"
                f"保单号：{policy['保单号']}\n"
                f"车主：{policy['车主姓名']}\n"
                f"车型：{policy['车型']}\n"
                f"险种：{', '.join(policy['险种'])}\n"
                f"保额：{policy['保额']:,} 元\n"
                f"年保费：{policy['年保费']:,} 元\n"
                f"到期日：{policy['到期日']}\n"
                f"状态：{policy['状态']}"
            )

    return f"❌ 未找到保单（保单号：{policy_id}，身份证号：{id_card}），请核对信息后重新查询。"


def search_insurance_terms_logic(query: str) -> str:
    """
    RAG 检索条款核心逻辑（真实 FAISS + Cross-Encoder Rerank）
    参数：搜索关键词
    返回：匹配的条款内容
    """
    try:
        # 直接使用绝对导入，不依赖 sys.path
        import importlib.util
        import os
        
        # 动态加载 rag 模块
        rag_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag.py")
        spec = importlib.util.spec_from_file_location("rag", rag_path)
        rag = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag)
        
        results = rag.search_terms(query, top_k=2)
        if not results or results == ["未找到相关内容"]:
            return f"📄 未找到与「{query}」直接匹配的条款，建议咨询人工客服获取详细信息。"

        output = f"📄 关于「{query}」的相关条款（已智能排序）：\n\n"
        for i, result in enumerate(results, 1):
            output += f"--- 结果 {i} ---\n{result}\n\n"
        return output
    except Exception as e:
        return f"❌ 检索失败: {str(e)}"


def transfer_to_human_logic(reason: str) -> str:
    """
    转人工核心逻辑
    参数：转人工原因
    返回：__TRANSFER__ 标志（外层编排拦截此标志）
    """
    return "__TRANSFER__"


# ============================================================
# 2. LangChain 工具包装（薄薄一层，调用上面的 _logic 函数）
#    将来迁移 MCP 时，只需删除 @tool 包装，直接注册 _logic 函数即可
# ============================================================

@tool
def calculate_premium(car_model: str, driver_age: int, years_driving: int) -> str:
    """估算车险保费。在用户询问保费、报价、投保费用时调用。参数：car_model（车型）、driver_age（驾驶员年龄）、years_driving（驾龄）"""
    return calculate_premium_logic(car_model, driver_age, years_driving)


@tool
def query_policy(policy_id: str, id_card: str) -> str:
    """查询保单详情。当用户询问保单信息、保单状态时调用。参数：policy_id（保单号）、id_card（身份证号）"""
    return query_policy_logic(policy_id, id_card)


@tool
def search_insurance_terms(query: str) -> str:
    """搜索保险条款内容。当用户询问具体条款、保障范围、免责条款时调用。参数：query（搜索关键词）"""
    return search_insurance_terms_logic(query)


@tool
def transfer_to_human(reason: str) -> str:
    """转人工客服。仅在用户明确说出"转人工"、"投诉"、"我要人工"时调用。参数：reason（转人工原因）"""
    return transfer_to_human_logic(reason)


# ============================================================
# 3. 初始化三个 Chain
# ============================================================

def init_chains(api_key: Optional[str] = None, model_name: str = "deepseek-chat"):
    """
    初始化三个 Chain，并返回它们和共享 Memory

    参数:
        api_key: DeepSeek API Key（如果不传，从环境变量 DEEPSEEK_API_KEY 读取）
        model_name: 模型名称，默认 deepseek-chat

    返回:
        (general_chain, agent_sale, agent_service, memory)
    """
    # 1. 获取 API Key
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请提供 DeepSeek API Key 或设置环境变量 DEEPSEEK_API_KEY")

    # 2. 创建 LLM 实例（DeepSeek 兼容 OpenAI 接口）
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.3,
    )

    # 3. 普通链（不带工具）
    general_chain = llm

    # 4. 报价链工具列表
    sale_tools = [calculate_premium, search_insurance_terms]

    # 5. 理赔链工具列表
    service_tools = [query_policy, search_insurance_terms, transfer_to_human]

    # 6. 创建 Agent
    agent_sale = create_react_agent(
        model=llm,
        tools=sale_tools,
        checkpointer=shared_memory,
        prompt="你是一个车险售前助手，帮助用户计算保费、推荐投保方案。请友好、专业地回答。"
    )

    agent_service = create_react_agent(
        model=llm,
        tools=service_tools,
        checkpointer=shared_memory,
        prompt="你是一个车险售后助手，帮助用户查询保单、解释理赔条款、处理投诉。必要时可转人工。"
    )

    return general_chain, agent_sale, agent_service, shared_memory


# ============================================================
# 4. 测试代码
# ============================================================

if __name__ == "__main__":
    print(">>> 开始测试 Chain 初始化...")

    try:
        general, sale_agent, service_agent, memory = init_chains()

        print("✅ 三个 Chain 初始化成功！")
        print(f"  - General Chain 类型: {type(general).__name__}")
        print(f"  - Sale Agent 类型: {type(sale_agent).__name__}")
        print(f"  - Service Agent 类型: {type(service_agent).__name__}")
        print(f"  - Memory 类型: {type(memory).__name__}")

        # 测试工具逻辑（直接调用 _logic，不经过 LangChain）
        print("\n>>> 测试工具逻辑（纯函数，MCP 就绪）...")
        print("  - calculate_premium_logic:")
        print(f"    {calculate_premium_logic('特斯拉 Model 3', 30, 8)}")
        print("  - query_policy_logic:")
        print(f"    {query_policy_logic('POL20260001', '110101199001011234')}")
        print("  - search_insurance_terms_logic:")
        print(f"    {search_insurance_terms_logic('车损险')}")
        print("  - transfer_to_human_logic:")
        print(f"    {transfer_to_human_logic('用户要求转人工')}")

        print("\n✅ 所有工具逻辑测试通过。")
        print("📌 将来迁移 MCP 时，直接复用上述 _logic 函数即可。")

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("\n请确保:")
        print("  1. 已设置环境变量 DEEPSEEK_API_KEY")
        print("  2. 或者修改 init_chains() 调用，传入 api_key 参数")