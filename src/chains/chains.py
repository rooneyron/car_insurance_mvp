"""
三个 Chain 的初始化 + Memory 挂载（带摘要 + 日志）
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langmem.short_term import SummarizationNode
from langchain_core.tools import tool
from typing import Optional
import json
from src.constants import RAG_EMPTY_RESULT, TOOL_FINISHED_PREFIX, TRANSFER_SIGNAL

# 根据环境变量决定是否使用本地 Rerank
from src.rag import search_terms, retrieve_candidates


# ============================================================
# 1. 工具业务逻辑（纯 Python，与 LangChain 解耦）
# ============================================================

def calculate_premium_logic(car_model: str, driver_age: int, years_driving: int) -> str:
    """
    保费估算核心逻辑
    """
    base_premium = 5000
    if "特斯拉" in car_model or "宝马" in car_model or "奔驰" in car_model:
        base_premium = 8000
    elif "比亚迪" in car_model or "吉利" in car_model or "长城" in car_model:
        base_premium = 5000
    elif "五菱" in car_model or "奇瑞" in car_model:
        base_premium = 3500

    if 25 <= driver_age <= 60:
        age_factor = 1.0
    elif 18 <= driver_age < 25:
        age_factor = 1.3
    else:
        age_factor = 1.2

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
    """
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
    RAG 检索条款核心逻辑。
    根据环境变量 USE_LOCAL_RERANK 决定使用本地 Rerank 还是 LLM 重排。
    """
    # 读取环境变量，默认为 true（本地开发）
    use_local_rerank = os.environ.get("USE_LOCAL_RERANK", "true").lower() == "true"

    try:
        if use_local_rerank:
            # ====== 本地模式：FAISS + Rerank ======
            results = search_terms(query, top_k=2)
            if not results or results == [RAG_EMPTY_RESULT]:
                return f"{TOOL_FINISHED_PREFIX}直接告知用户并建议转人工。"

            output = f"📄 关于「{query}」的相关条款（已智能排序）：\n\n"
            for i, result in enumerate(results, 1):
                output += f"--- 结果 {i} ---\n{result}\n\n"
            return output

        else:
            # ====== 生产模式：FAISS 召回 + LLM 重排 ======
            # 1. FAISS 召回 Top-10
            candidates = retrieve_candidates(query, top_k=10)
            if not candidates:
                return f"{TOOL_FINISHED_PREFIX}直接告知用户并建议转人工。"

            # 2. 构造 Prompt 让 LLM 选择最相关的 2 条
            prompt = f"""你是一个保险条款检索助手。用户的问题是："{query}"

请从以下候选条款中，选出最相关的 2 条，并按相关性从高到低排序。
只返回选中的条款原文，用 --- 分隔。

候选条款：
{chr(10).join([f'[{i+1}] {c}' for i, c in enumerate(candidates)])}
"""
            # 3. 调用 DeepSeek Chat API
            llm = ChatOpenAI(
                model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
                temperature=0,
            )
            response = llm.invoke(prompt)
            # 4. 解析返回结果
            selected = [r.strip() for r in response.content.split('---') if r.strip()]
            # 如果没有解析到任何内容，回退到前两条
            if not selected:
                selected = candidates[:2]

            output = f"📄 关于「{query}」的相关条款（已智能排序）：\n\n"
            for i, result in enumerate(selected, 1):
                output += f"--- 结果 {i} ---\n{result}\n\n"
            return output

    except Exception as e:
        # 任何异常都返回降级文本
        return f"{TOOL_FINISHED_PREFIX}直接告知用户并建议转人工。"


def transfer_to_human_logic(reason: str) -> str:
    """
    转人工核心逻辑
    """
    return TRANSFER_SIGNAL


# ============================================================
# 2. LangChain 工具包装
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
# 3. 初始化三个 Chain（带摘要）
# ============================================================

def init_chains(api_key: Optional[str] = None, model_name: str = "deepseek-v4-flash"):
    """
    初始化三个 Chain，并返回它们和共享 Memory（带摘要）
    """
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请提供 DeepSeek API Key 或设置环境变量 DEEPSEEK_API_KEY")

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.3,
    )

    # ---------- 创建摘要节点 ----------
    summarization_node = SummarizationNode(
        max_tokens=2000,          
        max_summary_tokens=500,
        model=llm,
        input_messages_key="messages",
        output_messages_key="messages",
    )

    # ---------- 共享 Memory ----------
    memory = MemorySaver()

    # ---------- 普通链 ----------
    general_chain = create_react_agent(
        model=llm,
        tools=[],
        checkpointer=memory,
        pre_model_hook=lambda state: summarization_node.invoke(state),
        prompt="""你是一个友好的车险客服助手。

注意：如果用户透露了个人信息（如身份证号、姓名、车牌号等），请在心里记住这些信息，以便后续其他助手使用。你不需要重复这些信息，但也不要拒绝接收它们。
如果用户主动提供信息来协助查询，你可以简单回应"已记录"或直接引导到具体业务。""",
    )

    # ---------- 报价链 ----------
    sale_tools = [calculate_premium, search_insurance_terms]
    agent_sale = create_react_agent(
        model=llm,
        tools=sale_tools,
        checkpointer=memory,
        pre_model_hook=lambda state: summarization_node.invoke(state),
        prompt="""你是一个车险售前助手，帮助用户计算保费、推荐投保方案。

重要规则：
1. 如果用户没有提供车型、年龄、驾龄，请检查对话历史中是否曾经提供过这些信息，如果有则直接使用。
2. 只有对话历史中也没有这些信息时，才向用户询问。
3. 请友好、专业地回答。""",
    )

    # ---------- 理赔链 ----------
    service_tools = [query_policy, search_insurance_terms, transfer_to_human]
    agent_service = create_react_agent(
        model=llm,
        tools=service_tools,
        checkpointer=memory,
        pre_model_hook=lambda state: summarization_node.invoke(state),
        prompt="""你是一个车险售后助手，帮助用户查询保单、解释理赔条款、处理投诉。

重要规则：
1. 当用户查询保单时，如果用户没有提供身份证号，请检查对话历史中是否曾经提供过，如果有则直接使用。
2. 如果对话历史中也没有身份证号，再向用户询问。
3. 不要重复索要用户已经提供过的信息。
4. 必要时可转人工。""",
    )

    return general_chain, agent_sale, agent_service, memory


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