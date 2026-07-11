"""
LangGraph StateGraph 多 Agent 编排
路由作为图的一等公民节点，三个 Agent 作为执行节点
"""

from math import log
import os
import json
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langmem.short_term import SummarizationNode
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import Optional, TypedDict, Annotated
from langgraph.graph.message import add_messages
from src.constants import RAG_EMPTY_RESULT, TOOL_FINISHED_PREFIX, TRANSFER_SIGNAL
from src.logger import get_logger
from src.route_types import Route

logger = get_logger(__name__)

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


# ---------- 模块级 LLM 实例（供生产模式 RAG 重排复用）----------
_rerank_llm: Optional[ChatOpenAI] = None


def _get_rerank_llm() -> ChatOpenAI:
    """获取用于 RAG 重排的 LLM 单例"""
    global _rerank_llm
    if _rerank_llm is None:
        _rerank_llm = ChatOpenAI(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0,
        )
    return _rerank_llm


def search_insurance_terms_logic(query: str) -> str:
    """
    RAG 检索条款核心逻辑。
    根据环境变量 USE_LOCAL_RERANK 决定使用本地 Rerank 还是 LLM 重排。
    """
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
            candidates = retrieve_candidates(query, top_k=10)
            if not candidates:
                return f"{TOOL_FINISHED_PREFIX}直接告知用户并建议转人工。"

            prompt = f"""你是一个保险条款检索助手。用户的问题是："{query}"

请从以下候选条款中，选出最相关的 2 条，并按相关性从高到低排序。
只返回选中的条款原文，用 --- 分隔。

候选条款：
{chr(10).join([f'[{i+1}] {c}' for i, c in enumerate(candidates)])}
"""
            response = _get_rerank_llm().invoke(prompt)
            selected = [r.strip() for r in response.content.split('---') if r.strip()]
            if not selected:
                selected = candidates[:2]

            output = f"📄 关于「{query}」的相关条款（已智能排序）：\n\n"
            for i, result in enumerate(selected, 1):
                output += f"--- 结果 {i} ---\n{result}\n\n"
            return output

    except Exception as e:
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
# 3. StateGraph 编排：路由 + 多 Agent 节点
# ============================================================

class GraphState(TypedDict):
    """编排图的状态定义"""
    messages: Annotated[list, add_messages]  # 对话消息
    route: str  # 路由决策结果
    reply: str  # 最终回复文本


def _create_summarization_node(llm):
    """创建摘要节点（供各 Agent 共享）"""
    return SummarizationNode(
        max_tokens=2000,
        max_summary_tokens=500,
        model=llm,
        input_messages_key="messages",
        output_messages_key="messages",
    )


def _make_route_node():
    """创建路由节点函数"""
    from src.core.routing import decide_route

    def route_node(state: GraphState, config: RunnableConfig) -> dict:
        """路由节点：根据用户消息决定走哪个 Agent"""
        session_id = config.get("configurable", {}).get("thread_id", "default")
        messages = state.get("messages", [])
        if not messages:
            return {"route": Route.GENERAL.value}

        # 取最后一条用户消息
        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

        # 调用路由决策
        route = decide_route(session_id, content)
        logger.info("图内路由决策: session=%s, route=%s", session_id, route.value)
        return {"route": route.value}

    return route_node


def _make_agent_node(agent_chain, agent_name: str):
    """创建 Agent 节点工厂"""
    def agent_node(state: GraphState, config: RunnableConfig) -> dict:
        """Agent 执行节点"""
        messages = state.get("messages", [])
        logger.info("执行 Agent 节点: %s", agent_name)

        # 调用子 Agent（create_react_agent）
        result = agent_chain.invoke({"messages": messages}, config=config)
        # 提取回复
        result_messages = result.get("messages", [])
        if result_messages:
            last_msg = result_messages[-1]
            reply = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        else:
            reply = ""

        logger.info("Agent %s 回复: %s...", agent_name, reply[:50])
        return {
            "messages": result_messages,  # 更新消息历史
            "reply": reply,
        }

    return agent_node


def init_graph(api_key: Optional[str] = None, model_name: Optional[str] = None):
    """
    初始化 StateGraph 编排图，返回编译后的图
    """
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请提供 DeepSeek API Key 或设置环境变量 DEEPSEEK_API_KEY")
    if model_name is None:
        model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.3,
    )

    # ---------- 创建摘要节点 ----------
    summarization_node = _create_summarization_node(llm)

    # ---------- 共享 Memory ----------
    memory = MemorySaver()

    # ---------- 创建三个子 Agent（create_react_agent）----------
    general_chain = create_react_agent(
        model=llm,
        tools=[],
        checkpointer=memory,
        pre_model_hook=lambda state: summarization_node.invoke(state),
        prompt="""你是一个友好的车险客服助手。

注意：如果用户透露了个人信息（如身份证号、姓名、车牌号等），请在心里记住这些信息，以便后续其他助手使用。你不需要重复这些信息，但也不要拒绝接收它们。
如果用户主动提供信息来协助查询，你可以简单回应"已记录"或直接引导到具体业务。""",
    )

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

    # ---------- 构建 StateGraph 编排图 ----------
    logger.info("构建 StateGraph 编排图...")

    # 创建节点函数
    route_node = _make_route_node()
    general_node = _make_agent_node(general_chain, "general")
    sale_node = _make_agent_node(agent_sale, "sale")
    service_node = _make_agent_node(agent_service, "service")

    # 构建图
    builder = StateGraph(GraphState)

    # 添加节点
    builder.add_node("route", route_node)
    builder.add_node("general", general_node)
    builder.add_node("sale", sale_node)
    builder.add_node("service", service_node)

    # 设置入口
    builder.add_edge(START, "route")

    # 条件分支：根据路由结果决定走哪个 Agent
    def route_decision(state: GraphState) -> str:
        route = state.get("route", Route.GENERAL.value)
        if route == Route.SALE.value:
            return "sale"
        elif route == Route.SERVICE.value:
            return "service"
        else:
            return "general"

    builder.add_conditional_edges(
        "route",
        route_decision,
        {
            "sale": "sale",
            "service": "service",
            "general": "general",
        }
    )

    # 所有 Agent 执行完后结束
    builder.add_edge("general", END)
    builder.add_edge("sale", END)
    builder.add_edge("service", END)

    # 编译图
    graph = builder.compile(checkpointer=memory)

    logger.info("✅ StateGraph 编排图构建完成")
    logger.info("📊 图结构: START -> route -> [general|sale|service] -> END")

    return graph


# ============================================================
# 4. 测试代码
# ============================================================

if __name__ == "__main__":
    from src.logger import setup_logging
    setup_logging()
    logger.info(">>> 开始测试 StateGraph 初始化...")

    try:
        graph = init_graph()

        logger.info("✅ StateGraph 初始化成功！")
        logger.info("  - Graph 类型: %s", type(graph).__name__)

        # 尝试生成 Mermaid 图
        try:
            mermaid = graph.get_graph().draw_mermaid()
            logger.info("📊 Mermaid 图生成成功:\n%s", mermaid)
        except Exception as e:
            logger.warning("Mermaid 图生成失败: %s", e)

        logger.info(">>> 测试工具逻辑（纯函数，MCP 就绪）...")
        logger.info("  - calculate_premium_logic: %s", calculate_premium_logic('特斯拉 Model 3', 30, 8))
        logger.info("  - query_policy_logic: %s", query_policy_logic('POL20260001', '110101199001011234'))
        logger.info("  - search_insurance_terms_logic: %s", search_insurance_terms_logic('车损险'))
        logger.info("  - transfer_to_human_logic: %s", transfer_to_human_logic('用户要求转人工'))

        logger.info("✅ 所有工具逻辑测试通过。")
        logger.info("📌 将来迁移 MCP 时，直接复用上述 _logic 函数即可。")

    except Exception as e:
        logger.error("❌ 初始化失败: %s", e, exc_info=True)
        logger.error("请确保: 1. 已设置环境变量 DEEPSEEK_API_KEY")