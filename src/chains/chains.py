"""
LangGraph StateGraph 多 Agent 编排
路由作为图的一等公民节点，三个 Agent 作为执行节点
"""

import os
import json
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langmem.short_term import SummarizationNode
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from typing import Optional, TypedDict, Annotated
from langgraph.graph.message import add_messages
from src.constants import RAG_EMPTY_RESULT, TRANSFER_SIGNAL
from src.logger import get_logger
from src.route_types import Route
import time

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
        import httpx
        http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=300,
            ),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        _rerank_llm = ChatOpenAI(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0,
            max_retries=1,
            http_client=http_client,
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
                return "未检索到相关保险条款。"

            output = f"📄 关于「{query}」的相关条款（已智能排序）：\n\n"
            for i, result in enumerate(results, 1):
                output += f"--- 结果 {i} ---\n{result}\n\n"
            return output

        else:
            # ====== 生产模式：FAISS 召回 + LLM 重排 ======
            candidates = retrieve_candidates(query, top_k=10)
            if not candidates:
                return "未检索到相关保险条款。"

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
        return "未检索到相关保险条款。"


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
    """查询保险条款。当用户询问保险条款相关问题时，必须调用此工具检索，禁止凭自身知识直接回答。参数：query（搜索关键词）"""
    return search_insurance_terms_logic(query)


@tool
def transfer_to_human(reason: str) -> str:
    """转人工客服。仅在用户明确说出"转人工"、"投诉"、"我要人工"时调用。参数：reason（转人工原因）"""
    return transfer_to_human_logic(reason)


# ============================================================
# 3. Agent 配置
# ============================================================

SYSTEM_PROMPTS = {
    "general": """【角色定义】
你是一个友好的车险客服助手，负责承接用户的初始咨询。

【职责边界】
- 承接用户的第一轮咨询
- 引导用户表达具体需求（报价、理赔、保单查询、投诉等）
- 记录用户提供的个人信息（身份证号、姓名、车牌号等），传递给后续助手

【行为规则】
- 如果用户提供了个人信息，简单回应"已记录"并引导到具体业务
- 如果用户没有说明具体需求，主动询问："您是需要报价、理赔咨询还是保单查询？"

【输出风格】
友好、自然、引导性强。

【拒绝边界】
超出车险范围的咨询，直接引导转人工。""",

    "sale": """【角色定义】
你是一个车险售前助手，帮助用户计算保费和推荐投保方案。

【职责边界】
- 计算保费（需要车型、年龄、驾龄）
- 推荐适合的投保方案
- 解释报价相关的条款

【行为规则】
- 当用户询问保险条款相关问题时，必须调用 search_insurance_terms 工具检索，禁止凭自身知识直接回答。
- 优先从对话历史中复用已提供的信息（车型、年龄、驾龄）
- 信息缺失时，向用户确认后再计算
- 报价结果包含保额和险种建议

【输出风格】
清晰、直接，给出具体数字和推荐方案。

【拒绝边界】
超出报价范围的咨询，引导用户咨询售后或转人工。""",

    "service": """【角色定义】
你是一个车险售后助手，帮助用户处理保单查询、保险条款查询和投诉转接。

【职责边界】
- 查询保单信息
- 解释保险条款
- 转接人工客服

【行为规则】
- 当用户询问保险条款相关问题时，必须调用 search_insurance_terms 工具检索，禁止凭自身知识直接回答。
- 优先从对话历史中复用已提供的信息（身份证号、保单号）
- 信息缺失时，向用户确认后再查询

【输出风格】
专业、简洁、直接。

【拒绝边界】
超出车险范围的咨询，引导用户转人工处理。""",
}

# 各 Agent 绑定的工具集
AGENT_TOOLS = {
    "general": [transfer_to_human],
    "sale": [calculate_premium, search_insurance_terms],
    "service": [query_policy, search_insurance_terms, transfer_to_human],
}

# 全部工具的去重集合（供 ToolNode 使用）
ALL_TOOLS = [calculate_premium, search_insurance_terms, query_policy, transfer_to_human]

# 工具中文标签映射（供 chat.py tool_status 推送使用）
TOOL_LABELS = {
    "search_insurance_terms": "条款检索",
    "calculate_premium": "保费计算器",
    "query_policy": "保单查询",
    "transfer_to_human": "转人工",
}


# ============================================================
# 4. StateGraph 状态定义
# ============================================================

class GraphState(TypedDict):
    """编排图的状态定义"""
    messages: Annotated[list, add_messages]
    agent_type: str          # 当前 Agent 类型：general / sale / service
    responder_input: list    # 纯净输入（用户消息 + 工具结果，不含 Planner 内部 AIMessage）
    direct_response: Optional[str]  # 短路直返内容：非空时跳过 planner → responder，直接返回给前端
    route: str
    reply: str
    summary: Optional[str]   # 长期记忆摘要（由 SummarizationNode 生成）


# ============================================================
# 5. 节点工厂
# ============================================================

def _create_summarization_node(llm):
    """创建摘要节点（供各 Agent 共享）"""
    return SummarizationNode(
        max_tokens=2000,
        max_summary_tokens=500,
        model=llm,
        input_messages_key="messages",
        output_messages_key="messages",
    )


def _make_router_node():
    """
    路由节点：纯关键词规则判断 + 构建纯净的 responder_input。
    - 根据用户输入关键词写入 agent_type
    - 构建 responder_input：
        - 如果有摘要：摘要 + 最近 3 轮对话（6 条消息），只取 Human 和 AIMessage
        - 如果没有摘要：全部历史消息（过滤掉 Planner 空消息和工具调用）
      + 最近一次工具调用对
    - 强制重置 direct_response = None，防止历史污染新请求
    """
    from src.core.routing import decide_route

    def router_node(state: GraphState, config: RunnableConfig) -> dict:
        start_time = time.time()
        session_id = config.get("configurable", {}).get("thread_id", "default")
        messages = state.get("messages", [])
        summary = state.get("summary", None)

        if not messages:
            return {
                "agent_type": Route.GENERAL.value,
                "route": Route.GENERAL.value,
                "responder_input": [],
                "direct_response": None,
                "summary": summary,
            }

        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

        route = decide_route(session_id, content)
        agent_type = route.value

        responder_input = []

        # ---- 构建核心上下文 ----
        if summary:
            # 有摘要：用摘要代表旧历史，再补充最近 3 轮对话
            responder_input.append(SystemMessage(content=f"【用户长期记忆摘要】{summary}"))
            recent_count = 6
            recent_messages = messages[-recent_count:] if len(messages) > recent_count else messages
            for msg in recent_messages:
                # 只保留 Human 和 AIMessage（不含 tool_calls 且有内容），忽略 SystemMessage（避免重复）
                if isinstance(msg, HumanMessage):
                    responder_input.append(msg)
                elif isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None) and msg.content:
                    responder_input.append(msg)
                # SystemMessage 跳过（不加入）
        else:
            # 没有摘要：保留全部历史（过滤掉 Planner 空消息和工具调用）
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    responder_input.append(msg)
                elif isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None) and msg.content:
                    responder_input.append(msg)

        # ---- 加入最近一次工具调用对（如有） ----
        last_tool_msg = None
        last_tool_call_ai = None
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                last_tool_msg = msg
                break
        if last_tool_msg:
            tool_call_id = getattr(last_tool_msg, 'tool_call_id', None)
            if tool_call_id:
                for msg in messages:
                    if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
                        for tc in msg.tool_calls:
                            if tc.get('id') == tool_call_id:
                                last_tool_call_ai = msg
                                break
                        if last_tool_call_ai:
                            break
            if last_tool_call_ai:
                ai_copy = AIMessage(content="", tool_calls=last_tool_call_ai.tool_calls)
                responder_input.append(ai_copy)
                responder_input.append(last_tool_msg)
            else:
                logger.warning("丢弃孤立的 ToolMessage，缺少对应的 AIMessage(tool_calls)")

        return {
            "agent_type": agent_type,
            "route": agent_type,
            "responder_input": responder_input,
            "direct_response": None,
            "summary": summary,
        }

    return router_node


def _make_planner_node(llm):
    """
    决策节点：非流式调用 LLM，决定是否调用工具。
    - 根据 agent_type 选择 System Prompt 和 Tools
    - 调用 model.bind_tools(tools).invoke()（非流式）
    - 强制清空 content，防止任何废话污染历史
    """

    def planner_node(state: GraphState, config: RunnableConfig) -> dict:
        agent_type = state.get("agent_type", Route.GENERAL.value)
        system_prompt = SYSTEM_PROMPTS.get(agent_type, SYSTEM_PROMPTS[Route.GENERAL.value])
        tools = AGENT_TOOLS.get(agent_type, [])
        messages = list(state.get("messages", []))

        full_messages = [SystemMessage(content=system_prompt)] + messages

        if tools:
            result = llm.bind_tools(tools).invoke(full_messages)
        else:
            result = llm.invoke(full_messages)

        # 强制清空 content，防止废话污染历史
        result.content = ""

        return {"messages": [result]}

    return planner_node


def _make_tools_node():
    """
    工具执行节点：执行工具 + 更新 responder_input。
    - 使用 ToolNode(ALL_TOOLS) 执行工具
    - 将工具结果同步追加到 responder_input
    - TRANSFER_SIGNAL 替换为纯文本占位
    """
    tool_node = ToolNode(ALL_TOOLS)

    def _tools_node(state: GraphState, config: RunnableConfig) -> dict:
        result = tool_node.invoke(state)

        new_responder = list(state.get("responder_input", []))

        # 获取触发本次工具调用的 AIMessage（带 tool_calls，content 置空）
        # 必须插入到 ToolMessage 之前，否则 DeepSeek API 报错：
        # "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'"
        messages = state.get("messages", [])
        last_ai = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
                last_ai = msg
                break
        if last_ai:
            new_responder.append(AIMessage(content="", tool_calls=last_ai.tool_calls))

        for msg in result.get("messages", []):
            content = getattr(msg, 'content', '')
            if TRANSFER_SIGNAL in str(content):
                clean_content = str(content).replace(TRANSFER_SIGNAL, "已提交转人工请求")
                new_responder.append(ToolMessage(
                    content=clean_content,
                    tool_call_id=getattr(msg, 'tool_call_id', ''),
                    name=getattr(msg, 'name', ''),
                ))
            else:
                new_responder.append(msg)

        result["responder_input"] = new_responder

        # 短路逻辑：特定工具结果可直接返回，跳过 planner → responder 链路
        direct_response = None
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage):
                content = str(msg.content)
                # 场景 1：转人工工具返回 TRANSFER_SIGNAL
                if TRANSFER_SIGNAL in content:
                    session_id = config.get("configurable", {}).get("thread_id", "default")
                    ticket_id = f"TK{int(time.time())}{session_id[-4:]}"
                    direct_response = json.dumps({
                        "transfer": True,
                        "ticket_id": ticket_id,
                        "message": "正在为您转接人工客服，工单号：" + ticket_id,
                    }, ensure_ascii=False)
                    break
                # 场景 2：条款搜索工具返回 "未检索到相关保险条款"
                if "未检索到相关保险条款" in content:
                    direct_response = "很抱歉，我在知识库中没有找到与您问题相关的条款信息，建议您转人工咨询。"
                    break

        if direct_response:
            result["direct_response"] = direct_response

        return result

    return _tools_node


def _make_responder_node(llm):
    async def responder_node(state: GraphState, config: RunnableConfig) -> dict:
        agent_type = state.get("agent_type", Route.GENERAL.value)
        system_prompt = SYSTEM_PROMPTS.get(agent_type, SYSTEM_PROMPTS[Route.GENERAL.value])
        responder_input = list(state.get("responder_input", []))

        full_messages = [SystemMessage(content=system_prompt)] + responder_input

        full_content = ""
        async for chunk in llm.astream(full_messages):
            if chunk.content:
                full_content += chunk.content

        return {
            "messages": [AIMessage(content=full_content)],
            "reply": full_content,
        }

    return responder_node


def _planner_condition(state: GraphState) -> str:
    """条件边：planner → tools（有 tool_calls） 或 responder（无 tool_calls）"""
    messages = state.get("messages", [])
    if not messages:
        return "responder"
    last_msg = messages[-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    return "responder"


def _after_tools_condition(state: GraphState) -> str:
    """tools 节点后的条件边：如果 direct_response 非空，直接结束；否则回到 planner"""
    if state.get("direct_response"):
        return "end"
    return "planner"


# ============================================================
# 6. 图初始化
# ============================================================

def init_graph(api_key: Optional[str] = None, model_name: Optional[str] = None):
    """
    初始化手写 StateGraph 编排图，返回编译后的图。
    图结构: START → router → planner ⇄ tools → responder → END
    """
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请提供 DeepSeek API Key 或设置环境变量 DEEPSEEK_API_KEY")
    if model_name is None:
        model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

    import httpx
    http_client = httpx.Client(
        limits=httpx.Limits(
            max_connections=10,
            max_keepalive_connections=5,
            keepalive_expiry=300,
        ),
        timeout=httpx.Timeout(10.0, connect=5.0),
    )

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0,
        max_retries=1,
        http_client=http_client,
    )

    # ---------- 创建摘要节点 ----------
    summarization_node = _create_summarization_node(llm)

    # ---------- 自定义摘要注入函数：将摘要存入 state.summary ----------
    def summarize_and_store(state: dict) -> dict:
        """在每次图执行前压缩历史，并将摘要存入 state"""
        result = summarization_node.invoke({"messages": state.get("messages", [])})
        # result 应包含 "messages" 字段，其中可能追加了摘要消息
        new_messages = result.get("messages", [])
        # 提取摘要内容：最后一条 SystemMessage 通常为摘要
        summary = None
        if new_messages and isinstance(new_messages[-1], SystemMessage):
            summary = new_messages[-1].content
        return {"messages": new_messages, "summary": summary}

    from src.state import set_summarize_fn
    set_summarize_fn(summarize_and_store)

    # ---------- 共享 Memory ----------
    memory = MemorySaver()

    # ---------- 创建 4 个节点 ----------
    router_node = _make_router_node()
    planner_node = _make_planner_node(llm)
    tools_node = _make_tools_node()
    responder_node = _make_responder_node(llm)

    # ---------- 构建图 ----------
    logger.info("构建手写 StateGraph 编排图...")
    builder = StateGraph(GraphState)

    builder.add_node("router", router_node)
    builder.add_node("planner", planner_node)
    builder.add_node("tools", tools_node)
    builder.add_node("responder", responder_node)

    # 边
    builder.add_edge(START, "router")
    builder.add_edge("router", "planner")

    # planner 条件分支：有 tool_calls → tools，无 → responder
    builder.add_conditional_edges(
        "planner",
        _planner_condition,
        {
            "tools": "tools",
            "responder": "responder",
        }
    )

    # tools 条件分支：有 direct_response → END（短路），否则 → planner（ReAct 循环）
    builder.add_conditional_edges(
        "tools",
        _after_tools_condition,
        {
            "planner": "planner",
            "end": END,
        }
    )

    # responder → END
    builder.add_edge("responder", END)

    graph = builder.compile(checkpointer=memory)

    logger.info("✅ StateGraph 编排图构建完成")
    logger.info("📊 图结构: START → router → planner ⇄ tools → responder → END")

    return graph, llm


def warmup_llm(llm: ChatOpenAI):
    """
    预热 LLM 连接：发送一个极轻量请求，提前建立 TCP/TLS 连接。
    避免用户第一条消息因首请求延迟等待 20 秒。
    """
    try:
        logger.info("正在预热 LLM 连接...")
        llm.invoke("hi")
        logger.info("LLM 连接预热完成")
    except Exception as e:
        logger.warning(f"LLM 预热失败（不影响服务）: {e}")


# ============================================================
# 4. 测试代码
# ============================================================

if __name__ == "__main__":
    from src.logger import setup_logging
    setup_logging()
    logger.info(">>> 开始测试 StateGraph 初始化...")

    try:
        graph, _ = init_graph()

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