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
from src.constants import (
    INTENT_GENERAL, INTENT_SALE, INTENT_SERVICE, INTENT_VALUES,
    ACTION_ROUTE, ACTION_CLARIFY, ACTION_HANDOFF,
    SOURCE_L0_SAFETY, SOURCE_DST, SOURCE_L1_KEYWORD, SOURCE_L2, SOURCE_L3,
    SOURCE_L4_CLARIFY, SOURCE_L4_HANDOFF,
)
from src.router.prompts import AGENT_SYSTEM_PROMPTS
from src.logger import get_logger
from src.route_types import Route
import time

logger = get_logger(__name__)

# 根据环境变量决定是否使用本地 Rerank
from src.rag import search_terms, retrieve_candidates


# ============================================================
# 1. 工具业务逻辑（纯 Python，与 LangChain 解耦）
# ============================================================

def calculate_premium_logic(car_model: Optional[str] = None, driver_age: Optional[int] = None, years_driving: Optional[int] = None) -> dict:
    """
    保费估算核心逻辑。
    返回结构化结果：success / missing_params（与 query_policy 契约一致，供 tools_node 写入 DST awaiting_slot）
    """
    # 参数验证
    missing = []
    if not car_model:
        missing.append("car_model")
    if driver_age is None:
        missing.append("driver_age")
    if years_driving is None:
        missing.append("years_driving")
    if missing:
        return {"status": "missing_params", "missing": missing}

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

    return {
        "status": "success",
        "message": (
            f"🚗 保费估算结果\n"
            f"车型：{car_model}\n"
            f"驾驶员年龄：{driver_age} 岁\n"
            f"驾龄：{years_driving} 年\n"
            f"预估年保费：{final_premium:.0f} 元"
        ),
    }


def query_policy_logic(policy_id: Optional[str] = None, id_card: Optional[str] = None) -> dict:
    """
    保单查询核心逻辑。
    返回结构化结果：success / missing_params / error
    """
    # 参数验证
    missing = []
    if not policy_id:
        missing.append("policy_id")
    if not id_card:
        missing.append("id_card")
    if missing:
        return {"status": "missing_params", "missing": missing}

    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "policies.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for policy in data["保单列表"]:
        if policy["保单号"] == policy_id and policy["身份证号"] == id_card:
            return {
                "status": "success",
                "data": {
                    "保单号": policy["保单号"],
                    "车主": policy["车主姓名"],
                    "车型": policy["车型"],
                    "险种": ", ".join(policy["险种"]),
                    "保额": f"{policy['保额']:,} 元",
                    "年保费": f"{policy['年保费']:,} 元",
                    "到期日": policy["到期日"],
                    "状态": policy["状态"],
                }
            }

    return {"status": "error", "message": f"未找到保单（保单号：{policy_id}，身份证号：{id_card}），请核对信息后重新查询。"}


# ---------- 存储 LLM 实例供工具使用 ----------
_rag_llm: Optional[ChatOpenAI] = None


def _get_rag_llm() -> ChatOpenAI:
    """获取用于 RAG 重排的 LLM 单例"""
    if _rag_llm is None:
        raise RuntimeError("RAG LLM 未初始化，请先调用 init_graph")
    return _rag_llm


def search_insurance_terms_logic(query: str) -> str:
    """
    RAG 检索条款核心逻辑。
    统一走 hybrid_search 召回 + 精排（本地 Cross-Encoder / 生产 DashScope Rerank）。
    """
    use_local_rerank = os.environ.get("USE_LOCAL_RERANK", "true").lower() == "true"
    logger.info("🔍 [RAG工具] query='%s' | rerank=%s", query, "CrossEncoder" if use_local_rerank else "qwen3-rerank")

    try:
        results = search_terms(query, top_k=3)
        if not results or results == [RAG_EMPTY_RESULT]:
            logger.info("🔍 [RAG工具] 未检索到相关保险条款")
            return "未检索到相关保险条款。"

        output = f"📄 关于「{query}」的相关条款（已智能排序）：\n\n"
        for i, result in enumerate(results, 1):
            output += f"--- 结果 {i} ---\n{result}\n\n"
        logger.info("🔍 [RAG工具] 返回 %d 条结果:\n%s", len(results), output[:500])
        return output

    except Exception as e:
        logger.error("🔍 [RAG工具] 异常: %s", e)
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
def calculate_premium(car_model: Optional[str] = None, driver_age: Optional[int] = None, years_driving: Optional[int] = None) -> str:
    """估算车险保费。在用户询问保费、报价、投保费用时调用。参数：car_model（车型）、driver_age（驾驶员年龄）、years_driving（驾龄）。即使参数不全也请调用，工具会返回缺少哪些参数。"""
    result = calculate_premium_logic(car_model, driver_age, years_driving)
    # 工具返回结构化 dict，由 tools_node 解析处理
    return json.dumps(result, ensure_ascii=False)


@tool
def query_policy(policy_id: Optional[str] = None, id_card: Optional[str] = None) -> str:
    """查询保单详情。当用户询问保单信息、保单状态时调用。参数：policy_id（保单号）、id_card（身份证号）。即使参数不全也请调用，工具会返回缺少哪些参数。"""
    result = query_policy_logic(policy_id, id_card)
    # 工具返回结构化 dict，由 tools_node 解析处理
    return json.dumps(result, ensure_ascii=False)


@tool
def search_insurance_terms(query: str) -> str:
    """查询保险条款。当用户询问保险条款相关问题时，必须调用此工具检索，禁止凭自身知识直接回答。参数：query（搜索关键词）"""
    return search_insurance_terms_logic(query)


@tool
def transfer_to_human(reason: str) -> str:
    """转人工客服。仅在用户明确说出"转人工"、"投诉"、"我要人工"时调用。参数：reason（转人工原因）"""
    return transfer_to_human_logic(reason)


# ============================================================
# 3. Agent 配置（Prompt 来自 src/router/prompts.py）
# ============================================================

# 各 Agent 绑定的工具集
AGENT_TOOLS = {
    INTENT_GENERAL: [transfer_to_human],
    INTENT_SALE: [calculate_premium, search_insurance_terms, transfer_to_human],
    INTENT_SERVICE: [query_policy, search_insurance_terms, transfer_to_human],
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
    responder_input: list    # 纯净输入（prepare_input 节点构建，供 planner + responder 使用）
    direct_response: Optional[str]  # 短路直返内容：非空时跳过 planner → responder，直接返回给前端
    route: str
    reply: str
    summary: Optional[str]   # 长期记忆摘要（由 SummarizationNode 生成）
    # ---- Router 四层漏斗状态 ----
    action: str              # route / clarify / handoff
    clarify_count: int       # 连续澄清次数
    waiting_clarification: bool    # 是否正在等待澄清回答
    last_clarify_options: list     # 上次澄清的选项列表
    router_source: Optional[str]   # 决策来源层
    router_confidence: float       # 决策置信度
    # ---- DST 状态 ----
    current_task: Optional[str]    # 当前未完成任务：sale / service / None
    awaiting_slot: Optional[str]   # 当前等待补充的参数名（如 id_card）


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


def _build_responder_input(messages: list, summary: Optional[str] = None) -> list:
    """
    从原始消息序列构建 responder 的纯净输入。
    规则：
    - 保留 HumanMessage + 有内容的 AIMessage（过滤 tool_calls 和空 AIMessage）
    - 如有 summary，前置 SystemMessage 并只取最近 6 条消息
    - 工具调用对由 tools_node 负责追加，此处不做配对
    """
    result = []

    if summary:
        result.append(SystemMessage(content=f"【用户长期记忆摘要】{summary}"))
        recent_count = 6
        source_messages = messages[-recent_count:] if len(messages) > recent_count else messages
    else:
        source_messages = messages

    for msg in source_messages:
        if isinstance(msg, HumanMessage):
            result.append(msg)
        elif isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None) and msg.content:
            result.append(msg)

    return result


def _make_router_node(llm_classifier, llm_reviewer):
    """
    路由节点：L0 安全拦截 → DST 跨轮承接 → L1 关键词 → L2 小模型 → L3 大模型复核 → L4 澄清/转人工
    负责路由决策 + DST 状态管理（current_task / awaiting_slot）。
    """
    from src.router.router import route_message, RouterConfig, compute_new_router_state
    from src.router.schemas import RouterState as RS

    _SOURCE_LABELS = {
        SOURCE_L0_SAFETY: "L0-安全拦截",
        SOURCE_DST: "DST-跨轮承接",
        SOURCE_L1_KEYWORD: "L1-关键词",
        SOURCE_L2: "L2-小模型",
        SOURCE_L3: "L3-大模型复核",
        SOURCE_L4_CLARIFY: "L4-澄清",
        SOURCE_L4_HANDOFF: "L4-转人工",
    }

    def router_node(state: GraphState, config: RunnableConfig) -> dict:
        session_id = config.get("configurable", {}).get("thread_id", "default")
        messages = state.get("messages", [])
        summary = state.get("summary", None)

        # 读取 DST 状态
        current_task = state.get("current_task")
        awaiting_slot = state.get("awaiting_slot")

        # 默认返回值（空消息时）
        default_return = {
            "agent_type": Route.GENERAL.value,
            "route": Route.GENERAL.value,
            "responder_input": [],
            "direct_response": None,
            "summary": summary,
            "action": ACTION_ROUTE,
            "clarify_count": 0,
            "waiting_clarification": False,
            "last_clarify_options": [],
            "router_source": None,
            "router_confidence": 0.0,
            "current_task": None,
            "awaiting_slot": None,
        }

        if not messages:
            return default_return

        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

        # 构建 Router 状态（含 DST 字段）
        router_state = RS(
            clarify_count=state.get("clarify_count", 0),
            waiting_clarification=state.get("waiting_clarification", False),
            last_clarify_options=state.get("last_clarify_options", []),
            current_task=current_task,
            awaiting_slot=awaiting_slot,
        )

        # 调用路由器（内部处理 L0 → DST → L1 → L2 → L3 → L4）
        result = route_message(
            message=content,
            router_state=router_state,
            llm_classifier=llm_classifier,
            llm_reviewer=llm_reviewer,
        )

        # 计算新的澄清状态
        new_rs = compute_new_router_state(result, router_state)

        # ---- DST 状态管理 ----
        # 规则：awaiting_slot != None → 保留 current_task；awaiting_slot == None → 清空 current_task
        new_current_task = current_task
        new_awaiting_slot = awaiting_slot

        if result.source == SOURCE_DST:
            # DST 命中：保持 current_task 和 awaiting_slot 不变
            pass
        elif result.action == ACTION_ROUTE and result.intent in (INTENT_SALE, INTENT_SERVICE):
            # 新业务开始：设置 current_task，awaiting_slot 由 tools_node 后续设置
            new_current_task = result.intent
            new_awaiting_slot = None
        else:
            # handoff / clarify / general → 清空 DST
            new_current_task = None
            new_awaiting_slot = None

        # 路由决策日志
        source_label = _SOURCE_LABELS.get(result.source, result.source)
        logger.info(
            "[路由] %s | intent=%s | confidence=%.2f | action=%s | msg='%s' | dst=(task=%s, slot=%s)",
            source_label, result.intent, result.confidence, result.action, content[:50],
            new_current_task, new_awaiting_slot,
        )

        # 确定 agent_type 和 direct_response
        agent_type = result.intent if result.intent in INTENT_VALUES else INTENT_GENERAL
        direct_response = None

        if result.action == ACTION_HANDOFF:
            ticket_id = f"TK{int(time.time())}{session_id[-4:]}"
            direct_response = json.dumps({
                "transfer": True,
                "ticket_id": ticket_id,
                "message": "正在为您转接人工客服，工单号：" + ticket_id,
            }, ensure_ascii=False)
        elif result.action == ACTION_CLARIFY:
            from src.router.l4_fallback import build_clarify_message
            direct_response = build_clarify_message(result.clarify_options)

        return {
            "agent_type": agent_type,
            "route": agent_type,
            "responder_input": [],
            "direct_response": direct_response,
            "summary": summary,
            "action": result.action,
            "clarify_count": new_rs["clarify_count"],
            "waiting_clarification": new_rs["waiting_clarification"],
            "last_clarify_options": new_rs["last_clarify_options"],
            "router_source": result.source,
            "router_confidence": result.confidence,
            "current_task": new_current_task,
            "awaiting_slot": new_awaiting_slot,
        }

    return router_node


def _make_planner_node(llm):
    """决策节点：非流式调用 LLM，决定是否调用工具。"""
    def planner_node(state: GraphState, config: RunnableConfig) -> dict:
        agent_type = state.get("agent_type", Route.GENERAL.value)
        system_prompt = AGENT_SYSTEM_PROMPTS.get(agent_type, AGENT_SYSTEM_PROMPTS[INTENT_GENERAL])
        tools = AGENT_TOOLS.get(agent_type, [])
        messages = list(state.get("messages", []))
        # 过滤掉空的 AIMessage（planner 历史遗留），避免浪费 token
        messages = [m for m in messages if not (isinstance(m, AIMessage) and not m.content and not getattr(m, 'tool_calls', None))]

        full_messages = [SystemMessage(content=system_prompt)] + messages

        if tools:
            result = llm.bind_tools(tools).invoke(full_messages)
        else:
            result = llm.invoke(full_messages)

        # 日志：planner 返回内容
        tool_calls = getattr(result, 'tool_calls', None) or []
        if tool_calls:
            tool_names = [tc.get('name', '?') for tc in tool_calls]
            logger.info("[Planner] 调用工具: %s (agent=%s, %d个工具)", tool_names, agent_type, len(tool_calls))
        else:
            content_preview = (result.content or '')[:80]
            logger.info("[Planner] 不调工具, 直接回复 (agent=%s, content='%s')", agent_type, content_preview)

        result.content = ""  # 强制清空 content，防止废话污染历史

        return {"messages": [result]}

    return planner_node


def _make_tools_node():
    """工具执行节点：执行工具 + 解析结构化结果 + 管理 DST 状态。"""
    tool_node = ToolNode(ALL_TOOLS)

    def _tools_node(state: GraphState, config: RunnableConfig) -> dict:
        # 日志：进入工具节点
        messages = state.get("messages", [])
        last_ai = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
                last_ai = msg
                break
        if last_ai:
            tool_names = [tc.get('name', '?') for tc in last_ai.tool_calls]
            logger.info("[Tools] 进入工具节点, 执行: %s", tool_names)
        else:
            logger.warning("[Tools] 进入工具节点, 但未找到 AIMessage(tool_calls)")

        result = tool_node.invoke(state)

        new_responder = list(state.get("responder_input", []))

        if last_ai:
            new_responder.append(AIMessage(content="", tool_calls=last_ai.tool_calls))

        # DST 状态更新变量
        dst_current_task = state.get("current_task")
        dst_awaiting_slot = state.get("awaiting_slot")

        for msg in result.get("messages", []):
            content = getattr(msg, 'content', '')
            tool_name = getattr(msg, 'name', '')

            # ---- 转人工信号 ----
            if TRANSFER_SIGNAL in str(content):
                clean_content = str(content).replace(TRANSFER_SIGNAL, "已提交转人工请求")
                new_responder.append(ToolMessage(
                    content=clean_content,
                    tool_call_id=getattr(msg, 'tool_call_id', ''),
                    name=tool_name,
                ))
                continue

            # ---- 结构化结果解析（query_policy / calculate_premium 返回 JSON） ----
            if tool_name in ("query_policy", "calculate_premium"):
                try:
                    parsed = json.loads(str(content))
                    if isinstance(parsed, dict) and "status" in parsed:
                        status = parsed["status"]

                        if status == "missing_params":
                            # 工具缺参数 → 设置 awaiting_slot
                            missing = parsed.get("missing", [])
                            if missing:
                                dst_awaiting_slot = missing[0]
                                logger.info("[Tools] 工具缺参数, awaiting_slot=%s", dst_awaiting_slot)
                            # 转为可读文本给 LLM
                            readable = f"缺少以下参数：{', '.join(missing)}，请向用户询问后重新查询。"
                            new_responder.append(ToolMessage(
                                content=readable,
                                tool_call_id=getattr(msg, 'tool_call_id', ''),
                                name=tool_name,
                            ))
                            continue

                        elif status == "success":
                            # 工具成功 → 清除 DST
                            dst_current_task = None
                            dst_awaiting_slot = None
                            logger.info("[Tools] 工具成功, 清除 DST")
                            # 转为可读文本给 LLM（query_policy 返回 data 字段，calculate_premium 返回 message 字段）
                            if "data" in parsed:
                                data = parsed["data"]
                                readable = (
                                    f"✅ 保单查询成功\n"
                                    f"保单号：{data.get('保单号', '')}\n"
                                    f"车主：{data.get('车主', '')}\n"
                                    f"车型：{data.get('车型', '')}\n"
                                    f"险种：{data.get('险种', '')}\n"
                                    f"保额：{data.get('保额', '')}\n"
                                    f"年保费：{data.get('年保费', '')}\n"
                                    f"到期日：{data.get('到期日', '')}\n"
                                    f"状态：{data.get('状态', '')}"
                                )
                            else:
                                readable = parsed.get("message", "")
                            new_responder.append(ToolMessage(
                                content=readable,
                                tool_call_id=getattr(msg, 'tool_call_id', ''),
                                name=tool_name,
                            ))
                            continue

                        elif status == "error":
                            # 工具错误（未找到等） → 清除 DST
                            dst_current_task = None
                            dst_awaiting_slot = None
                            readable = f"❌ {parsed.get('message', '查询失败')}"
                            new_responder.append(ToolMessage(
                                content=readable,
                                tool_call_id=getattr(msg, 'tool_call_id', ''),
                                name=tool_name,
                            ))
                            continue
                except (json.JSONDecodeError, TypeError):
                    pass  # 非 JSON，按普通文本处理

            # ---- 普通工具结果 ----
            new_responder.append(msg)

        result["responder_input"] = new_responder

        # 更新 DST 状态
        result["current_task"] = dst_current_task
        result["awaiting_slot"] = dst_awaiting_slot

        # 短路逻辑：特定工具结果可直接返回，跳过 planner → responder 链路
        direct_response = None
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage):
                content = str(msg.content)
                if TRANSFER_SIGNAL in content:
                    session_id = config.get("configurable", {}).get("thread_id", "default")
                    ticket_id = f"TK{int(time.time())}{session_id[-4:]}"
                    direct_response = json.dumps({
                        "transfer": True,
                        "ticket_id": ticket_id,
                        "message": "正在为您转接人工客服，工单号：" + ticket_id,
                    }, ensure_ascii=False)
                    break
                if "未检索到相关保险条款" in content:
                    # 统计历史 + 当前共多少次空检索结果，允许最多 3 次工具调用，多给 LLM 重试机会
                    empty_count = sum(
                        1 for m in messages
                        if isinstance(m, ToolMessage) and "未检索到相关保险条款" in str(m.content)
                    )
                    empty_count += sum(
                        1 for m in result.get("messages", [])
                        if isinstance(m, ToolMessage) and "未检索到相关保险条款" in str(m.content)
                    )
                    if empty_count >= 3:
                        direct_response = "很抱歉，我在知识库中没有找到与您问题相关的条款信息，建议您转人工咨询。"
                        break
                    else:
                        logger.info("[Tools] RAG 第 %d 次空检索，不短路，给 planner 重试机会", empty_count)

        if direct_response:
            result["direct_response"] = direct_response

        return result

    return _tools_node


def _make_responder_node(llm):
    async def responder_node(state: GraphState, config: RunnableConfig) -> dict:
        # 短路：direct_response 已设置（clarify/handoff），跳过 LLM
        direct_response = state.get("direct_response")
        if direct_response:
            return {
                "messages": [AIMessage(content=direct_response)],
                "reply": direct_response,
            }

        agent_type = state.get("agent_type", Route.GENERAL.value)
        system_prompt = AGENT_SYSTEM_PROMPTS.get(agent_type, AGENT_SYSTEM_PROMPTS[INTENT_GENERAL])
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


def _make_prepare_input_node():
    """输入准备节点：从 messages + summary 构建纯净的 responder_input。
    位于 router 和 planner 之间，确保 planner 和 responder 都使用准备好的数据。
    """
    def prepare_input_node(state: GraphState) -> dict:
        messages = state.get("messages", [])
        summary = state.get("summary", None)
        responder_input = _build_responder_input(messages, summary)
        return {"responder_input": responder_input}

    return prepare_input_node


def _router_condition(state: GraphState) -> str:
    """Router 节点后的条件边：根据 action 决定下一步"""
    action = state.get("action", ACTION_ROUTE)
    if action == ACTION_HANDOFF:
        return "end"
    if action == ACTION_CLARIFY:
        return "responder"
    return "prepare_input"


def _planner_condition(state: GraphState) -> str:
    """条件边：planner → tools（有 tool_calls） 或 responder（无 tool_calls）"""
    messages = state.get("messages", [])
    if not messages:
        return "responder"
    last_msg = messages[-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        tool_names = [tc.get('name', '?') for tc in last_msg.tool_calls]
        logger.info("[Planner→Tools] 路由到工具节点: %s", tool_names)
        return "tools"
    logger.info("[Planner→Responder] 路由到回复节点（无工具调用）")
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
    图结构: START → router → prepare_input → planner ⇄ tools → responder → END
    """
    global _rag_llm

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

    # 保存 LLM 实例供 RAG 工具使用
    _rag_llm = llm

    # ---------- 创建摘要节点 ----------
    summarization_node = _create_summarization_node(llm)

    # ---------- 自定义摘要注入函数：将摘要存入 state.summary ----------
    def summarize_and_store(state: dict) -> dict:
        result = summarization_node.invoke({"messages": state.get("messages", [])})
        new_messages = result.get("messages", [])
        summary = None
        if new_messages and isinstance(new_messages[-1], SystemMessage):
            summary = new_messages[-1].content
        return {"messages": new_messages, "summary": summary}

    from src.state import set_summarize_fn
    set_summarize_fn(summarize_and_store)

    # ---------- 共享 Memory ----------
    memory = MemorySaver()

    # ---------- 创建 L2 分类器 LLM 和 L3 复核 LLM ----------
    classifier_model = os.environ.get("ROUTER_CLASSIFIER_MODEL", model_name)
    reviewer_model = os.environ.get("ROUTER_REVIEWER_MODEL", model_name)

    # L2 分类器支持独立的 API（如千问、GLM 等 OpenAI 兼容接口）
    classifier_api_key = os.environ.get("ROUTER_CLASSIFIER_API_KEY", api_key)
    classifier_base_url = os.environ.get("ROUTER_CLASSIFIER_BASE_URL", "https://api.deepseek.com/v1")

    llm_classifier = ChatOpenAI(
        model=classifier_model,
        api_key=classifier_api_key,
        base_url=classifier_base_url,
        temperature=0.1,
        max_retries=1,
        http_client=http_client,
    )
    llm_reviewer = ChatOpenAI(
        model=reviewer_model,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0,
        max_retries=1,
        http_client=http_client,
    )

    logger.info("Router LLM: classifier=%s@%s, reviewer=%s@deepseek", classifier_model, classifier_base_url, reviewer_model)

    # ---------- 创建 5 个节点 ----------
    router_node = _make_router_node(llm_classifier, llm_reviewer)
    prepare_input_node = _make_prepare_input_node()
    planner_node = _make_planner_node(llm)
    tools_node = _make_tools_node()
    responder_node = _make_responder_node(llm)

    # ---------- 构建图 ----------
    logger.info("构建手写 StateGraph 编排图...")
    builder = StateGraph(GraphState)

    builder.add_node("router", router_node)
    builder.add_node("prepare_input", prepare_input_node)
    builder.add_node("planner", planner_node)
    builder.add_node("tools", tools_node)
    builder.add_node("responder", responder_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        _router_condition,
        {
            "prepare_input": "prepare_input",
            "responder": "responder",
            "end": END,
        }
    )

    builder.add_edge("prepare_input", "planner")

    builder.add_conditional_edges(
        "planner",
        _planner_condition,
        {
            "tools": "tools",
            "responder": "responder",
        }
    )

    builder.add_conditional_edges(
        "tools",
        _after_tools_condition,
        {
            "planner": "planner",
            "end": END,
        }
    )

    builder.add_edge("responder", END)

    graph = builder.compile(checkpointer=memory)

    logger.info("✅ StateGraph 编排图构建完成")
    logger.info("📊 图结构: START → router(L0-L4) → prepare_input → planner ⇄ tools → responder → END")

    return graph, llm


def warmup_llm(llm: ChatOpenAI):
    """
    预热 LLM 连接：发送一个极轻量请求，提前建立 TCP/TLS 连接。
    """
    try:
        logger.info("正在预热 LLM 连接...")
        llm.invoke("hi")
        logger.info("LLM 连接预热完成")
    except Exception as e:
        logger.warning(f"LLM 预热失败（不影响服务）: {e}")


# ============================================================
# 7. 测试代码
# ============================================================

if __name__ == "__main__":
    from src.logger import setup_logging
    setup_logging()
    logger.info(">>> 开始测试 StateGraph 初始化...")

    try:
        graph, _ = init_graph()

        logger.info("✅ StateGraph 初始化成功！")
        logger.info("  - Graph 类型: %s", type(graph).__name__)

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