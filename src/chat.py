"""
核心对话逻辑
调用 StateGraph 编排图，处理 Token 统计、转人工等业务逻辑。
路由决策已移入图内，由 route 节点负责。
"""

import time
import json
from src.route_types import Route
from src.error_types import ErrorCode, USER_ERROR_MESSAGES, DEFAULT_ERROR_MESSAGE
from src.constants import TRANSFER_SIGNAL, TOOL_FINISHED_PREFIX
from src.token_usage import add_tokens, get_today_usage, is_budget_exceeded
from src.timer import Timer
from src import state
from src.logger import get_logger
from langgraph.errors import GraphRecursionError

logger = get_logger(__name__)


def _extract_usage(result_dict):
    """从 LangGraph 结果中提取 usage_metadata"""
    if "usage_metadata" in result_dict:
        return result_dict["usage_metadata"]
    if "messages" in result_dict and result_dict["messages"]:
        last_msg = result_dict["messages"][-1]
        if hasattr(last_msg, "usage_metadata"):
            return last_msg.usage_metadata
        if isinstance(last_msg, dict) and "usage_metadata" in last_msg:
            return last_msg["usage_metadata"]
    return {}


def _check_transfer_flag(result_dict):
    """检查是否需要转人工"""
    messages = result_dict.get("messages", [])
    for msg in messages:
        if hasattr(msg, 'content') and TRANSFER_SIGNAL in str(msg.content):
            return True
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                if isinstance(tc, dict) and tc.get('name') == 'transfer_to_human':
                    return True
                elif hasattr(tc, 'get') and tc.get('name') == 'transfer_to_human':
                    return True
        if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
            for tc in msg.additional_kwargs['tool_calls']:
                if isinstance(tc, dict) and tc.get('function', {}).get('name') == 'transfer_to_human':
                    return True
    return False


def _log_token_and_perf(session_id, route, input_tokens, output_tokens, cached_tokens, timer):
    """统一记录 Token 和性能日志"""
    logger.info("Token日志: %s", json.dumps({
        "timestamp": time.time(),
        "session_id": session_id,
        "route": route.value if hasattr(route, 'value') else str(route),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": input_tokens + output_tokens,
        "daily_usage": get_today_usage()
    }, ensure_ascii=False))
    logger.info("性能日志 - 总耗时: %.0fms\n%s", timer.get_total_ms(), timer.get_report())


def chat_api(session_id: str, message: str) -> dict:
    """
    核心对话接口
    调用 StateGraph 编排图，图内自动完成路由和 Agent 调度。
    """
    # ---------- 输入长度校验 ----------
    if len(message) > 1000:
        return {
            "success": -1,
            "error_msg": USER_ERROR_MESSAGES.get(ErrorCode.INPUT_TOO_LONG, DEFAULT_ERROR_MESSAGE),
            "content": {}
        }

    # ---------- 每日 Token 限额检查 ----------
    if is_budget_exceeded():
        return {
            "success": -1,
            "error_msg": "今日 Token 配额已用完，请明天再试",
            "content": {}
        }

    timer = Timer()
    timer.start("总耗时")

    logger.debug("收到消息: %s...", message[:30])

    # 配置：thread_id 用于 Memory 持久化
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}

    try:
        timer.start("图执行")

        # 调用编排图（路由 + Agent 调度都在图内完成）
        result = state.graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config
        )

        timer.stop("图执行")

        # ---------- 从图状态提取结果 ----------
        # 路由结果
        route_str = result.get("route", Route.GENERAL.value)
        try:
            route = Route(route_str)
        except ValueError:
            route = Route.GENERAL

        # 回复文本
        reply = result.get("reply", "")
        if not reply and result.get("messages"):
            last_msg = result["messages"][-1]
            reply = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

        # 转人工检查
        transfer_flag = _check_transfer_flag(result)

        # ---------- 处理 TOOL_FINISHED 信号 ----------
        if reply.startswith(TOOL_FINISHED_PREFIX):
            clean_reply = reply.replace(TOOL_FINISHED_PREFIX, "").strip()
            usage = _extract_usage(result)
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cached_tokens = usage.get("input_token_details", {}).get("cache_read", 0)
            add_tokens(input_tokens, output_tokens)
            timer.stop("总耗时")
            _log_token_and_perf(session_id, route, input_tokens, output_tokens, cached_tokens, timer)
            return {
                "success": 0,
                "content": {
                    "reply": clean_reply or "暂未找到相关信息，建议转人工咨询。",
                    "transfer": False,
                },
                "route": route,
                "elapsed_ms": timer.get_total_ms(),
            }

        timer.stop("总耗时")

        # ---------- 提取 Token 使用量 ----------
        usage = _extract_usage(result)
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cached_tokens = usage.get("input_token_details", {}).get("cache_read", 0)
        add_tokens(input_tokens, output_tokens)

        _log_token_and_perf(session_id, route, input_tokens, output_tokens, cached_tokens, timer)

        # ---------- 处理转人工 ----------
        if transfer_flag:
            ticket_id = f"TK{int(time.time())}{session_id[-4:]}"
            clean_reply = reply.replace(TRANSFER_SIGNAL, "").strip()
            return {
                "success": 0,
                "content": {
                    "reply": clean_reply or "正在为您转接人工客服，请稍候...",
                    "transfer": True,
                    "ticket_id": ticket_id
                },
                "route": route,
                "elapsed_ms": timer.get_total_ms()
            }

        # ---------- 正常返回 ----------
        return {
            "success": 0,
            "content": {
                "reply": reply,
                "transfer": False
            },
            "route": route,
            "elapsed_ms": timer.get_total_ms()
        }

    except GraphRecursionError as e:
        logger.error("GraphRecursionError: %s", e, exc_info=True)
        return {
            "success": -1,
            "error_msg": USER_ERROR_MESSAGES.get(ErrorCode.GRAPH_RECURSION_LIMIT, DEFAULT_ERROR_MESSAGE),
            "content": {}
        }

    except Exception as e:
        logger.error("chat_api 异常: %s", e, exc_info=True)
        return {
            "success": -1,
            "error_msg": USER_ERROR_MESSAGES.get(ErrorCode.UNKNOWN, DEFAULT_ERROR_MESSAGE),
            "content": {}
        }
"""
核心对话逻辑
处理路由决策、Agent 调用、Token 统计、转人工等业务流程。
"""

import time
import json
from src.route_types import Route
from src.error_types import ErrorCode, USER_ERROR_MESSAGES, DEFAULT_ERROR_MESSAGE
from src.constants import TRANSFER_SIGNAL
from src.token_usage import add_tokens, get_today_usage, is_budget_exceeded
from src.timer import Timer
from src import state
from src.logger import get_logger
from langgraph.errors import GraphRecursionError

logger = get_logger(__name__)


def _extract_usage(result_dict):
    """从 LangGraph 结果中提取 usage_metadata"""
    if "usage_metadata" in result_dict:
        return result_dict["usage_metadata"]
    if "messages" in result_dict and result_dict["messages"]:
        last_msg = result_dict["messages"][-1]
        if hasattr(last_msg, "usage_metadata"):
            return last_msg.usage_metadata
        if isinstance(last_msg, dict) and "usage_metadata" in last_msg:
            return last_msg["usage_metadata"]
    return {}


def _check_transfer_flag(result_dict):
    """检查是否需要转人工"""
    messages = result_dict.get("messages", [])
    for msg in messages:
        if hasattr(msg, 'content') and TRANSFER_SIGNAL in str(msg.content):
            return True
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                if isinstance(tc, dict) and tc.get('name') == 'transfer_to_human':
                    return True
                elif hasattr(tc, 'get') and tc.get('name') == 'transfer_to_human':
                    return True
        if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
            for tc in msg.additional_kwargs['tool_calls']:
                if isinstance(tc, dict) and tc.get('function', {}).get('name') == 'transfer_to_human':
                    return True
    return False


def _log_token_and_perf(session_id, route, input_tokens, output_tokens, cached_tokens, timer):
    """统一记录 Token 和性能日志"""
    logger.info("Token日志: %s", json.dumps({
        "timestamp": time.time(),
        "session_id": session_id,
        "route": route.value if hasattr(route, 'value') else str(route),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": input_tokens + output_tokens,
        "daily_usage": get_today_usage()
    }, ensure_ascii=False))
    logger.info("性能日志 - 总耗时: %.0fms\n%s", timer.get_total_ms(), timer.get_report())


def chat_api(session_id: str, message: str) -> dict:
    from src.core.routing import decide_route

    # ---------- 输入长度校验 ----------
    if len(message) > 1000:
        return {
            "success": -1,
            "error_msg": USER_ERROR_MESSAGES.get(ErrorCode.INPUT_TOO_LONG, DEFAULT_ERROR_MESSAGE),
            "content": {}
        }

    # ---------- 每日 Token 限额检查 ----------
    if is_budget_exceeded():
        return {
            "success": -1,
            "error_msg": "今日 Token 配额已用完，请明天再试",
            "content": {}
        }

    timer = Timer()
    timer.start("总耗时")

    logger.debug("收到消息: %s...", message[:30])

    timer.start("路由决策")
    route = decide_route(session_id, message)
    timer.stop("路由决策")
    logger.info("路由决策: session=%s, route=%s", session_id, route)

    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}

    try:
        timer.start("Agent 调用")
        result = None

        if route == Route.GENERAL:
            result = state.general_chain.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config=config
            )
            last_msg = result["messages"][-1]
            reply = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            transfer_flag = False

        elif route == Route.SALE:
            result = state.agent_sale.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config=config
            )
            last_msg = result["messages"][-1]
            reply = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            transfer_flag = _check_transfer_flag(result)
            # 检查工具终止信号
            if reply.startswith("TOOL_FINISHED:"):
                clean_reply = reply.replace("TOOL_FINISHED:", "").strip()
                usage = _extract_usage(result)
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cached_tokens = usage.get("input_token_details", {}).get("cache_read", 0)
                add_tokens(input_tokens, output_tokens)
                timer.stop("Agent 调用")
                timer.stop("总耗时")
                _log_token_and_perf(session_id, route, input_tokens, output_tokens, cached_tokens, timer)
                return {
                    "success": 0,
                    "content": {
                        "reply": clean_reply or "暂未找到相关信息，建议转人工咨询。",
                        "transfer": False,
                    },
                    "route": route,
                    "elapsed_ms": timer.get_total_ms(),
                }

        elif route == Route.SERVICE:
            result = state.agent_service.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config=config
            )
            last_msg = result["messages"][-1]
            reply = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            transfer_flag = _check_transfer_flag(result)

        else:
            return {"success": -1, "error_msg": DEFAULT_ERROR_MESSAGE, "content": {}}

        timer.stop("Agent 调用")
        timer.stop("总耗时")

        # ---------- 提取 Token 使用量 ----------
        usage = _extract_usage(result) if result is not None else {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cached_tokens = usage.get("input_token_details", {}).get("cache_read", 0)
        add_tokens(input_tokens, output_tokens)

        _log_token_and_perf(session_id, route, input_tokens, output_tokens, cached_tokens, timer)

        # ---------- 处理转人工 ----------
        if transfer_flag:
            ticket_id = f"TK{int(time.time())}{session_id[-4:]}"
            clean_reply = reply.replace("__TRANSFER__", "").strip()
            return {
                "success": 0,
                "content": {
                    "reply": clean_reply or "正在为您转接人工客服，请稍候...",
                    "transfer": True,
                    "ticket_id": ticket_id
                },
                "route": route,
                "elapsed_ms": timer.get_total_ms()
            }

        # ---------- 正常返回 ----------
        return {
            "success": 0,
            "content": {
                "reply": reply,
                "transfer": False
            },
            "route": route,
            "elapsed_ms": timer.get_total_ms()
        }

    except GraphRecursionError as e:
        logger.error("GraphRecursionError: %s", e, exc_info=True)
        return {
            "success": -1,
            "error_msg": USER_ERROR_MESSAGES.get(ErrorCode.GRAPH_RECURSION_LIMIT, DEFAULT_ERROR_MESSAGE),
            "content": {}
        }

    except Exception as e:
        logger.error("chat_api 异常: %s", e, exc_info=True)
        return {
            "success": -1,
            "error_msg": USER_ERROR_MESSAGES.get(ErrorCode.UNKNOWN, DEFAULT_ERROR_MESSAGE),
            "content": {}
        }
