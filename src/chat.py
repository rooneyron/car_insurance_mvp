"""
核心对话逻辑
调用 StateGraph 编排图，处理 Token 统计、转人工等业务逻辑。
路由决策已移入图内，由 route 节点负责。
"""

import time
import json
import uuid
import threading
from src.route_types import Route
from src.context import set_trace_id
from src.error_types import ErrorCode, USER_ERROR_MESSAGES, DEFAULT_ERROR_MESSAGE
from src.constants import TRANSFER_SIGNAL, TOOL_TRANSFER_NAME, MAX_INPUT_LENGTH, GRAPH_RECURSION_LIMIT
from src.token_usage import add_tokens, get_today_usage, is_budget_exceeded
from src.timer import Timer
from src.timing_callback import create_timing_handler
from src import state
from src.logger import get_logger
from langgraph.errors import GraphRecursionError

logger = get_logger(__name__)


# ============================================================
# Session 内存锁：防止同一 session 并发请求
# ============================================================
_session_locks = {}          # session_id -> threading.Lock
_session_locks_guard = threading.Lock()  # 保护 dict 的元锁


def _try_acquire_session_lock(session_id: str) -> bool:
    """尝试获取 session 锁，返回是否成功（非阻塞）"""
    with _session_locks_guard:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        lock = _session_locks[session_id]
    return lock.acquire(blocking=False)


def _release_session_lock(session_id: str):
    """释放 session 锁并清理"""
    with _session_locks_guard:
        lock = _session_locks.pop(session_id, None)
    if lock is not None:
        try:
            lock.release()
        except RuntimeError:
            pass  # 已经释放


def _extract_usage(result_dict):
    """从 LangGraph 结果中提取 usage_metadata"""
    if "usage_metadata" in result_dict:
        return result_dict["usage_metadata"] or {}
    if "messages" in result_dict and result_dict["messages"]:
        last_msg = result_dict["messages"][-1]
        if hasattr(last_msg, "usage_metadata"):
            return last_msg.usage_metadata or {}
        if isinstance(last_msg, dict) and "usage_metadata" in last_msg:
            return last_msg["usage_metadata"] or {}
    return {}


def _check_transfer_flag(result_dict, history_count: int = 0):
    """
    检查是否需要转人工。
    只检查本轮新增的消息（跳过 history_count 之前的历史消息），
    避免 MemorySaver 持久化的旧信号被重复检测。
    """
    messages = result_dict.get("messages", [])
    # 只检查本轮新增的消息
    new_messages = messages[history_count:]
    for msg in new_messages:
        # 检查消息内容中的转人工信号
        if hasattr(msg, 'content') and TRANSFER_SIGNAL in str(msg.content):
            return True
        # 检查 tool_calls 中是否调用了 transfer_to_human
        tool_calls = getattr(msg, 'tool_calls', None) or []
        for tc in tool_calls:
            if tc.get('name') == TOOL_TRANSFER_NAME:
                return True
    return False


def _process_usage(result):
    """提取 Token 使用量并累计"""
    usage = _extract_usage(result)
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cached_tokens = usage.get("input_token_details", {}).get("cache_read", 0)
    add_tokens(input_tokens, output_tokens)
    return input_tokens, output_tokens, cached_tokens


def _log_token_and_perf(session_id, route, input_tokens, output_tokens, cached_tokens, timer, handler):
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
    
    # 从 timing handler 取 LLM/Tool 细粒度记录
    detail_records = handler.drain_records()
    
    report_lines = []
    for rec in detail_records:
        report_lines.append(f"  ├── {rec['label']}: {rec['ms']:.0f}ms")
    # Timer 的粗粒度记录（总耗时、图执行）
    report_lines.append(timer.get_report())
    
    logger.info("性能日志 - 总耗时: %.0fms\n%s", timer.get_total_ms(), "\n".join(report_lines))


def _error_response(error_code: ErrorCode):
    """构造统一格式的错误响应"""
    return {
        "success": -1,
        "error_msg": USER_ERROR_MESSAGES.get(error_code, DEFAULT_ERROR_MESSAGE),
        "content": {}
    }


def chat_api(session_id: str, message: str) -> dict:
    """
    核心对话接口
    调用 StateGraph 编排图，图内自动完成路由和 Agent 调度。
    """
    # 设置 trace_id（API 入口）
    trace_id = f"TR{int(time.time() * 1000)}{uuid.uuid4().hex[:4]}"
    set_trace_id(trace_id)

    # ---------- Session 并发锁 ----------
    if not _try_acquire_session_lock(session_id):
        return _error_response(ErrorCode.SESSION_BUSY)

    try:
        return _chat_api_inner(session_id, message)
    finally:
        _release_session_lock(session_id)


def _chat_api_inner(session_id: str, message: str) -> dict:
    """chat_api 内部实现（已持有 session 锁）"""
    # ---------- 输入校验 ----------
    if not message or not message.strip():
        return _error_response(ErrorCode.INPUT_EMPTY)
    if len(message) > MAX_INPUT_LENGTH:
        return _error_response(ErrorCode.INPUT_TOO_LONG)

    # ---------- 每日 Token 限额检查 ----------
    if is_budget_exceeded():
        return _error_response(ErrorCode.BUDGET_EXCEEDED)

    timer = Timer()
    timer.start("总耗时")

    logger.debug("收到消息: %s...", message[:30])

    # 配置：thread_id 用于 Memory 持久化，callbacks 用于性能监控
    timing_handler = create_timing_handler()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
        "callbacks": [timing_handler],
    }

    try:
        timer.start("图执行")

        # ---------- 入口摘要：在进入图之前压缩历史消息 ----------
        history_count = 0
        try:
            prev_state = state.graph.get_state(config).values
            messages = list(prev_state.get("messages", []))
            if messages and state.summarize_fn:
                summarized = state.summarize_fn({"messages": messages})
                state.graph.update_state(config, summarized)
                history_count = len(summarized.get("messages", []))
            else:
                history_count = len(messages)
        except Exception:
            pass

        # 调用编排图（路由 + Agent 调度都在图内完成）
        result = state.graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config
        )

        timer.stop("图执行")

        # ---------- 从图状态提取结果 ----------
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

        # 转人工检查（只检查本轮新增消息）
        transfer_flag = _check_transfer_flag(result, history_count)

        # ---------- 提取 Token 使用量（统一处理） ----------
        input_tokens, output_tokens, cached_tokens = _process_usage(result)
        timer.stop("总耗时")
        _log_token_and_perf(session_id, route, input_tokens, output_tokens, cached_tokens, timer, timing_handler)

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
                "route": route.value,
                "elapsed_ms": timer.get_total_ms()
            }

        # ---------- 正常返回 ----------
        return {
            "success": 0,
            "content": {
                "reply": reply,
                "transfer": False
            },
            "route": route.value,
            "elapsed_ms": timer.get_total_ms()
        }

    except GraphRecursionError as e:
        logger.error("GraphRecursionError: %s", e, exc_info=True)
        return _error_response(ErrorCode.GRAPH_RECURSION_LIMIT)

    except Exception as e:
        logger.error("chat_api 异常: %s", e, exc_info=True)
        return _error_response(ErrorCode.UNKNOWN)


async def chat_api_stream(session_id: str, message: str):
    """
    流式对话接口（async generator）
    逐 token 产出回复文本，供 Gradio 实时展示。
    yield 的值为 (partial_text: str, metadata: dict | None)
    """
    # 设置 Trace_id（流式入口）
    trace_id = f"TR{int(time.time() * 1000)}{uuid.uuid4().hex[:4]}"
    set_trace_id(trace_id)

    # ---------- Session 并发锁 ----------
    if not _try_acquire_session_lock(session_id):
        yield USER_ERROR_MESSAGES.get(ErrorCode.SESSION_BUSY, ""), {"error": True}
        return

    try:
        async for item in _chat_api_stream_inner(session_id, message):
            yield item
    finally:
        _release_session_lock(session_id)


async def _chat_api_stream_inner(session_id: str, message: str):
    """chat_api_stream 内部实现（已持有 session 锁）"""
    # ---------- 输入校验 ----------
    if not message or not message.strip():
        yield _error_response(ErrorCode.INPUT_EMPTY).get("error_msg", ""), {"error": True}
        return
    if len(message) > MAX_INPUT_LENGTH:
        yield _error_response(ErrorCode.INPUT_TOO_LONG).get("error_msg", ""), {"error": True}
        return
    if is_budget_exceeded():
        yield _error_response(ErrorCode.BUDGET_EXCEEDED).get("error_msg", ""), {"error": True}
        return

    timer = Timer()
    timer.start("总耗时")
    logger.debug("[stream] 收到消息: %s...", message[:30])

    timing_handler = create_timing_handler()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
        "callbacks": [timing_handler],
    }

    try:
        # ---------- 入口摘要：在进入图之前压缩历史消息 ----------
        try:
            prev_state = state.graph.get_state(config).values
            messages = list(prev_state.get("messages", []))
            if messages and state.summarize_fn:
                summarized = state.summarize_fn({"messages": messages})
                state.graph.update_state(config, summarized)
        except Exception:
            pass

        full_text = ""
        current_text = ""
        has_tool_calls = False
        transfer_detected = False  # 检测转人工工具调用
        route = Route.GENERAL
        text_buffer = ""  # 缓冲 LLM 文本，防止工具调用时闪烁

        try:
            async for event in state.graph.astream_events(
                {"messages": [{"role": "user", "content": message}]},
                config=config,
                version="v2",
            ):
                kind = event.get("event", "")

                # LLM 逐 token 流式输出
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        # 检查是否包含 tool_calls（LLM 决定调用工具）
                        tool_call_chunks = getattr(chunk, 'tool_call_chunks', None)
                        if tool_call_chunks:
                            # 工具调用开始，丢弃缓冲文本
                            has_tool_calls = True
                            text_buffer = ""
                            current_text = ""
                        elif chunk.content and isinstance(chunk.content, str):
                            if has_tool_calls:
                                # 工具调用后的新 LLM 输出，直接流式显示
                                current_text += chunk.content
                                full_text = current_text
                                yield current_text, None
                            else:
                                # 首次 LLM 输出，缓冲（可能后续会调工具）
                                text_buffer += chunk.content

                # 工具调用执行中（丢弃缓冲）
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "")
                    if tool_name == TOOL_TRANSFER_NAME:
                        transfer_detected = True
                    text_buffer = ""
                    current_text = ""
                    has_tool_calls = True

                # LLM 单轮结束
                elif kind == "on_chat_model_end":
                    # 检测文本中的转人工信号（流式 chunk 可能切断信号词，在累积文本中检查）
                    if text_buffer and TRANSFER_SIGNAL in text_buffer:
                        transfer_detected = True
                        text_buffer = text_buffer.replace(TRANSFER_SIGNAL, "")
                    if has_tool_calls and current_text and TRANSFER_SIGNAL in current_text:
                        transfer_detected = True
                        current_text = current_text.replace(TRANSFER_SIGNAL, "")

                    if text_buffer and not has_tool_calls:
                        # 没有工具调用，输出缓冲文本
                        current_text = text_buffer
                        full_text = text_buffer
                        text_buffer = ""
                        yield current_text, None
                    elif has_tool_calls:
                        # 工具调用后的 LLM 结束，重置缓冲
                        text_buffer = ""

                # 捕获路由结果
                elif kind == "on_chain_end":
                    data = event.get("data", {})
                    output = data.get("output", {})
                    if isinstance(output, dict) and "route" in output:
                        try:
                            route = Route(output["route"])
                        except (ValueError, KeyError):
                            pass

        except GraphRecursionError as e:
            logger.error("[stream] GraphRecursionError: %s", e, exc_info=True)
            yield USER_ERROR_MESSAGES.get(ErrorCode.GRAPH_RECURSION_LIMIT, ""), {"error": True}
            return
        except Exception as e:
            logger.error("[stream] 异常: %s", e, exc_info=True)
            yield USER_ERROR_MESSAGES.get(ErrorCode.UNKNOWN, ""), {"error": True}
            return

        timer.stop("总耗时")
        elapsed_ms = timer.get_total_ms()

        # ---------- 后处理 ----------
        # 转人工信号（基于工具调用检测）
        transfer = False
        if transfer_detected:
            transfer = True
            ticket_id = f"TK{int(time.time())}{session_id[-4:]}"
            # 追加转人工提示和工单号
            current_text += f"\n\n\u200b\n\n\U0001f504 已为您转接人工客服，工单号：{ticket_id}"
            full_text = current_text

        # 记录性能日志
        detail_records = timing_handler.drain_records()
        report_lines = [f"  \u251c\u2500\u2500 {r['label']}: {r['ms']:.0f}ms" for r in detail_records]
        report_lines.append(timer.get_report())
        logger.info("[stream] 性能日志 - 总耗时: %.0fms\n%s", elapsed_ms, "\n".join(report_lines))

        metadata = {
            "route": route.value,
            "transfer": transfer,
            "elapsed_ms": elapsed_ms,
        }
        yield current_text, metadata

    except Exception as e:
        logger.error("[stream] 未预期异常: %s", e, exc_info=True)
        yield USER_ERROR_MESSAGES.get(ErrorCode.UNKNOWN, ""), {"error": True}