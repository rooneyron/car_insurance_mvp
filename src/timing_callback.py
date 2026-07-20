"""
性能监控回调
记录每次 LLM 调用和工具调用的耗时（每次单独一条日志，不汇总）。
同时收集记录供 chat.py 写入性能报告。
"""

import time
import json
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from src.logger import get_logger

logger = get_logger(__name__)


class TimingCallbackHandler(BaseCallbackHandler):
    """
    通过 on_chat_model_* 回调追踪 ChatOpenAI 每次调用耗时。
    用栈匹配 start/end，避免 SummarizationNode 等内部调用导致计数错乱。
    """

    def __init__(self):
        self._llm_start_stack: list[tuple[str, float]] = []  # [(call_id, start_time), ...]
        self._tool_start_times: dict[str, dict] = {}
        self._llm_counter = 0
        self._tool_counter = 0
        self._records: list[dict] = []  # 已完成的耗时记录

    # ========== LLM 调用追踪 ==========

    def _push_llm_start(self):
        self._llm_counter += 1
        call_id = f"llm_{self._llm_counter}"
        self._llm_start_stack.append((call_id, time.time()))

    def _pop_llm_end(self):
        if self._llm_start_stack:
            call_id, start_time = self._llm_start_stack.pop()
            elapsed_ms = (time.time() - start_time) * 1000
            self._records.append({"label": f"{call_id} LLM", "ms": elapsed_ms})

    # --- on_chat_model_* (ChatOpenAI 主回调，最可靠) ---

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs):
        self._push_llm_start()
        # 打印完整 prompt（供调试对比）
        call_id = f"llm_{self._llm_counter}"
        msg_list = messages[0] if messages else []
        prompt_data = []
        total_chars = 0
        for m in msg_list:
            role = m.type if hasattr(m, 'type') else 'unknown'
            # 如果是带 tool_calls 的 AIMessage，显示工具调用信息而非空 content
            if role == 'ai' and getattr(m, 'tool_calls', None):
                tool_names = [tc.get('name', 'unknown') for tc in m.tool_calls]
                content = f"[调用工具: {', '.join(tool_names)}]"
            else:
                content = m.content if hasattr(m, 'content') else str(m)
            total_chars += len(content)
            prompt_data.append({"role": role, "content": content})
        logger.info("📝 [%s] 完整 prompt（%d条消息, %d字符）:\n%s",
                    call_id, len(prompt_data), total_chars,
                    json.dumps(prompt_data, ensure_ascii=False, indent=2))

    def on_chat_model_end(self, response, **kwargs):
        self._pop_llm_end()

    def on_chat_model_error(self, error, **kwargs):
        logger.warning("⏱️ LLM 调用出错: %s", error)

    # --- on_llm_* (底层回调，ChatOpenAI 内部也会触发，用于兜底) ---

    def on_llm_start(self, serialized: Dict[str, Any], prompts: list, **kwargs):
        # 如果 on_chat_model_start 已经触发（栈顶 call_id 的 counter 和当前一致），跳过
        # 否则说明是走 on_llm_start 路径的 LLM 调用
        if self._llm_start_stack:
            last_id = self._llm_start_stack[-1][0]
            last_num = int(last_id.split("_")[1])
            if last_num == self._llm_counter:
                return  # 已被 on_chat_model_start 记录
        self._push_llm_start()

    def on_llm_end(self, response, **kwargs):
        # 只在栈还有未结束的调用时才处理
        if self._llm_start_stack:
            self._pop_llm_end()

    def on_llm_error(self, error, **kwargs):
        logger.warning("⏱️ LLM 调用出错: %s", error)

    # ========== 工具调用追踪 ==========

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        self._tool_counter += 1
        call_id = f"tool_{self._tool_counter}"
        tool_name = serialized.get("name", "unknown")
        self._tool_start_times[call_id] = {"start": time.time(), "name": tool_name}

    def on_tool_end(self, output: str, **kwargs):
        call_id = None
        for cid in reversed(list(self._tool_start_times.keys())):
            if isinstance(self._tool_start_times[cid], dict) and "start" in self._tool_start_times[cid]:
                call_id = cid
                break
        if call_id:
            info = self._tool_start_times.pop(call_id)
            elapsed_ms = (time.time() - info["start"]) * 1000
            self._records.append({"label": f"{call_id} {info['name']}", "ms": elapsed_ms})

    def on_tool_error(self, error, **kwargs):
        logger.warning("⏱️ 工具调用出错: %s", error)

    # ========== 辅助方法 ==========

    def drain_records(self) -> list[dict]:
        """取出并清空所有状态（每次请求调用一次，计数器归零）"""
        records = self._records.copy()
        self._records.clear()
        self._llm_counter = 0
        self._tool_counter = 0
        self._llm_start_stack.clear()
        self._tool_start_times.clear()
        return records


def create_timing_handler() -> TimingCallbackHandler:
    """创建新的 TimingCallbackHandler 实例（每次请求独立）"""
    return TimingCallbackHandler()
