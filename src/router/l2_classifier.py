"""
L2：小模型主分类器
Router 的核心层，处理绝大多数正常请求。
输入：当前消息
输出：intent / confidence / sentiment
"""

import json
import re
from typing import List
from src.router.schemas import L2Result
from src.router.prompts import L2_PROMPT
from src.constants import INTENT_GENERAL, INTENT_VALUES
from src.constants import SENTIMENT_NEUTRAL, SENTIMENT_VALUES
from src.logger import get_logger

logger = get_logger(__name__)


def _parse_json_from_text(text: str) -> dict:
    """从 LLM 输出中提取 JSON"""
    text = text.strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试提取 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试提取 { ... } 部分
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def classify(message: str, llm) -> L2Result:
    """
    调用 LLM 进行意图分类。

    Args:
        message: 当前用户消息
        llm: LangChain ChatModel 实例

    Returns:
        L2Result
    """
    prompt = L2_PROMPT.replace("{message}", message)

    try:
        response = llm.invoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)
        logger.debug("[L2] 原始输出: %s", raw_text[:300])

        parsed = _parse_json_from_text(raw_text)
        if not parsed:
            logger.warning("[L2] JSON 解析失败，返回默认结果")
            return L2Result(intent=INTENT_GENERAL, confidence=0.2, sentiment=SENTIMENT_NEUTRAL)

        intent = parsed.get("intent", INTENT_GENERAL)
        confidence = float(parsed.get("confidence", 0.2))
        sentiment = parsed.get("sentiment", SENTIMENT_NEUTRAL)

        # 校验 intent 值
        if intent not in INTENT_VALUES:
            logger.warning("[L2] 未知 intent 值: %s，回退为 general", intent)
            intent = INTENT_GENERAL

        # 校验 sentiment 值
        if sentiment not in SENTIMENT_VALUES:
            sentiment = SENTIMENT_NEUTRAL

        # 校验 confidence 范围
        confidence = max(0.0, min(1.0, confidence))

        result = L2Result(
            intent=intent,
            confidence=confidence,
            sentiment=sentiment,
        )
        logger.info("[L2] intent=%s confidence=%.2f sentiment=%s",
                     intent, confidence, sentiment)
        return result

    except Exception as e:
        logger.error("[L2] LLM 调用失败: %s", e)
        return L2Result(intent=INTENT_GENERAL, confidence=0.2, sentiment=SENTIMENT_NEUTRAL)
