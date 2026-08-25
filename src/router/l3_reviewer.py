"""
L3：大模型复核层
仅处理 L2 不确定的边界案例（0.5 <= confidence < 0.7）。
独立重新判断，参考 L2 结果进行复核。
"""

import json
import re
from src.router.schemas import L2Result
from src.router.prompts import L3_PROMPT
from src.constants import INTENT_VALUES, INTENT_GENERAL
from src.logger import get_logger

logger = get_logger(__name__)


def _parse_json_from_text(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def review(message: str, l2_result: L2Result, llm) -> dict:
    """
    调用大模型复核 L2 的分类结果。

    Args:
        message: 当前用户消息
        l2_result: L2 分类结果
        llm: LangChain ChatModel 实例

    Returns:
        dict: {"intent": str, "confidence": float, "reason": str}
    """
    prompt = L3_PROMPT.replace("{message}", message)
    prompt = prompt.replace("{l2_intent}", l2_result.intent)
    prompt = prompt.replace("{l2_confidence}", f"{l2_result.confidence:.2f}")
    prompt = prompt.replace("{l2_sentiment}", l2_result.sentiment)

    try:
        response = llm.invoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)
        logger.debug("[L3] 原始输出: %s", raw_text[:300])

        parsed = _parse_json_from_text(raw_text)
        if not parsed:
            logger.warning("[L3] JSON 解析失败，使用 L2 结果兜底")
            return {
                "intent": l2_result.intent,
                "confidence": l2_result.confidence,
                "reason": "L3 解析失败，沿用 L2 结果",
            }

        intent = parsed.get("intent", l2_result.intent)
        confidence = float(parsed.get("confidence", l2_result.confidence))
        reason = parsed.get("reason", "")

        if intent not in INTENT_VALUES:
            intent = l2_result.intent

        confidence = max(0.0, min(1.0, confidence))

        result = {"intent": intent, "confidence": confidence, "reason": reason}
        logger.info("[L3] intent=%s confidence=%.2f reason=%s", intent, confidence, reason)
        return result

    except Exception as e:
        logger.error("[L3] LLM 调用失败: %s", e)
        return {
            "intent": l2_result.intent,
            "confidence": l2_result.confidence,
            "reason": f"L3 调用异常，沿用 L2: {e}",
        }
