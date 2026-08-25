"""
Router 主入口：串联 L0 → DST → L1 → L2 → L3 → L4
"""

import json
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

from src.router.schemas import RouterResult, RouterState, L2Result, DSTEscapeResult
from src.router.l0_safety import check_safety
from src.router.l1_keyword import check_keywords
from src.router.l2_classifier import classify as l2_classify
from src.router.l3_reviewer import review as l3_review
from src.router.l4_fallback import generate_clarify_options, build_clarify_message, should_handoff
from src.router.dst import dst_escape_check, check_cancel_phrase
from src.constants import (
    INTENT_GENERAL, INTENT_HANDOFF, INTENT_CLARIFY,
    ACTION_ROUTE, ACTION_CLARIFY, ACTION_HANDOFF,
    SOURCE_L0_SAFETY, SOURCE_DST, SOURCE_L1_KEYWORD, SOURCE_L2, SOURCE_L2_DST_ESCAPE, SOURCE_L3,
    SOURCE_L4_CLARIFY, SOURCE_L4_HANDOFF,
    SENTIMENT_NEUTRAL,
)
from src.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 配置
# ============================================================

@dataclass
class RouterConfig:
    """Router 阈值配置"""
    l2_accept_threshold: float = 0.7     # L2 >= 此值直接接受
    l2_review_threshold: float = 0.5     # L2 >= 此值且 < accept → L3
    l3_accept_threshold: float = 0.6     # L3 >= 此值接受


_DEFAULT_CONFIG = RouterConfig()


# ============================================================
# 辅助函数
# ============================================================

def _reset_clarify_state() -> dict:
    """返回澄清状态重置值"""
    return {
        "clarify_count": 0,
        "waiting_clarification": False,
        "last_clarify_options": [],
    }


def _log_route(result: RouterResult, message: str, l1_result: tuple,
               l2_result: Optional[L2Result] = None,
               l3_result: Optional[dict] = None,
               clarify_count: int = 0):
    """记录路由日志"""
    log_data = {
        "request_id": f"R{int(time.time() * 1000)}{uuid.uuid4().hex[:4]}",
        "current_message": message[:100],
        "l0_hit": result.source == "l0_safety",
        "l1_result": {"intent": l1_result[0], "source": l1_result[1]} if l1_result[0] else None,
        "l2_intent": l2_result.intent if l2_result else None,
        "l2_confidence": l2_result.confidence if l2_result else None,
        "l2_sentiment": l2_result.sentiment if l2_result else None,
        "l3_called": l3_result is not None,
        "l3_intent": l3_result.get("intent") if l3_result else None,
        "l3_confidence": l3_result.get("confidence") if l3_result else None,
        "final_intent": result.intent,
        "final_action": result.action,
        "clarify_count": clarify_count,
    }
    logger.info("[Router] %s", json.dumps(log_data, ensure_ascii=False))


# ============================================================
# 主路由函数
# ============================================================

def route_message(
    message: str,
    router_state: RouterState,
    llm_classifier,
    llm_reviewer,
    config: Optional[RouterConfig] = None,
) -> RouterResult:
    """
    路由主入口：L0 → DST → L1 → L2 → L3 → L4

    Args:
        message: 当前用户消息文本
        router_state: 跨轮 Router 状态（含 DST 字段）
        llm_classifier: 用于 L2 分类的 LLM 实例
        llm_reviewer: 用于 L3 复核的 LLM 实例
        config: 阈值配置（可选）

    Returns:
        RouterResult
    """
    if config is None:
        config = _DEFAULT_CONFIG

    l1_result = (None, None, False)
    l2_result = None
    l3_result = None

    # ===================== L0：安全拦截 =====================
    safety = check_safety(message)
    if safety:
        result = RouterResult(
            intent=INTENT_HANDOFF,
            confidence=1.0,
            source=SOURCE_L0_SAFETY,
            action=ACTION_HANDOFF,
        )
        _log_route(result, message, l1_result, clarify_count=router_state.clarify_count)
        return result

    # ===================== DST：跨轮承接 =====================
    current_task = router_state.current_task
    awaiting_slot = router_state.awaiting_slot
    dst_l2_result = None  # DST escape 时可能携带可复用的 L2 结果
    
    if awaiting_slot and current_task:
        # 运行逃生判断（规则 → L1 → L2 兜底）
        dst_result: DSTEscapeResult = dst_escape_check(message, current_task, llm_classifier)
    
        if dst_result.decision == "escape":
            logger.info("[DST] 逃生命中, 清空 DST (was task=%s, slot=%s)", current_task, awaiting_slot)
            # 清空 DST 状态
            router_state.current_task = None
            router_state.awaiting_slot = None
            # 保存 L2 结果（如果有），供后续复用
            dst_l2_result = dst_result.l2_result
            # fall through to normal router
        else:
            # continue：继续当前任务
            logger.info("[DST] 继续当前任务: task=%s, slot=%s", current_task, awaiting_slot)
            result = RouterResult(
                intent=current_task,
                confidence=1.0,
                source=SOURCE_DST,
                action=ACTION_ROUTE,
            )
            _log_route(result, message, l1_result, clarify_count=router_state.clarify_count)
            return result
    
    # ===================== DST escape 有 L2 结果：直接复用 =====================
    # DST 阶段调用过 L2 且导致 escape，不允许再次调用 L2。
    # 直接按正常 L2 阈值决定 route / L3 / L4。
    if dst_l2_result is not None:
        l2_result = dst_l2_result
        logger.info("[Router] 复用 DST L2 结果: intent=%s, confidence=%.2f",
                    l2_result.intent, l2_result.confidence)
    
        if l2_result.confidence >= config.l2_accept_threshold:
            result = RouterResult(
                intent=l2_result.intent,
                confidence=l2_result.confidence,
                source=SOURCE_L2_DST_ESCAPE,
                sentiment=l2_result.sentiment,
                action=ACTION_ROUTE,
            )
            _log_route(result, message, l1_result, l2_result, clarify_count=router_state.clarify_count)
            return result
    
        if l2_result.confidence >= config.l2_review_threshold:
            l3_result = l3_review(message, l2_result, llm_reviewer)
            if l3_result["confidence"] >= config.l3_accept_threshold:
                result = RouterResult(
                    intent=l3_result["intent"],
                    confidence=l3_result["confidence"],
                    source=SOURCE_L3,
                    sentiment=l2_result.sentiment,
                    action=ACTION_ROUTE,
                )
                _log_route(result, message, l1_result, l2_result, l3_result,
                           clarify_count=router_state.clarify_count)
                return result
            # L3 不通过 → fall through to L4
        # else: L2 低置信度 → fall through to L4
    
        # 进入 L4（复用 DST L2 结果时跳过 L1）
        new_clarify_count = router_state.clarify_count + 1
        if should_handoff(new_clarify_count):
            result = RouterResult(
                intent=INTENT_HANDOFF,
                confidence=0.0,
                source=SOURCE_L4_HANDOFF,
                sentiment=l2_result.sentiment,
                action=ACTION_HANDOFF,
            )
            _log_route(result, message, l1_result, l2_result, l3_result,
                       clarify_count=new_clarify_count)
            return result
    
        options = generate_clarify_options()
        clarify_msg = build_clarify_message(options)
        result = RouterResult(
            intent=INTENT_CLARIFY,
            confidence=0.0,
            source=SOURCE_L4_CLARIFY,
            sentiment=l2_result.sentiment,
            action=ACTION_CLARIFY,
            clarify_options=options,
        )
        _log_route(result, message, l1_result, l2_result, l3_result,
                   clarify_count=new_clarify_count)
        return result
    
    # ===================== 取消短语检测 =====================
    # DST escape 后（无 L2 结果）或无 DST 时，检查消息是否包含取消短语。
    # 如果包含，跳过 L1（避免"不查了"中的"查"被误匹配为 service 关键词），
    # 直接交给 L2 分类（L2 会将其分类为 general）。
    has_cancel_phrase = check_cancel_phrase(message)
    if has_cancel_phrase:
        logger.info("[Router] 命中取消短语，跳过 L1")
    
    # ===================== L1：关键词层 =====================
    if not has_cancel_phrase:
        l1_intent, l1_source, l1_unique = check_keywords(message)
        l1_result = (l1_intent, l1_source, l1_unique)
    
        if l1_intent is not None:
            result = RouterResult(
                intent=l1_intent,
                confidence=1.0,
                source=SOURCE_L1_KEYWORD,
                action=ACTION_ROUTE,
            )
            _log_route(result, message, l1_result, clarify_count=router_state.clarify_count)
            return result
    
    # ===================== L2：小模型主分类 =====================
    l2_result = l2_classify(message, llm_classifier)

    # ---- L2 阈值决策 ----
    if l2_result.confidence >= config.l2_accept_threshold:
        # L2 高置信度 → 直接接受
        result = RouterResult(
            intent=l2_result.intent,
            confidence=l2_result.confidence,
            source=SOURCE_L2,
            sentiment=l2_result.sentiment,
            action=ACTION_ROUTE,
        )
        _log_route(result, message, l1_result, l2_result, clarify_count=router_state.clarify_count)
        return result

    if l2_result.confidence >= config.l2_review_threshold:
        # L2 中等置信度 → L3 复核
        l3_result = l3_review(message, l2_result, llm_reviewer)

        if l3_result["confidence"] >= config.l3_accept_threshold:
            # L3 通过 → 接受 L3 结果
            result = RouterResult(
                intent=l3_result["intent"],
                confidence=l3_result["confidence"],
                source=SOURCE_L3,
                sentiment=l2_result.sentiment,
                action=ACTION_ROUTE,
            )
            _log_route(result, message, l1_result, l2_result, l3_result,
                       clarify_count=router_state.clarify_count)
            return result
        else:
            # L3 不通过 → L4
            pass  # fall through to L4
    else:
        # L2 低置信度 → 直接 L4
        pass  # fall through to L4

    # ===================== L4：澄清 / 转人工 =====================
    new_clarify_count = router_state.clarify_count + 1

    if should_handoff(new_clarify_count):
        # 连续两次澄清无效 → 转人工
        result = RouterResult(
            intent=INTENT_HANDOFF,
            confidence=0.0,
            source=SOURCE_L4_HANDOFF,
            sentiment=l2_result.sentiment if l2_result else SENTIMENT_NEUTRAL,
            action=ACTION_HANDOFF,
        )
        _log_route(result, message, l1_result, l2_result, l3_result,
                   clarify_count=new_clarify_count)
        return result

    # 第一次/非连续澄清 → 生成选项
    options = generate_clarify_options()
    clarify_msg = build_clarify_message(options)

    result = RouterResult(
        intent=INTENT_CLARIFY,
        confidence=0.0,
        source=SOURCE_L4_CLARIFY,
        sentiment=l2_result.sentiment if l2_result else SENTIMENT_NEUTRAL,
        action=ACTION_CLARIFY,
        clarify_options=options,
    )
    _log_route(result, message, l1_result, l2_result, l3_result,
               clarify_count=new_clarify_count)
    return result


# ============================================================
# 状态更新辅助（供 router_node 调用）
# ============================================================

def compute_new_router_state(
    result: RouterResult,
    current_state: RouterState,
) -> dict:
    """
    根据路由结果计算新的 Router 状态（用于更新 GraphState）。
    返回 dict，可直接 merge 到 GraphState 更新中。

    注意：DST 字段（current_task / awaiting_slot）由 router_node 单独管理，
    此函数只负责澄清相关状态。
    """
    if result.action == ACTION_ROUTE:
        # 成功路由 → 重置澄清状态
        return {
            "clarify_count": 0,
            "waiting_clarification": False,
            "last_clarify_options": [],
        }

    if result.action == ACTION_HANDOFF:
        # 转人工 → 重置澄清状态
        return {
            "clarify_count": 0,
            "waiting_clarification": False,
            "last_clarify_options": [],
        }

    if result.action == ACTION_CLARIFY:
        # 澄清 → 更新计数和选项
        return {
            "clarify_count": current_state.clarify_count + 1,
            "waiting_clarification": True,
            "last_clarify_options": result.clarify_options,
        }

    # fallback
    return {
        "clarify_count": 0,
        "waiting_clarification": False,
        "last_clarify_options": [],
    }
