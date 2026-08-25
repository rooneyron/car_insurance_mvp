"""
DST（对话状态跟踪）：逃生判断

当系统存在未完成任务（awaiting_slot != None）时，
判断用户当前消息是否应该离开当前未完成任务。

两级判断：
1. 显式意图（纯规则，不调用 LLM）
   - 显式取消短语 → escape
   - L1 关键词：唯一意图 + 不同业务 → escape
2. L2 分类器兜底（调用现有 l2_classify）
   - L2 判断为 sale/service 且与当前任务不同且置信度 >= 0.75 → escape
   - 其他所有情况 → continue

返回 DSTEscapeResult：
- decision: "escape" / "continue"
- l2_result: DST 阶段调用过 L2 时保存结果（供 Router 复用），否则 None
"""

from src.router.l1_keyword import check_keywords
from src.router.l2_classifier import classify as l2_classify
from src.router.schemas import L2Result, DSTEscapeResult
from src.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 第一层：显式取消短语
# ============================================================

# 明确表达"取消当前任务"的短语。
# 命中任意一个即可直接 escape，不需要调用 LLM。
# 同时供 Router 使用（DST escape 后跳过 L1，避免否定表达被误匹配为业务关键词）。
CANCEL_PHRASES = [
    "算了",
    "不查了",
    "不用了",
    "不问了",
    "取消了",
    "不想了",
    "不弄了",
    "先不查了",
    "不办了",
    "不想查了",
    "不想买了",
]

# ============================================================
# L2 置信度阈值
# ============================================================

# 只有 L2 置信度 >= 此值且为不同业务线时，才认为值得打断当前任务
_L2_ESCAPE_THRESHOLD = 0.75


# ============================================================
# 取消短语检测（供 DST 和 Router 共用）
# ============================================================

def check_cancel_phrase(message: str) -> bool:
    """
    检测消息是否包含显式取消短语。
    供 dst_escape_check 和 Router 共用。
    """
    msg = (message or "").strip()
    if not msg:
        return False
    for phrase in CANCEL_PHRASES:
        if phrase in msg:
            return True
    return False


# ============================================================
# 第一层：显式意图判断（纯规则）
# ============================================================

def _check_explicit_intent(message: str, current_task: str) -> str:
    """
    纯规则判断用户消息是否包含显式 escape 意图。

    规则层只判断 escape（取消、业务切换），不判断 continue。
    非 escape 的消息全部交给 L2 处理。

    返回：
        "cancel_phrase"   — 命中取消短语
        "l1_keyword"      — L1 命中不同业务关键词
        None              — 规则层无法判断，需要进入 L2
    """
    msg = (message or "").strip()
    if not msg:
        return None

    # 1. 检查显式取消短语
    for phrase in CANCEL_PHRASES:
        if phrase in msg:
            logger.info("[DST-explicit] 命中取消短语: %r", phrase)
            return "cancel_phrase"

    # 2. 调用 L1 关键词层
    # 只有同时满足三个条件才 escape：
    #   - is_unique == True（唯一意图）
    #   - l1_intent in (sale, service)（是业务意图）
    #   - l1_intent != current_task（与当前任务不同）
    try:
        l1_intent, l1_source, l1_unique = check_keywords(msg)
        if (
            l1_unique
            and l1_intent in ("sale", "service")
            and l1_intent != current_task
        ):
            logger.info(
                "[DST-explicit] L1 命中不同业务: l1_intent=%s != current_task=%s → escape",
                l1_intent, current_task,
            )
            return "l1_keyword"
    except Exception as exc:
        logger.error("[DST-explicit] L1 调用失败: %s，继续", exc)

    # 规则层无法判断，交给 L2
    return None


# ============================================================
# DST Escape Check 主函数
# ============================================================

def dst_escape_check(
    message: str,
    current_task: str,
    llm,
) -> DSTEscapeResult:
    """
    判断用户当前消息是否应该退出当前 DST 任务。

    判断顺序：
    1. 显式取消短语 → escape (l2_result=None)
    2. L1 唯一不同业务 → escape (l2_result=None)
    3. L2 分类器：不同业务 + confidence >= 0.75 → escape (l2_result=结果)
    4. 其他所有情况 → continue

    Args:
        message: 用户当前消息。
        current_task: 当前 DST 任务："sale" / "service"。
        llm: 用于 L2 分类的小模型实例（与 Router 共用同一个）。

    Returns:
        DSTEscapeResult(decision, l2_result)
    """

    msg = (message or "").strip()

    # 空消息直接继续当前任务，避免误清状态
    if not msg:
        logger.info("[DST-escape] 空消息，默认 continue")
        return DSTEscapeResult(decision="continue")

    # ========================================================
    # 第一层：显式意图判断（纯规则）
    # ========================================================

    rule_source = _check_explicit_intent(msg, current_task)

    if rule_source is not None:
        logger.info("[DST-escape] 规则层 escape: source=%s", rule_source)
        # 取消短语 / L1 直接 escape，没有调用过 L2
        return DSTEscapeResult(decision="escape", l2_result=None)

    # ========================================================
    # 第二层：L2 分类器兜底
    # ========================================================

    try:
        l2_result = l2_classify(msg, llm)

        logger.info(
            "[DST-escape] L2 判断: intent=%s, confidence=%.2f, current_task=%s",
            l2_result.intent,
            l2_result.confidence,
            current_task,
        )

        # escape 条件：
        # 1. L2 intent 是 sale 或 service（是业务意图）
        # 2. 且与当前任务不同（是新业务）
        # 3. 且置信度 >= 0.75（证据充分）
        if (
            l2_result.intent in ("sale", "service")
            and l2_result.intent != current_task
            and l2_result.confidence >= _L2_ESCAPE_THRESHOLD
        ):
            logger.info(
                "[DST-escape] L2 高置信度不同业务: %s (%.2f) != %s → escape",
                l2_result.intent, l2_result.confidence, current_task,
            )
            # L2 导致 escape → 保存 L2 结果供 Router 复用
            return DSTEscapeResult(decision="escape", l2_result=l2_result)

        # 其他所有情况（general、同业务、低置信度）→ continue
        return DSTEscapeResult(decision="continue", l2_result=l2_result)

    except Exception as exc:
        # L2 失败时必须采用保守策略：
        # 不清空当前任务，继续 DST。
        logger.error(
            "[DST-escape] L2 调用失败: %s，默认 continue",
            exc,
        )
        return DSTEscapeResult(decision="continue")
