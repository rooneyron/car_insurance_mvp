"""
L4：澄清 / 转人工兜底层
当 L2/L3 均无法确定意图时，生成澄清选项让用户选择。
连续两次澄清且用户未提供有效信息 → 转人工。
"""

from typing import List
from src.constants import INTENT_SALE, INTENT_SERVICE, INTENT_GENERAL
from src.logger import get_logger

logger = get_logger(__name__)

# 默认澄清选项（根据 intent 映射动态生成）
_CLARIFY_INTENT_LABELS = {
    INTENT_SALE: "购买/续保车险",
    INTENT_SERVICE: "理赔/事故/保单查询",
    INTENT_GENERAL: "其他问题",
}


def generate_clarify_options() -> List[str]:
    """生成澄清选项列表"""
    return [
        _CLARIFY_INTENT_LABELS[INTENT_SALE],
        _CLARIFY_INTENT_LABELS[INTENT_SERVICE],
        _CLARIFY_INTENT_LABELS[INTENT_GENERAL],
    ]


def build_clarify_message(options: List[str]) -> str:
    """构建澄清消息文本"""
    lines = ["请问您想咨询哪方面的问题？\n"]
    for i, opt in enumerate(options, 1):
        lines.append(f"  {i}. {opt}")
    return "\n".join(lines)


def should_handoff(clarify_count: int) -> bool:
    """
    判断是否应该转人工。
    条件：连续两次澄清且用户均未提供有效信息。
    注意：clarify_count 在调用此函数前已由 router.py 更新。
    """
    return clarify_count >= 2
