"""
路由器数据结构定义
"""

from dataclasses import dataclass, field
from typing import List, Optional
from src.constants import ACTION_ROUTE, SENTIMENT_NEUTRAL


@dataclass
class RouterResult:
    """Router 单次调用的最终输出"""
    intent: str              # sale / service / general
    confidence: float        # 0~1
    source: str              # l0_safety / dst / l1_keyword / l2 / l3 / l4_clarify / l4_handoff
    sentiment: str = SENTIMENT_NEUTRAL       # positive / neutral / negative
    action: str = ACTION_ROUTE            # route / clarify / handoff
    clarify_options: List[str] = field(default_factory=list)


@dataclass
class RouterState:
    """跨轮持久化的 Router 状态（存入 GraphState）"""
    clarify_count: int = 0
    waiting_clarification: bool = False
    last_clarify_options: List[str] = field(default_factory=list)
    # ---- DST 状态 ----
    current_task: Optional[str] = None      # sale / service / None
    awaiting_slot: Optional[str] = None     # 当前等待补充的参数名（如 id_card）


@dataclass
class L2Result:
    """L2 小模型分类结果"""
    intent: str
    confidence: float
    sentiment: str
    # top2 备选（新 prompt 输出 alternatives）：[{"intent": str, "confidence": float}]
    alternatives: List[dict] = field(default_factory=list)

    @property
    def second_intent(self) -> str:
        """第二候选意图，无则空串"""
        if self.alternatives:
            return str(self.alternatives[0].get("intent", ""))
        return ""

    @property
    def second_confidence(self) -> float:
        """第二候选置信度，无则 0.0"""
        if self.alternatives:
            try:
                return float(self.alternatives[0].get("confidence", 0.0))
            except (ValueError, TypeError):
                return 0.0
        return 0.0

    @property
    def margin(self) -> float:
        """top1-top2 置信度差；无第二候选时视为果断，返回 confidence 本身"""
        if not self.alternatives:
            return self.confidence
        return self.confidence - self.second_confidence


@dataclass
class DSTEscapeResult:
    """DST Escape Check 的结构化返回"""
    decision: str                          # "escape" / "continue"
    l2_result: Optional[L2Result] = None   # DST 阶段调用过 L2 时保存结果，否则 None
