from enum import Enum
from src.constants import INTENT_SALE, INTENT_SERVICE, INTENT_GENERAL


class Route(str, Enum):
    GENERAL = INTENT_GENERAL
    SALE = INTENT_SALE
    SERVICE = INTENT_SERVICE


ROUTE_LABELS = {
    Route.GENERAL: "💬 闲聊",
    Route.SALE: "💰 售前",
    Route.SERVICE: "🛠️ 售后",
}