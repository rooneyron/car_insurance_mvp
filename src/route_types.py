from enum import Enum

class Route(str, Enum):
    GENERAL = "general"
    SALE = "sale"
    SERVICE = "service"

ROUTE_LABELS = {
    Route.GENERAL: "💬 闲聊",
    Route.SALE: "💰 报价链",
    Route.SERVICE: "🛠️ 理赔链",
}