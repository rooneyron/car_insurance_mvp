"""
跨模块共享常量
"""

# ---------- 应用元信息 ----------
APP_VERSION = "0.1.0"
SERVICE_NAME = "car_insurance_mvp"

# ---------- RAG 层 ----------
RAG_EMPTY_RESULT = "__RAG_EMPTY__"  # 哨兵值，用于 rag.py 返回空结果时比较
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
FAISS_RECALL_TOP_K = 10

# ---------- 工具层信号 ----------
TOOL_TRANSFER_NAME = "transfer_to_human"

# ---------- 转人工信号 ----------
TRANSFER_SIGNAL = "__TRANSFER__"

# ---------- Router 意图 ----------
INTENT_SALE = "sale"
INTENT_SERVICE = "service"
INTENT_GENERAL = "general"
INTENT_HANDOFF = "handoff"
INTENT_CLARIFY = "clarify"
INTENT_VALUES = {INTENT_SALE, INTENT_SERVICE, INTENT_GENERAL}

# ---------- Router 动作 ----------
ACTION_ROUTE = "route"
ACTION_CLARIFY = "clarify"
ACTION_HANDOFF = "handoff"

# ---------- Router 决策来源 ----------
SOURCE_L0_SAFETY = "l0_safety"
SOURCE_DST = "dst"                    # DST 跨轮承接
SOURCE_L1_KEYWORD = "l1_keyword"
SOURCE_L2 = "l2"
SOURCE_L2_DST_ESCAPE = "l2_dst_escape"  # DST escape 复用的 L2 结果
SOURCE_L3 = "l3"
SOURCE_L4_CLARIFY = "l4_clarify"
SOURCE_L4_HANDOFF = "l4_handoff"

# ---------- 情绪 ----------
SENTIMENT_POSITIVE = "positive"
SENTIMENT_NEUTRAL = "neutral"
SENTIMENT_NEGATIVE = "negative"
SENTIMENT_VALUES = {SENTIMENT_POSITIVE, SENTIMENT_NEUTRAL, SENTIMENT_NEGATIVE}

# ---------- Router 配置路径 ----------
ROUTER_CONFIG_PATH = "config/config.yaml"

# ---------- 输入限制 ----------
MAX_INPUT_LENGTH = 1000
GRAPH_RECURSION_LIMIT = 50

# ---------- JWT ----------
JWT_ALGORITHM = "HS256"

# ---------- API 路径白名单（无需 Token） ----------
PUBLIC_PATHS = {"/health", "/", "/manifest.json", "/favicon.ico", "/queryToken", "/docs", "/openapi.json"}
