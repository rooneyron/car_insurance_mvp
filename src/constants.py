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

# ---------- 输入限制 ----------
MAX_INPUT_LENGTH = 1000
GRAPH_RECURSION_LIMIT = 50

# ---------- JWT ----------
JWT_ALGORITHM = "HS256"

# ---------- API 路径白名单（无需 Token） ----------
PUBLIC_PATHS = {"/health", "/", "/manifest.json", "/favicon.ico", "/queryToken", "/docs", "/openapi.json"}
