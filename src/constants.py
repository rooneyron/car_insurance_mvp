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

# RAG 空结果时的友好回复
RAG_FALLBACK_MESSAGE = "暂未找到与您问题相匹配的保险条款，建议您转人工客服咨询。我们会为您安排专业人员解答。"

# RAG 无结果时，工具返回的完整指令（含 XML 标签隔离）
RAG_NO_RESULT_INSTRUCTION = (
    "【系统指令】未检索到相关保险条款。"
    "你必须将下方 <fallback> 标签内的文案原封不动地回复给用户，"
    "禁止添加、修改、解释或总结任何内容。\n\n"
    f"<fallback>\n{RAG_FALLBACK_MESSAGE}\n</fallback>"
)