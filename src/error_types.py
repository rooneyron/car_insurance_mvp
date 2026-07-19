from enum import Enum

class ErrorCode(str, Enum):
    # LLM 层
    API_KEY_INVALID = "API_KEY_INVALID"
    API_TIMEOUT = "API_TIMEOUT"
    API_5XX = "API_5XX"
    API_NETWORK_ERROR = "API_NETWORK_ERROR"
    API_UNKNOWN = "API_UNKNOWN"
    
    # RAG 层
    FAISS_LOAD_FAILED = "FAISS_LOAD_FAILED"
    RERANK_LOAD_FAILED = "RERANK_LOAD_FAILED"
    RERANK_TIMEOUT = "RERANK_TIMEOUT"
    RAG_SCORE_LOW = "RAG_SCORE_LOW"
    
    # 输入层
    INPUT_EMPTY = "INPUT_EMPTY"
    INPUT_TOO_LONG = "INPUT_TOO_LONG"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    SESSION_BUSY = "SESSION_BUSY"
    
    # 系统层
    GRAPH_RECURSION_LIMIT = "GRAPH_RECURSION_LIMIT"
    UNKNOWN = "UNKNOWN"

USER_ERROR_MESSAGES = {
    ErrorCode.API_KEY_INVALID: "系统服务暂时不可用，请稍后再试",
    ErrorCode.API_TIMEOUT: "系统响应超时，请稍后再试",
    ErrorCode.API_5XX: "系统服务暂时不可用，请稍后再试",
    ErrorCode.API_NETWORK_ERROR: "网络连接异常，请稍后再试",
    ErrorCode.API_UNKNOWN: "系统服务暂时不可用，请稍后再试",
    ErrorCode.FAISS_LOAD_FAILED: "知识库暂时无法访问，已为您切换到基础模式",
    ErrorCode.RERANK_LOAD_FAILED: "知识库暂时无法访问，已为您切换到基础模式",
    ErrorCode.RERANK_TIMEOUT: "检索超时，已为您切换到基础模式",
    ErrorCode.RAG_SCORE_LOW: "未找到足够相关的条款，已为您切换到基础模式",
    ErrorCode.INPUT_EMPTY: "请输入内容",
    ErrorCode.INPUT_TOO_LONG: "输入内容过长，请控制在1000字符以内",
    ErrorCode.BUDGET_EXCEEDED: "今日 Token 配额已用完，请明天再试",
    ErrorCode.SESSION_BUSY: "当前会话正在处理中，请稍后再试",
    ErrorCode.GRAPH_RECURSION_LIMIT: "处理超时，请重新提问",
    ErrorCode.UNKNOWN: "系统繁忙，请稍后再试",
}

DEFAULT_ERROR_MESSAGE = "系统繁忙，请稍后再试"