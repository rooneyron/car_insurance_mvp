"""
RAG 向量检索模块（纯本地，零外部 API 依赖）
特性：
1. 文本切割（RecursiveCharacterTextSplitter）
2. 本地 Embedding（BAAI/bge-small-zh-v1.5，仅 33MB）
3. 向量持久化（避免重复计算）
4. 双阶段检索（FAISS 粗排 + Cross-Encoder 精排）
"""

import os
import re
import pickle
import time
import numpy as np
from typing import List, Dict, Tuple
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.constants import RAG_EMPTY_RESULT, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, FAISS_RECALL_TOP_K
from src.logger import get_logger

logger = get_logger(__name__)

# ---------- 全局配置 ----------
# 本地 Embedding 模型（33MB，轻量快速）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# 本地 Rerank 模型（1.1GB，精准重排）
RERANK_MODEL = "BAAI/bge-reranker-base"

FAISS_INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PKL_PATH = "data/chunks.pkl"
TERMS_FILE_PATH = "data/insurance_terms.txt"

# RAG 检索质量阈值（Rerank 分数低于此值视为无效，可在 .env 中调整）
RAG_SCORE_THRESHOLD = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.6"))

# ---------- 全局变量 ----------
_index = None
_chunks: List[Dict] = []
_embedding_model = None   # 将使用 fastembed.TextEmbedding
_reranker = None


# ---------- 工具函数 ----------
def _log_missed_query(query: str, best_score: float = None, faiss_recall: int = None):
    """记录检索失败或低质量的查询"""
    pass


# ---------- 1. 文本切割 ----------
def load_and_chunk_terms(file_path: str = TERMS_FILE_PATH) -> List[Dict[str, str]]:
    """读取条款文件，按段落粗分割后，再用字符级切割器切分"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"条款文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_sections = re.split(r'===+', content)
    chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )

    for section in raw_sections:
        section = section.strip()
        if not section:
            continue
        lines = section.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        if len(body) > 500:
            sub_docs = splitter.split_text(body)
            for sub in sub_docs:
                chunks.append({
                    "title": title,
                    "content": sub,
                    "full_text": f"{title}\n{sub}"
                })
        else:
            chunks.append({
                "title": title,
                "content": body,
                "full_text": f"{title}\n{body}"
            })

    logger.info("切割完成，共生成 %d 个文本块", len(chunks))
    return chunks


# ---------- 2. 构建 & 持久化向量库 ----------
def build_or_load_index() -> Tuple[faiss.Index, List[Dict]]:
    """如果本地存在索引则加载，否则构建并保存"""
    global _chunks, _embedding_model

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PKL_PATH):
        logger.info("检测到本地索引文件，正在加载...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PKL_PATH, "rb") as f:
            _chunks = pickle.load(f)
        logger.info("加载成功，共 %d 个块", len(_chunks))
        return index, _chunks

    logger.info("未找到本地索引，开始构建...")

    # 1. 加载 Embedding 模型（使用 fastembed，轻量级）
    if _embedding_model is None:
        logger.info("正在加载轻量级 Embedding 模型 (fastembed/bge-small-zh-v1.5)...")
        from fastembed import TextEmbedding
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        logger.info("Embedding 模型加载完成")

    # 2. 切割文本
    chunks = load_and_chunk_terms()
    _chunks = chunks
    texts = [c["full_text"] for c in chunks]

    # 3. 向量化（fastembed 返回生成器，需转为列表）
    vectors_generator = _embedding_model.embed(texts)
    vectors = list(vectors_generator)  # 每个向量是 numpy 数组
    vector_array = np.array(vectors).astype('float32')

    # 4. 构建 FAISS
    dim = vector_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vector_array)

    # 5. 保存到本地
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PKL_PATH, "wb") as f:
        pickle.dump(chunks, f)

    logger.info("构建完成，索引已保存至 %s", FAISS_INDEX_PATH)
    return index, chunks


# ---------- 3. 初始化 ----------
def init_rag():
    """
    初始化 RAG 系统。
    根据 USE_LOCAL_RERANK 环境变量决定是否加载 1.1GB Rerank 模型。
    """
    global _index, _chunks, _embedding_model, _reranker

    # 加载/构建向量库
    _index, _chunks = build_or_load_index()

    # 加载 Embedding 模型（如果还没加载）
    if _embedding_model is None:
        logger.info("正在加载轻量级 Embedding 模型...")
        from fastembed import TextEmbedding
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        logger.info("Embedding 模型加载完成")

    # ---------- 检查是否跳过 Rerank ----------
    use_local_rerank = os.environ.get("USE_LOCAL_RERANK", "true").lower() == "true"
    if not use_local_rerank:
        logger.info("生产环境：跳过加载本地 Rerank 模型（1.1GB）")
        return  # 直接返回，不加载 Rerank

    # 加载 Rerank 模型（仅在本地开发时加载，延迟导入 sentence_transformers）
    if _reranker is None:
        logger.info("正在加载本地 Rerank 模型 (BAAI/bge-reranker-base)，约 1.1GB...")
        # 延迟导入，避免生产环境安装 sentence_transformers
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL, max_length=512, local_files_only=True)
        logger.info("Rerank 模型加载完成")


def _faiss_search(query: str, top_k: int = FAISS_RECALL_TOP_K) -> list:
    """FAISS 粗排，返回候选文本列表"""
    query_embedding = list(_embedding_model.embed([query]))[0]
    query_vec = np.array([query_embedding]).astype('float32')

    retrieve_k = min(top_k, len(_chunks))
    distances, indices = _index.search(query_vec, retrieve_k)

    candidates = []
    for idx in indices[0]:
        if 0 <= idx < len(_chunks):
            candidates.append(_chunks[idx]["full_text"])
    return candidates


# ---------- 4. 检索 + Rerank ----------
def search_terms(query: str, top_k: int = 3) -> List[str]:
    """
    双阶段检索：
    1. FAISS 粗排（召回 Top-10）
    2. Cross-Encoder 精排（输出 Top-K）
    如果精排最高分低于阈值，返回 ["未找到相关内容"]，并记录日志。
    """
    global _index, _chunks, _embedding_model, _reranker

    if _index is None or not _chunks:
        init_rag()

    # Step A: FAISS 粗排
    candidates = _faiss_search(query)

    # 如果 FAISS 完全搜不到任何候选
    if not candidates:
        _log_missed_query(query, faiss_recall=0)
        return [RAG_EMPTY_RESULT]

    # Step B: Cross-Encoder 精排
    pairs = [[query, cand] for cand in candidates]
    scores = _reranker.predict(pairs)

    sorted_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # 检查最高分是否低于阈值
    best_score = sorted_results[0][1]
    if best_score < RAG_SCORE_THRESHOLD:
        _log_missed_query(query, best_score=best_score)
        return [RAG_EMPTY_RESULT]

    # 质量合格，返回 Top-K
    final_results = [item[0] for item in sorted_results[:top_k]]
    return final_results


def retrieve_candidates(query: str, top_k: int = 10) -> List[str]:
    """
    仅执行 FAISS 检索，不进行 Rerank。
    用于生产环境，配合 LLM 做重排。
    """
    global _index, _chunks, _embedding_model

    if _index is None or not _chunks:
        init_rag()

    return _faiss_search(query, top_k)


# ---------- 5. 测试代码 ----------
if __name__ == "__main__":
    from src.logger import setup_logging
    setup_logging()
    logger.info(">>> 开始测试生产级 RAG 系统（纯本地，零外部 API）...")
    logger.info(">>> 首次运行将自动下载 Embedding 模型（33MB）和 Rerank 模型（1.1GB）...")

    init_rag()

    test_queries = [
        "车损险赔自然灾害吗",
        "第三方责任险免赔率是多少",
        "玻璃险能赔天窗吗",
        "交强险保额是多少",
        "自燃险赔多少"
    ]

    for q in test_queries:
        logger.info("用户问: %s", q)
        results = search_terms(q, top_k=2)
        for i, r in enumerate(results):
            preview = r[:100] + "..." if len(r) > 100 else r
            logger.info("  [%d] (Reranked) %s", i+1, preview)