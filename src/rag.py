"""
RAG 向量检索模块
支持两种模式：
- 本地模式（USE_LOCAL_RERANK=true）：FAISS + Cross-Encoder Rerank + 阈值过滤
- 生产模式（USE_LOCAL_RERANK=false）：FAISS 召回 + LLM 相关性分类 + 降级
"""

import os
import re
import pickle
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.constants import RAG_EMPTY_RESULT, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, FAISS_RECALL_TOP_K
from src.logger import get_logger

logger = get_logger(__name__)

# ---------- 全局配置 ----------
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

FAISS_INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PKL_PATH = "data/chunks.pkl"
TERMS_FILE_PATH = "data/insurance_terms.txt"

# RAG 检索质量阈值（Rerank 分数低于此值视为无效）
RAG_SCORE_THRESHOLD = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.6"))

# ---------- 全局变量 ----------
_index = None
_chunks: List[Dict] = []
_embedding_model = None
_reranker = None

# ---------- 工具函数 ----------
def _log_missed_query(query: str, best_score: float = None, faiss_recall: int = None):
    """记录检索失败或低质量的查询"""
    # 简单记录，可扩展
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

    if _embedding_model is None:
        logger.info("正在加载轻量级 Embedding 模型 (fastembed/bge-small-zh-v1.5)...")
        from fastembed import TextEmbedding
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        logger.info("Embedding 模型加载完成")

    chunks = load_and_chunk_terms()
    _chunks = chunks
    texts = [c["full_text"] for c in chunks]

    vectors_generator = _embedding_model.embed(texts)
    vectors = list(vectors_generator)
    vector_array = np.array(vectors).astype('float32')

    dim = vector_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vector_array)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PKL_PATH, "wb") as f:
        pickle.dump(chunks, f)

    logger.info("构建完成，索引已保存至 %s", FAISS_INDEX_PATH)
    return index, chunks


# ---------- 3. 初始化 ----------
def init_rag():
    """初始化 RAG 系统，根据环境变量决定是否加载 Rerank 模型"""
    global _index, _chunks, _embedding_model, _reranker

    _index, _chunks = build_or_load_index()

    if _embedding_model is None:
        logger.info("正在加载轻量级 Embedding 模型...")
        from fastembed import TextEmbedding
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        logger.info("Embedding 模型加载完成")

    use_local_rerank = os.environ.get("USE_LOCAL_RERANK", "true").lower() == "true"
    if not use_local_rerank:
        logger.info("生产环境：跳过加载本地 Rerank 模型（1.1GB）")
        return

    if _reranker is None:
        logger.info("正在加载本地 Rerank 模型 (BAAI/bge-reranker-base)，约 1.1GB...")
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL, max_length=512, local_files_only=True)
        logger.info("Rerank 模型加载完成")


def _faiss_search(query: str, top_k: int = FAISS_RECALL_TOP_K) -> List[str]:
    """FAISS 粗排，返回候选文本列表"""
    if _index is None or not _chunks:
        init_rag()

    query_embedding = list(_embedding_model.embed([query]))[0]
    query_vec = np.array([query_embedding]).astype('float32')

    retrieve_k = min(top_k, len(_chunks))
    distances, indices = _index.search(query_vec, retrieve_k)

    candidates = []
    for idx in indices[0]:
        if 0 <= idx < len(_chunks):
            candidates.append(_chunks[idx]["full_text"])
    return candidates


# ---------- 4. 本地 Rerank 模式 ----------
def search_terms_local(query: str, top_k: int = 3) -> List[str]:
    """
    本地模式：FAISS + Cross-Encoder Rerank + 阈值过滤
    """
    global _index, _chunks, _embedding_model, _reranker

    if _index is None or not _chunks:
        init_rag()

    candidates = _faiss_search(query)
    if not candidates:
        return [RAG_EMPTY_RESULT]

    pairs = [[query, cand] for cand in candidates]
    scores = _reranker.predict(pairs)
    sorted_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    best_score = sorted_results[0][1]
    logger.info("🔍 [RAG] Rerank 最高分=%.4f (阈值=%.2f)", best_score, RAG_SCORE_THRESHOLD)
    if best_score < RAG_SCORE_THRESHOLD:
        logger.info("🔍 [RAG] 最高分低于阈值，返回空结果")
        return [RAG_EMPTY_RESULT]

    return [item[0] for item in sorted_results[:top_k]]


# ---------- 5. 生产模式：LLM 相关性分类 ----------
def filter_candidates_by_llm(query: str, candidates: List[str], llm, min_relevant: int = 2) -> Tuple[List[str], bool]:
    """
    用 LLM 对候选列表进行相关性分类，返回通过相关的候选列表，以及是否满足最小数量。
    """
    if not candidates:
        return [], False

    # 构建 Prompt：让 LLM 对每条候选做“相关/不相关”二元判断
    prompt = f"""你是一个保险条款相关性评估助手。

用户问题："{query}"

请逐一判断以下候选条款是否**直接相关**于用户问题。
判断标准：候选条款的内容是否能直接回答或解释用户的问题。
如果候选条款的内容与问题无关（例如问“太空飞船”但条款是“车损险”），则标记为“不相关”。

候选条款：
{chr(10).join([f'[{i+1}] {text[:200]}...' for i, text in enumerate(candidates)])}

请按以下格式返回结果，每行一个判断：
1: 相关/不相关
2: 相关/不相关
...
只返回判断结果，不要额外解释。
"""
    response = llm.invoke(prompt)
    lines = response.content.strip().split('\n')
    relevant_indices = []
    for line in lines:
        if ':' in line:
            idx_str, label = line.split(':', 1)
            try:
                idx = int(idx_str.strip()) - 1
                if '相关' in label and '不相关' not in label:
                    relevant_indices.append(idx)
            except ValueError:
                continue

    # 去重并按原始顺序
    relevant_indices = sorted(set(relevant_indices))
    # 取出对应的候选文本
    relevant_candidates = [candidates[i] for i in relevant_indices if 0 <= i < len(candidates)]

    is_enough = len(relevant_candidates) >= min_relevant
    return relevant_candidates, is_enough


# ---------- 6. 统一入口 ----------
def search_terms(query: str, top_k: int = 3, llm=None) -> List[str]:
    """
    统一检索入口，根据环境变量 USE_LOCAL_RERANK 决定使用本地模式还是生产模式。
    生产模式需要传入 llm 实例用于相关性分类。
    """
    use_local_rerank = os.environ.get("USE_LOCAL_RERANK", "true").lower() == "true"

    if use_local_rerank:
        return search_terms_local(query, top_k)
    else:
        # 生产模式
        if llm is None:
            raise ValueError("生产模式下必须传入 llm 实例")
        candidates = _faiss_search(query, top_k=FAISS_RECALL_TOP_K)
        if not candidates:
            return [RAG_EMPTY_RESULT]

        relevant, is_enough = filter_candidates_by_llm(query, candidates, llm, min_relevant=2)
        if not is_enough:
            logger.info("🔍 [RAG] LLM 判断相关候选不足2条，返回空结果")
            return [RAG_EMPTY_RESULT]

        # 取前 top_k 条（可进一步排序，此处按原始顺序取前 top_k）
        return relevant[:top_k]


# ---------- 6. 兼容旧接口 ----------
def retrieve_candidates(query: str, top_k: int = 10) -> List[str]:
    """仅执行 FAISS 检索，不进行任何过滤。保留给其他模块使用"""
    return _faiss_search(query, top_k)


# ---------- 7. 测试代码 ----------
if __name__ == "__main__":
    from src.logger import setup_logging
    setup_logging()
    logger.info(">>> 开始测试 RAG 系统...")

    init_rag()

    # 测试本地模式
    os.environ["USE_LOCAL_RERANK"] = "true"
    logger.info("本地模式测试:")
    results = search_terms("车损险赔自然灾害吗", top_k=2)
    logger.info(results)

    # 如果要测试生产模式，需要传入 llm，这里略