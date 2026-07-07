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
from sentence_transformers import SentenceTransformer

# ---------- 全局配置 ----------
# 本地 Embedding 模型（33MB，轻量快速）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# 本地 Rerank 模型（1.1GB，精准重排）
RERANK_MODEL = "BAAI/bge-reranker-base"

FAISS_INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PKL_PATH = "data/chunks.pkl"
TERMS_FILE_PATH = "data/insurance_terms.txt"

# RAG 检索质量阈值（Rerank 分数低于此值视为无效）
RAG_SCORE_THRESHOLD = 0.6

# ---------- 全局变量 ----------
_index = None
_chunks: List[Dict] = []
_embedding_model: SentenceTransformer = None
_reranker = None


# ---------- 工具函数 ----------
def _log_missed_query(query: str, best_score: float = None, faiss_recall: int = None):
    """
    记录检索失败或低质量的查询到日志文件
    参数:
        query: 用户查询
        best_score: Rerank 最高分（如果存在）
        faiss_recall: FAISS 召回数量（0 表示完全未命中）
    """
    log_dir = "data"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "missed_queries.log")
    with open(log_path, "a", encoding="utf-8") as f:
        if faiss_recall == 0:
            f.write(f"{time.time()} | query: {query} | FAISS_recall: 0 | best_score: N/A\n")
        else:
            f.write(f"{time.time()} | query: {query} | best_score: {best_score:.3f}\n")


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
        chunk_size=500,
        chunk_overlap=50,
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

    print(f"[RAG] 切割完成，共生成 {len(chunks)} 个文本块")
    return chunks


# ---------- 2. 构建 & 持久化向量库 ----------
def build_or_load_index() -> Tuple[faiss.Index, List[Dict]]:
    """如果本地存在索引则加载，否则构建并保存"""
    global _chunks, _embedding_model

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PKL_PATH):
        print("[RAG] 检测到本地索引文件，正在加载...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PKL_PATH, "rb") as f:
            _chunks = pickle.load(f)
        print(f"[RAG] 加载成功，共 {len(_chunks)} 个块")
        return index, _chunks

    print("[RAG] 未找到本地索引，开始构建...")

    # 1. 加载 Embedding 模型（首次运行自动下载，约 33MB）
    if _embedding_model is None:
        print("[RAG] 正在加载本地 Embedding 模型 (BAAI/bge-small-zh-v1.5)，约 33MB...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        print("[RAG] Embedding 模型加载完成")

    # 2. 切割文本
    chunks = load_and_chunk_terms()
    _chunks = chunks
    texts = [c["full_text"] for c in chunks]

    # 3. 向量化（本地，不调用任何 API）
    vectors = _embedding_model.encode(texts, normalize_embeddings=True)
    vector_array = np.array(vectors).astype('float32')

    # 4. 构建 FAISS
    dim = vector_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vector_array)

    # 5. 保存到本地
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PKL_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[RAG] 构建完成，索引已保存至 {FAISS_INDEX_PATH}")
    return index, chunks


# ---------- 3. 初始化 ----------
def init_rag():
    global _index, _chunks, _embedding_model, _reranker

    # 加载/构建向量库
    _index, _chunks = build_or_load_index()

    # 加载 Embedding 模型（如果还没加载）
    if _embedding_model is None:
        print("[RAG] 正在加载本地 Embedding 模型...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        print("[RAG] Embedding 模型加载完成")

    # 加载 Rerank 模型
    if _reranker is None:
        print("[RAG] 正在加载本地 Rerank 模型 (BAAI/bge-reranker-base)，约 1.1GB...")
        # 国内用户可取消注释下一行使用镜像
        # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL, max_length=512, local_files_only=True)
        print("[RAG] Rerank 模型加载完成")


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
    start_faiss = time.time()
    query_vec = _embedding_model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec).astype('float32')

    retrieve_k = min(10, len(_chunks))
    distances, indices = _index.search(query_vec, retrieve_k)

    candidates = []
    for idx in indices[0]:
        if idx >= 0 and idx < len(_chunks):
            candidates.append(_chunks[idx]["full_text"])

    elapsed_faiss = (time.time() - start_faiss) * 1000
    print(f"[RAG计时] FAISS 检索: {elapsed_faiss:.0f}ms, 召回 {len(candidates)} 个候选")

    # 如果 FAISS 完全搜不到任何候选
    if not candidates:
        _log_missed_query(query, faiss_recall=0)
        return ["未找到相关内容"]

    # Step B: Cross-Encoder 精排
    start_rerank = time.time()
    pairs = [[query, cand] for cand in candidates]
    scores = _reranker.predict(pairs)
    elapsed_rerank = (time.time() - start_rerank) * 1000
    print(f"[RAG计时] Rerank 推理: {elapsed_rerank:.0f}ms (对 {len(candidates)} 个候选)")

    sorted_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # 检查最高分是否低于阈值
    best_score = sorted_results[0][1]
    if best_score < RAG_SCORE_THRESHOLD:
        _log_missed_query(query, best_score=best_score)
        return ["未找到相关内容"]

    # 质量合格，返回 Top-K
    final_results = [item[0] for item in sorted_results[:top_k]]
    return final_results


# ---------- 5. 测试代码 ----------
if __name__ == "__main__":
    print(">>> 开始测试生产级 RAG 系统（纯本地，零外部 API）...")
    print(">>> 首次运行将自动下载 Embedding 模型（33MB）和 Rerank 模型（1.1GB）...")

    init_rag()

    test_queries = [
        "车损险赔自然灾害吗",
        "第三方责任险免赔率是多少",
        "玻璃险能赔天窗吗",
        "交强险保额是多少",
        "自燃险赔多少"
    ]

    for q in test_queries:
        print(f"\n>>> 用户问: {q}")
        results = search_terms(q, top_k=2)
        for i, r in enumerate(results):
            preview = r[:100] + "..." if len(r) > 100 else r
            print(f"  [{i+1}] (Reranked) {preview}")