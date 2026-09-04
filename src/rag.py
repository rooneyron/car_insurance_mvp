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
_last_rag_pipeline_stats = {}  # 最近一次 RAG 管线统计（供评估脚本使用）
_last_rag_query = ""  # 最近一次 RAG 工具接收到的 query（LLM 改写后的）

def get_last_rag_pipeline_stats() -> dict:
    """获取最近一次 RAG 管线统计"""
    return _last_rag_pipeline_stats.copy()

def get_last_rag_query() -> str:
    """获取最近一次 RAG 工具接收到的 query"""
    return _last_rag_query

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


# ---------- 2.5 HF 缓存路径解析 ----------
def _resolve_hf_cached_path(repo_id: str) -> Optional[str]:
    """
    解析 HuggingFace repo_id 对应的本地缓存快照路径。
    transformers>=5.x 的 cached_files 可能无法通过 repo_id 定位缓存，
    此函数直接从 HF hub 缓存目录结构中找到快照路径。
    返回快照路径，未找到则返回 None。
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id and repo.repo_type == "model":
                # 取最新快照
                revisions = sorted(repo.revisions, key=lambda r: r.commit_hash)
                if revisions:
                    return revisions[-1].snapshot_path
    except Exception:
        pass
    return None


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
        model_path = _resolve_hf_cached_path(RERANK_MODEL) or RERANK_MODEL
        _reranker = CrossEncoder(model_path, max_length=512, local_files_only=True)
        logger.info("Rerank 模型加载完成 (path=%s)", model_path)


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


# ---------- 4. 本地 Cross-Encoder Rerank ----------
def _rerank_by_cross_encoder(query: str, candidates: List[str], top_k: int = 3) -> List[str]:
    """
    本地模式：Cross-Encoder Rerank + 阈值过滤
    """
    global _reranker
    
    if _reranker is None:
        logger.warning("Cross-Encoder 未加载，跳过 rerank")
        return candidates[:top_k]
    
    pairs = [[query, cand] for cand in candidates]
    scores = _reranker.predict(pairs)
    sorted_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    best_score = sorted_results[0][1] if sorted_results else 0
    if best_score < RAG_SCORE_THRESHOLD:
        return []
    
    return [item[0] for item in sorted_results[:top_k]]


# ---------- 5. 生产模式：DashScope qwen3-rerank ----------
RERANK_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
DASHSCOPE_RERANK_MODEL = "qwen3-rerank"


def _rerank_by_dashscope(query: str, candidates: List[str], top_k: int = 3) -> List[str]:
    """
    生产模式：DashScope qwen3-rerank API 精排 + 阈值过滤
    """
    import requests
    
    if not candidates:
        return []
    
    api_key = os.environ.get("ROUTER_CLASSIFIER_API_KEY")
    if not api_key:
        logger.warning("ROUTER_CLASSIFIER_API_KEY 未配置，跳过 rerank")
        return candidates[:top_k]
    
    payload = {
        "model": DASHSCOPE_RERANK_MODEL,
        "input": {
            "query": query,
            "documents": candidates
        },
        "parameters": {
            "return_documents": True,
            "top_n": min(len(candidates), len(candidates))
        }
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.post(RERANK_API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            logger.warning("Rerank API 返回 %d，跳过 rerank", resp.status_code)
            return candidates[:top_k]
        
        result = resp.json()
        rerank_results = result.get("output", {}).get("results", [])
        
        if not rerank_results:
            return candidates[:top_k]
        
        sorted_candidates = []
        for r in rerank_results:
            score = r.get("relevance_score", 0)
            if score < RAG_SCORE_THRESHOLD:
                continue
            doc = r.get("document", {})
            if isinstance(doc, dict):
                text = doc.get("text", "")
            else:
                text = str(doc)
            sorted_candidates.append(text)
        
        return sorted_candidates[:top_k] if sorted_candidates else []
        
    except Exception as e:
        logger.warning("Rerank API 调用异常: %s，返回原始候选", e)
        return candidates[:top_k]


# ---------- 6. 统一入口 ----------
def search_terms(query: str, top_k: int = 3, llm=None) -> List[str]:
    """
    统一检索入口。
    召回阶段统一走 hybrid_search（险种识别 + FAISS + BM25 + Metadata过滤 + RRF）。
    精排阶段根据 USE_LOCAL_RERANK 决定：
      - true: Cross-Encoder 本地精排
      - false: DashScope qwen3-rerank API 精排
    """
    # 记录 LLM 改写后的 query + 初始化管线统计
    global _last_rag_query, _last_rag_pipeline_stats
    _last_rag_query = query
    
    # 统一召回：hybrid_search
    recall_results = hybrid_search(query, top_k=30)
    
    if not recall_results:
        _last_rag_pipeline_stats.update({
            "rerank_total_scored": 0, "rerank_above_threshold": 0,
            "final_returned_count": 0, "final_returned_ids": [], "empty_result": True
        })
        return [RAG_EMPTY_RESULT]
    
    # 提取候选内容（保留 id 映射）
    # 精排用 enriched text（含 metadata 前缀），提升 cross-encoder 打分准确度
    candidates = []
    enriched_to_original = {}
    id_map = {}
    for r in recall_results:
        meta = r.get("metadata", {})
        enriched = _build_enriched_text(r["content"], meta)
        candidates.append(enriched)
        enriched_to_original[enriched] = r["content"]
        id_map[r["content"]] = r["id"]
    
    # 精排：根据环境选择
    use_local_rerank = os.environ.get("USE_LOCAL_RERANK", "true").lower() == "true"
    
    if use_local_rerank:
        reranked = _rerank_by_cross_encoder(query, candidates, top_k=top_k)
    else:
        reranked = _rerank_by_dashscope(query, candidates, top_k=top_k)
    
    # 补充精排统计
    if not reranked or reranked == [RAG_EMPTY_RESULT]:
        _last_rag_pipeline_stats.update({
            "rerank_total_scored": len(candidates),
            "rerank_above_threshold": 0,
            "final_returned_count": 0,
            "final_returned_ids": [],
            "empty_result": True
        })
        return [RAG_EMPTY_RESULT]
    
    # enriched text → 原始 content → chunk_id
    original_results = [enriched_to_original.get(text, text) for text in reranked]
    returned_ids = [id_map.get(text, "?") for text in original_results if text in id_map]
    _last_rag_pipeline_stats.update({
        "rerank_total_scored": len(candidates),
        "rerank_above_threshold": len(reranked),
        "final_returned_count": len(reranked),
        "final_returned_ids": returned_ids,
        "empty_result": False
    })
    
    return original_results


# ---------- 6. 兼容旧接口 ----------
def retrieve_candidates(query: str, top_k: int = 10) -> List[str]:
    """仅执行 FAISS 检索，不进行任何过滤。保留给其他模块使用"""
    return _faiss_search(query, top_k)


# =============================================================================
# 混合检索模块：险种识别 + BM25 + Metadata过滤 + RRF融合
# =============================================================================

import json
from collections import defaultdict

# ---------- 险种别名表 ----------
INSURANCE_ALIASES = {
    "车损险": ["车损险", "车损", "机动车损失保险", "车辆损失险", "车损险条款"],
    "三者险": ["三者险", "三者", "第三者", "机动车第三者责任保险", "第三者责任险", "三责险"],
    "车上人员险": ["车上人员", "座位险", "车上人员责任保险", "车上人员责任险", "司乘险"],
    "交强险": ["交强险", "强制险", "交强", "机动车交通事故责任强制保险", "交强险条款"],
}

# 险种标准全称映射
INSURANCE_FULL_NAMES = {
    "车损险": "机动车损失保险",
    "三者险": "机动车第三者责任保险",
    "车上人员险": "机动车车上人员责任保险",
    "交强险": "机动车交通事故责任强制保险",
}

# BM25 自定义词典（车险专业术语）
BM25_CUSTOM_DICT = [
    "机动车损失保险", "机动车第三者责任保险", "机动车车上人员责任保险",
    "机动车交通事故责任强制保险", "保险责任", "责任免除", "赔偿处理",
    "免赔额", "折旧系数", "保险金额", "车上人员", "被保险人", "投保人", "保险人"
]

# 全局变量
_chunk_metadata: Dict[str, Dict] = {}
_bm25_index = None
_bm25_corpus: List[List[str]] = []
_bm25_chunk_ids: List[str] = []


def _load_chunk_metadata() -> Dict[str, Dict]:
    """加载 chunk 元数据"""
    global _chunk_metadata
    if not _chunk_metadata:
        metadata_path = "data/chunk_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                _chunk_metadata = json.load(f)
        else:
            # 从 chunks.pkl 构建
            if os.path.exists(CHUNKS_PKL_PATH):
                with open(CHUNKS_PKL_PATH, "rb") as f:
                    chunks = pickle.load(f)
                _chunk_metadata = {
                    str(i): {"content": c["content"], "metadata": c["metadata"]}
                    for i, c in enumerate(chunks)
                }
    return _chunk_metadata


def _build_enriched_text(content: str, metadata: dict) -> str:
    """拼接 metadata 前缀到 content 前面，用于 BM25 索引和精排。

    格式示例："【三者险】【责任免除】第二十三条..."
    空字段自动跳过。
    """
    parts = []
    ins_type = metadata.get("insurance_type", "")
    if ins_type:
        parts.append(f"【{ins_type}】")
    section = metadata.get("section", "")
    if section:
        parts.append(f"【{section}】")
    return "".join(parts) + content


# ---------- 1. 险种识别 ----------
def detect_insurance_type(query: str) -> Optional[str]:
    """
    基于规则匹配识别查询中的险种类型
    返回: insurance_type 或 None
    """
    for insurance_type, aliases in INSURANCE_ALIASES.items():
        for alias in aliases:
            if alias in query:
                return insurance_type
    return None


# ---------- 2. 查询替换 ----------
def replace_insurance_abbreviation(query: str, insurance_type: Optional[str]) -> str:
    """
    将查询中的险种简称替换为标准全称
    只替换一次，避免重复替换导致的问题
    """
    if insurance_type is None:
        return query
    
    full_name = INSURANCE_FULL_NAMES.get(insurance_type)
    if not full_name:
        return query
    
    # 如果查询中已经包含全称，不需要替换
    if full_name in query:
        return query
    
    # 获取该险种的所有别名
    aliases = INSURANCE_ALIASES.get(insurance_type, [])
    
    # 按长度降序排列，优先替换长别名
    aliases_sorted = sorted(aliases, key=len, reverse=True)
    
    # 只替换第一个匹配的别名
    for alias in aliases_sorted:
        if alias in query:
            return query.replace(alias, full_name, 1)  # 只替换第一次出现
    
    return query


# ---------- 3. BM25 关键词检索 ----------
def _init_bm25():
    """初始化 BM25 索引"""
    global _bm25_index, _bm25_corpus, _bm25_chunk_ids
    
    if _bm25_index is not None:
        return
    
    import jieba
    from rank_bm25 import BM25Okapi
    
    # 添加自定义词典
    for term in BM25_CUSTOM_DICT:
        jieba.add_word(term)
    
    # 加载 chunk 数据
    metadata = _load_chunk_metadata()
    
    _bm25_corpus = []
    _bm25_chunk_ids = []
    
    for chunk_id in sorted(metadata.keys(), key=int):
        chunk_data = metadata[chunk_id]
        content = chunk_data["content"]
        chunk_meta = chunk_data.get("metadata", {})
        # 拼接 insurance_type 和 section 前缀，提升关键词匹配
        enriched = _build_enriched_text(content, chunk_meta)
        tokens = list(jieba.cut(enriched))
        _bm25_corpus.append(tokens)
        _bm25_chunk_ids.append(chunk_id)
    
    # 构建 BM25 索引
    _bm25_index = BM25Okapi(_bm25_corpus)
    logger.info("BM25 索引构建完成，共 %d 条文档", len(_bm25_corpus))


def bm25_search(query: str, top_k: int = 50) -> List[Tuple[str, float]]:
    """
    BM25 关键词检索
    返回: [(chunk_id, score), ...] 按分数降序
    """
    import jieba
    
    _init_bm25()
    
    # 查询分词
    query_tokens = list(jieba.cut(query))
    
    # BM25 检索
    scores = _bm25_index.get_scores(query_tokens)
    
    # 获取 top_k
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((_bm25_chunk_ids[idx], float(scores[idx])))
    
    return results


# ---------- 4. 向量检索（返回 chunk_id + score） ----------
def vector_search(query: str, top_k: int = 50) -> List[Tuple[str, float]]:
    """
    FAISS 向量检索
    返回: [(chunk_id, score), ...] 按分数降序
    """
    if _index is None or not _chunks:
        init_rag()
    
    query_embedding = list(_embedding_model.embed([query]))[0]
    query_vec = np.array([query_embedding]).astype('float32')
    
    retrieve_k = min(top_k, len(_chunks))
    distances, indices = _index.search(query_vec, retrieve_k)
    
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if 0 <= idx < len(_chunks):
            results.append((str(idx), float(score)))
    
    return results


# ---------- 5. Metadata 过滤 ----------
def filter_by_insurance_type(
    results: List[Tuple[str, float]], 
    insurance_type: Optional[str]
) -> List[Tuple[str, float]]:
    """
    根据险种类型过滤结果
    保留 insurance_type 匹配或为"通用"的结果
    """
    if insurance_type is None:
        return results
    
    metadata = _load_chunk_metadata()
    filtered = []
    
    for chunk_id, score in results:
        chunk_meta = metadata.get(chunk_id, {})
        chunk_ins_type = chunk_meta.get("metadata", {}).get("insurance_type", "")
        
        # 保留匹配或通用的结果
        if chunk_ins_type == insurance_type or chunk_ins_type == "通用":
            filtered.append((chunk_id, score))
    
    return filtered


# ---------- 6. RRF 融合 ----------
def rrf_fuse(
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    top_k: int = 10,
    k: int = 60
) -> List[Dict]:
    """
    RRF (Reciprocal Rank Fusion) 融合两路检索结果
    返回: [{"id": "0", "content": "...", "metadata": {...}, "rrf_score": 0.123, ...}, ...]
    """
    metadata = _load_chunk_metadata()
    
    # 计算每个 chunk_id 的 RRF 分数
    rrf_scores = defaultdict(float)
    vector_ranks = {}
    bm25_ranks = {}
    
    # 向量检索排名
    for rank, (chunk_id, _) in enumerate(vector_results, 1):
        vector_ranks[chunk_id] = rank
        rrf_scores[chunk_id] += 1.0 / (k + rank)
    
    # BM25 检索排名
    for rank, (chunk_id, _) in enumerate(bm25_results, 1):
        bm25_ranks[chunk_id] = rank
        rrf_scores[chunk_id] += 1.0 / (k + rank)
    
    # 按 RRF 分数排序
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for chunk_id, rrf_score in sorted_chunks:
        chunk_meta = metadata.get(chunk_id, {})
        results.append({
            "id": chunk_id,
            "content": chunk_meta.get("content", ""),
            "metadata": chunk_meta.get("metadata", {}),
            "rrf_score": rrf_score,
            "vector_rank": vector_ranks.get(chunk_id),
            "bm25_rank": bm25_ranks.get(chunk_id),
        })
    
    return results


# ---------- 7. 混合检索主函数 ----------
def hybrid_search(query: str, top_k: int = 10) -> List[Dict]:
    """
    混合检索主函数
    流程：险种识别 → 查询替换 → 查询扩展 → 向量检索 + BM25 → Metadata过滤 → RRF融合
    """
    from src.query_expander import expand_query
    
    # 1. 险种识别
    insurance_type = detect_insurance_type(query)
    
    # 2. 查询替换
    expanded_query = replace_insurance_abbreviation(query, insurance_type)
    
    # 3. 查询扩展（仅用于 BM25，向量检索用原始 query 避免语义污染）
    bm25_query = expand_query(expanded_query, insurance_type)
    
    # 4. 双路检索
    vector_results = vector_search(expanded_query, top_k=50)
    bm25_results = bm25_search(bm25_query, top_k=50)
    
    # 5. Metadata 过滤
    vector_filtered = filter_by_insurance_type(vector_results, insurance_type)
    bm25_filtered = filter_by_insurance_type(bm25_results, insurance_type)
    
    # 6. RRF 融合
    final_results = rrf_fuse(vector_filtered, bm25_filtered, top_k=top_k)
    
    # 存储中间统计（供 search_terms 组合完整管线数据）
    global _last_rag_pipeline_stats
    _last_rag_pipeline_stats = {
        "vector_returned": len(vector_results),
        "bm25_returned": len(bm25_results),
        "after_filter_vector": len(vector_filtered),
        "after_filter_bm25": len(bm25_filtered),
        "rrf_candidate_pool": len(final_results),
        "rrf_top_ids": [r["id"] for r in final_results[:10]],
    }
    
    # 7. 打印调试信息
    print(f"\n{'='*60}")
    print(f"混合检索调试信息")
    print(f"{'='*60}")
    print(f"原始查询: {query}")
    print(f"识别险种: {insurance_type or 'None'}")
    print(f"向量查询: {expanded_query}")
    print(f"BM25查询: {bm25_query}")
    print(f"向量召回数: {len(vector_results)} → 过滤后: {len(vector_filtered)}")
    print(f"BM25召回数: {len(bm25_results)} → 过滤后: {len(bm25_filtered)}")
    print(f"融合结果数: {len(final_results)}")
    if final_results:
        top_ids = [r["id"] for r in final_results[:5]]
        print(f"Top5 IDs: {top_ids}")
    print(f"{'='*60}\n")
    
    return final_results


# ---------- 8. 测试代码 ----------
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