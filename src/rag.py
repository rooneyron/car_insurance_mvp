"""
RAG 检索模块（ParadeDB 存储层）
双路召回：pgvector(HNSW, 余弦) + pg_search(BM25, whitespace 分词) → RRF 融合 → 精排。
精排两种模式：
- 本地模式（USE_LOCAL_RERANK=true）：Cross-Encoder Rerank + 阈值过滤
- 生产模式（USE_LOCAL_RERANK=false）：DashScope qwen3-rerank API 精排 + 阈值过滤
"""

import os
import re
import time
import threading
from typing import List, Dict, Tuple, Optional
import jieba
import psycopg2
from psycopg2.extras import execute_values, Json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.constants import RAG_EMPTY_RESULT, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, VECTOR_RECALL_TOP_K
from src.db import get_conn
from src.logger import get_logger

logger = get_logger(__name__)

# ---------- 全局配置 ----------
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_MAX_LENGTH = 512         # CrossEncoder 序列长度上限（≠ embedding 维度 EMBEDDING_DIM，勿混淆）

TERMS_FILE_PATH = "data/insurance_terms.txt"

# RAG 检索质量阈值（Rerank 分数低于此值视为无效）
RAG_SCORE_THRESHOLD = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.6"))

# ---------- 全局变量 ----------
_embedding_model = None
_reranker = None
_pg_ready = False  # PG 连通 + 元数据缓存就绪标志（供 search_terms 组件状态日志）
_last_rag_pipeline_stats = {}  # 最近一次 RAG 管线统计（供评估脚本使用）
_last_rag_query = ""  # 最近一次 RAG 工具接收到的 query（LLM 改写后的）

# ---------- 熔断降级（PG 不可用时降级为大模型裸答） ----------
class RAGRetrievalError(Exception):
    """PG 检索异常（连接失败/查询异常）——区别于“检索无结果”的正常业务空返回。"""


# 熔断打开时返回给上层的降级提示：chains.py/Agent 无需任何改动，
# LLM 看到此上下文即会裸答并附带免责声明。
RAG_FALLBACK_NOTICE = (
    "【系统提示】知识库检索服务当前不可用，以下回答未参考车险条款原文，"
    "仅基于通用保险知识，可能存在不准确之处，仅供参考。"
)

# 最近一次 vector/bm25 检索是否触发 psycopg2.Error（供 hybrid_search 判定 PG 运行时故障）
_last_pg_error = False


class SimpleCircuitBreaker:
    """简易熔断器（无第三方依赖）：closed(正常) → open(熔断) → half_open(试探) 状态机。

    仅“PG 连接/查询异常”计为失败；“检索无结果”属正常业务，不计失败、不触发熔断。
    模块级单例，全局共享状态；用 Lock 保护状态迁移。
    """
    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        """是否放行本次检索。open 态超过恢复窗口后自动转 half_open 放行一次试探。"""
        with self._lock:
            if self.state == self.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = self.HALF_OPEN
                    logger.info("[熔断器] open→half_open：已过恢复窗口 %ds，放行一次试探", self.recovery_timeout)
                    return True
                return False
            return True  # CLOSED / HALF_OPEN 均放行

    def record_success(self):
        """检索成功（含正常空结果）：half_open→closed 恢复并重置失败计数。"""
        with self._lock:
            if self.state != self.CLOSED:
                logger.info("[熔断器] %s→closed：PG 恢复正常，重置失败计数", self.state)
            self.state = self.CLOSED
            self.failure_count = 0

    def record_failure(self):
        """PG 异常：累加失败计数。half_open 试探失败立即回 open；closed 达阈值则熔断。"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == self.HALF_OPEN:
                self.state = self.OPEN
                logger.info("[熔断器] half_open→open：试探失败，重新计时 %ds", self.recovery_timeout)
            elif self.failure_count >= self.failure_threshold:
                self.state = self.OPEN
                logger.info("[熔断器] closed→open：连续失败 %d 次，熔断打开", self.failure_count)


# 模块级单例：全局共享熔断状态（连续失败 3 次熔断，30s 后试探恢复）
_circuit_breaker = SimpleCircuitBreaker(failure_threshold=3, recovery_timeout=30)


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
def _check_pg_and_warm_cache():
    """PG 连通性预检（SELECT 1）+ chunk 元数据缓存预热，置 _pg_ready。

    迁移 ParadeDB 后不再降级 FAISS：PG 不可用时检索返回空，由上层兜底。
    """
    global _pg_ready
    try:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        finally:
            conn.close()
        _load_chunk_metadata()
        _pg_ready = True
        logger.info("[RAG-启动] PG 连通 + 元数据缓存预热完成 | %d 条 chunk", len(_chunk_metadata))
    except psycopg2.Error as e:
        _pg_ready = False
        logger.error("[RAG-启动] PG 连通性预检失败：%s（检索将返回空，不降级 FAISS）", e)


def init_rag():
    """初始化 RAG 系统：Embedding 模型 + PG 连通预检 +（本地模式）Rerank 模型"""
    global _embedding_model, _reranker

    if _embedding_model is None:
        logger.info("正在加载轻量级 Embedding 模型...")
        from fastembed import TextEmbedding
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        logger.info("Embedding 模型加载完成")

    _check_pg_and_warm_cache()

    use_local_rerank = os.environ.get("USE_LOCAL_RERANK", "true").lower() == "true"
    if not use_local_rerank:
        logger.info("生产环境：跳过加载本地 Rerank 模型（1.1GB）")
        return

    if _reranker is None:
        logger.info("正在加载本地 Rerank 模型 (BAAI/bge-reranker-base)，约 1.1GB...")
        from sentence_transformers import CrossEncoder
        model_path = _resolve_hf_cached_path(RERANK_MODEL) or RERANK_MODEL
        _reranker = CrossEncoder(model_path, max_length=RERANK_MAX_LENGTH, local_files_only=True)
        logger.info("Rerank 模型加载完成 (path=%s)", model_path)


def init_rag_components():
    """启动时一次性预加载所有 RAG 组件，避免首个请求触发懒加载。

    相比 init_rag()（Embedding + CrossEncoder + PG 预检），额外预加载：
      - jieba 分词器词典 + BM25 自定义术语（否则首请求 Building prefix dict 约 0.35s）
      - PG 连通预检 + chunk 元数据缓存预热
      - 同义词词典
      - CrossEncoder 预热推理（消除首次 predict 的 torch warmup）
    各组件均有全局守卫，幂等可重复调用。生产模式（USE_LOCAL_RERANK=false）跳过本地 CrossEncoder 与 torch import。
    """
    global _embedding_model, _reranker
    t_all = time.time()

    # 生产模式不加载本地 Rerank，也不 import torch（requirements-prod 部署环境无 torch，
    # 无条件 import 会让整个预加载 abort、只剩一条 warning）；仅本地精排模式设置推理线程。
    use_local_rerank = os.environ.get("USE_LOCAL_RERANK", "true").lower() == "true"
    if use_local_rerank:
        # torch CPU 推理线程数：实测本机 8 线程最优（默认仅 6 偏保守，>8 因超线程争抢反而慢）。
        # 只改并行度、不改模型/候选/检索逻辑，精排分数完全一致=零质量损失。可用 RERANK_NUM_THREADS 覆盖。
        import torch
        try:
            _n_threads = int(os.environ.get("RERANK_NUM_THREADS", "8"))
        except ValueError:
            logger.warning("RERANK_NUM_THREADS=%r 非法，回退为默认 8", os.environ.get("RERANK_NUM_THREADS"))
            _n_threads = 8
        torch.set_num_threads(_n_threads)
        logger.info("[RAG-启动] torch推理线程数设为: %d（实测8最优）", _n_threads)

    # ① jieba 分词器（预热词典 + BM25 自定义术语，避免首请求 Building prefix dict）
    t = time.time()
    jieba.initialize()
    _ensure_custom_dict()
    logger.info("[RAG-启动] jieba加载: %.3fs", time.time() - t)

    # ② PG 连通预检 + chunk 元数据缓存预热（ParadeDB 存储层）
    t = time.time()
    _check_pg_and_warm_cache()
    logger.info("[RAG-启动] PG预检+元数据预热: %.3fs | PG连通=%s", time.time() - t, _pg_ready)

    # ③ Embedding 模型
    t = time.time()
    if _embedding_model is None:
        from fastembed import TextEmbedding
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    logger.info("[RAG-启动] Embedding模型加载: %.3fs", time.time() - t)

    # ④ CrossEncoder 精排模型（仅本地精排模式）
    if use_local_rerank:
        t = time.time()
        if _reranker is None:
            from sentence_transformers import CrossEncoder
            model_path = _resolve_hf_cached_path(RERANK_MODEL) or RERANK_MODEL
            _reranker = CrossEncoder(model_path, max_length=RERANK_MAX_LENGTH, local_files_only=True)
        logger.info("[RAG-启动] CrossEncoder模型加载: %.3fs", time.time() - t)
    else:
        logger.info("[RAG-启动] 生产模式：跳过本地 CrossEncoder（精排走 qwen3-rerank API）")

    # ⑤ 同义词词典
    t = time.time()
    from src.query_expander import load_synonym_dict
    _syn = load_synonym_dict()
    logger.info("[RAG-启动] 同义词词典加载: %.3fs | %d条", time.time() - t, len(_syn))

    # ⑥ CrossEncoder 预热推理（消除首次 predict 的 torch warmup）
    if _reranker is not None:
        t = time.time()
        _reranker.predict([["预热", "预热"]])
        logger.info("[RAG-启动] CrossEncoder预热推理: %.3fs", time.time() - t)

    logger.info("[RAG-启动] 全部组件初始化完成: %.3fs", time.time() - t_all)


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

    _t_start = time.time()
    # 组件状态确认：第二次请求起三者应全为 True，否则说明单例未生效
    logger.info("[RAG-请求] 组件状态: jieba已加载=%s | PG连通=%s | CrossEncoder已加载=%s",
                jieba.dt.initialized, _pg_ready, _reranker is not None)
    
    # 熔断闸门：open 态直接降级为大模型裸答提示，不走 PG 检索（避免反复超时浪费）
    if not _circuit_breaker.allow_request():
        logger.info("[熔断降级] 熔断器打开(open)，跳过检索直接返回降级提示 | query=%s", query)
        _last_rag_pipeline_stats.update({
            "rerank_total_scored": 0, "rerank_above_threshold": 0,
            "final_returned_count": 0, "final_returned_ids": [],
            "empty_result": True, "circuit_breaker": "open",
        })
        return [RAG_FALLBACK_NOTICE]

    # 统一召回：hybrid_search（PG 连接/查询异常会抛 RAGRetrievalError）
    try:
        recall_results = hybrid_search(query, top_k=30)
    except RAGRetrievalError as e:
        # PG 异常 → 计失败（连续达阈值则熔断）+ 降级为裸答提示
        _circuit_breaker.record_failure()
        logger.info("[熔断降级] 检索异常，记熔断失败(count=%d, state=%s)：%s | 返回降级提示",
                    _circuit_breaker.failure_count, _circuit_breaker.state, e)
        _last_rag_pipeline_stats.update({
            "rerank_total_scored": 0, "rerank_above_threshold": 0,
            "final_returned_count": 0, "final_returned_ids": [],
            "empty_result": True, "circuit_breaker": "failure",
        })
        return [RAG_FALLBACK_NOTICE]
    else:
        # 正常返回（含“无结果”的合法空返回）→ 记成功；half_open 试探成功即恢复 closed
        _circuit_breaker.record_success()
    
    if not recall_results:
        _last_rag_pipeline_stats.update({
            "rerank_total_scored": 0, "rerank_above_threshold": 0,
            "final_returned_count": 0, "final_returned_ids": [], "empty_result": True
        })
        logger.info("[RAG-计时] 总耗时: %.3fs（无召回，提前返回）", time.time() - _t_start)
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
    
    _t_rerank = time.time()
    if use_local_rerank:
        reranked = _rerank_by_cross_encoder(query, candidates, top_k=top_k)
    else:
        reranked = _rerank_by_dashscope(query, candidates, top_k=top_k)
    logger.info("[RAG-计时] ⑥精排(%s): %.3fs | 输入%d条→输出%d条",
                "CrossEncoder" if use_local_rerank else "qwen3-rerank",
                time.time() - _t_rerank, len(candidates), len(reranked))
    
    # 补充精排统计
    if not reranked or reranked == [RAG_EMPTY_RESULT]:
        _last_rag_pipeline_stats.update({
            "rerank_total_scored": len(candidates),
            "rerank_above_threshold": 0,
            "final_returned_count": 0,
            "final_returned_ids": [],
            "empty_result": True
        })
        logger.info("[RAG-计时] 总耗时: %.3fs（精排后为空）", time.time() - _t_start)
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
    
    logger.info("[RAG-计时] 总耗时: %.3fs", time.time() - _t_start)
    return original_results


# ---------- 6. 兼容旧接口 ----------
def retrieve_candidates(query: str, top_k: int = 10) -> List[str]:
    """仅执行向量检索（ParadeDB pgvector），不做险种过滤。保留给其他模块使用。"""
    hits = vector_search(query, top_k=top_k)
    if not hits:
        return []
    metadata = _load_chunk_metadata()
    return [metadata[cid]["content"] for cid, _ in hits if cid in metadata]


# =============================================================================
# 混合检索模块：险种识别 + BM25 + Metadata过滤 + RRF融合
# =============================================================================

from collections import defaultdict

# 通用条款标记（filter_by_insurance_type 保留匹配险种或通用的结果）
GENERIC_INSURANCE_TYPE = "通用"

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


def _load_chunk_metadata() -> Dict[str, Dict]:
    """从 ParadeDB documents 表加载 chunk 元数据（id→{content,metadata}），带全局缓存。

    id 统一转 str，与 rrf_fuse/filter_by_insurance_type 的字符串 id 约定对齐。
    """
    global _chunk_metadata
    if _chunk_metadata:
        return _chunk_metadata
    try:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, content, metadata FROM documents")
                rows = cur.fetchall()
        finally:
            conn.close()
        _chunk_metadata = {
            str(rid): {"content": content, "metadata": meta}
            for rid, content, meta in rows
        }
    except psycopg2.Error as e:
        logger.error("[RAG] 加载 chunk 元数据失败：%s", e)
        _chunk_metadata = {}
    return _chunk_metadata


def refresh_chunk_metadata_cache() -> Dict[str, Dict]:
    """清空并重新加载 chunk 元数据缓存（灌库后调用，避免读到旧缓存）。"""
    global _chunk_metadata
    _chunk_metadata = {}
    return _load_chunk_metadata()


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


# ---------- 3. BM25 词典/分词 + 关键词检索（ParadeDB pg_search） ----------
_jieba_dict_ready = False


def _ensure_custom_dict():
    """向 jieba 注入车险专业术语词典（幂等，全局守卫）。"""
    global _jieba_dict_ready
    if _jieba_dict_ready:
        return
    for term in BM25_CUSTOM_DICT:
        jieba.add_word(term)
    _jieba_dict_ready = True


# BM25 token 中不允许出现的字符（pg_search whitespace 分词按空格切，标点会污染 term）
_BM25_TOKEN_INVALID_CHARS = set(" \t\n():\"'{}[]^~*?\\/&|!+-")


def _tokenize_for_bm25(text: str) -> List[str]:
    """jieba 切分 + 过滤空/含标点 token，供 PG BM25 查询串使用。"""
    tokens = []
    for tok in jieba.cut(text):
        tok = tok.strip()
        if not tok:
            continue
        if any(ch in _BM25_TOKEN_INVALID_CHARS for ch in tok):
            continue
        tokens.append(tok)
    return tokens


def bm25_search(query: str, top_k: int = VECTOR_RECALL_TOP_K) -> List[Tuple[str, float]]:
    """PG BM25 检索：jieba 切分查询 → paradedb.parse → @@@ 匹配 content_tokens。

    返回: [(chunk_id_str, score), ...] 按 BM25 分数降序。
    """
    _ensure_custom_dict()
    tokens = _tokenize_for_bm25(query)
    if not tokens:
        return []
    query_str = "content_tokens:(" + " ".join(tokens) + ")"
    sql = (
        "SELECT id, paradedb.score(id) FROM documents "
        "WHERE id @@@ paradedb.parse(%s) "
        "ORDER BY paradedb.score(id) DESC LIMIT %s"
    )
    try:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (query_str, top_k))
                rows = cur.fetchall()
        finally:
            conn.close()
        return [(str(rid), float(score)) for rid, score in rows]
    except psycopg2.Error as e:
        global _last_pg_error
        _last_pg_error = True  # 供 hybrid_search 判定 PG 运行时故障（触发熔断）
        logger.error("[RAG] bm25_search 失败：%s", e)
        return []


# ---------- 4. 向量检索（ParadeDB pgvector, 余弦） ----------
def _embed_texts(texts: List[str]) -> List[List[float]]:
    """批量文本向量化（fastembed）。懒加载 _embedding_model。"""
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return [list(map(float, v)) for v in _embedding_model.embed(texts)]


def _vec_literal(vec: List[float]) -> str:
    """将 float 列表序列化为 pgvector 字面量 '[v1,v2,...]'。"""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def vector_search(query: str, top_k: int = VECTOR_RECALL_TOP_K) -> List[Tuple[str, float]]:
    """PG 向量检索：query 向量化 → HNSW 余弦近邻（embedding <=> q）。

    返回: [(chunk_id_str, cosine_sim), ...] 按相似度降序（sim = 1 - 余弦距离）。
    """
    try:
        q_vec = _embed_texts([query])[0]
    except Exception as e:
        logger.error("[RAG] vector_search 查询向量化失败：%s", e)
        return []
    literal = _vec_literal(q_vec)
    sql = (
        "SELECT id, 1 - (embedding <=> %s::vector) AS sim FROM documents "
        "ORDER BY embedding <=> %s::vector LIMIT %s"
    )
    try:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (literal, literal, top_k))
                rows = cur.fetchall()
        finally:
            conn.close()
        return [(str(rid), float(sim)) for rid, sim in rows]
    except psycopg2.Error as e:
        global _last_pg_error
        _last_pg_error = True  # 供 hybrid_search 判定 PG 运行时故障（触发熔断）
        logger.error("[RAG] vector_search 失败：%s", e)
        return []


# ---------- 4.5 入库（一次性灌库，tools/ingest_to_pg.py 调用） ----------
def ingest_documents(chunks: List[Dict], embeddings: List[List[float]]) -> Dict:
    """批量写入 documents 表（ON CONFLICT DO NOTHING 保证幂等）。

    chunks: [{"id":int, "content":str, "metadata":dict, "content_tokens":str, "source":str}, ...]
    embeddings: 与 chunks 等长的向量列表（已算好，dim 必须=512）。
    返回: {"inserted":n, "skipped":m, "total":t}
    """
    if not chunks:
        return {"inserted": 0, "skipped": 0, "total": 0}
    if len(chunks) != len(embeddings):
        raise ValueError(f"chunks({len(chunks)}) 与 embeddings({len(embeddings)}) 数量不一致")
    values = [
        (int(c["id"]), c["content"], Json(c.get("metadata", {})),
         c.get("content_tokens", ""), c.get("source", ""), _vec_literal(embeddings[i]))
        for i, c in enumerate(chunks)
    ]
    sql = (
        "INSERT INTO documents (id, content, metadata, content_tokens, source, embedding) "
        "VALUES %s ON CONFLICT (id) DO NOTHING"
    )
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                # 单批提交：execute_values 默认 page_size=100 会拆成多条 INSERT，
                # 而 cur.rowcount 只反映最后一条 → inserted 计数失真（实测 128 条误报 28）。
                # 本表数据量为百级，一次性提交使 rowcount = 真实插入行数，幂等重跑报告才准确。
                execute_values(cur, sql, values,
                               template="(%s, %s, %s, %s, %s, %s::vector)",
                               page_size=len(values))
                inserted = cur.rowcount
        total = len(chunks)
        return {"inserted": inserted, "skipped": total - inserted, "total": total}
    finally:
        conn.close()


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
        if chunk_ins_type == insurance_type or chunk_ins_type == GENERIC_INSURANCE_TYPE:
            filtered.append((chunk_id, score))
    
    return filtered


# ---------- 6. RRF 融合 ----------
# RRF 平滑常数（标准值 60：越大越弱化头部排名的优势）
RRF_K = 60


def rrf_fuse(
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    top_k: int = 10,
    k: int = RRF_K
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
    PG 连接/查询异常时抛 RAGRetrievalError（供 search_terms 触发熔断降级）。
    """
    global _last_pg_error
    _last_pg_error = False  # 重置本轮 PG 故障标志（由 vector/bm25 的 except psycopg2.Error 置位）
    from src.query_expander import expand_query
    
    # 1. 险种识别
    insurance_type = detect_insurance_type(query)
    
    # 2. 查询替换
    expanded_query = replace_insurance_abbreviation(query, insurance_type)
    
    # 3. 查询扩展（仅用于 BM25，向量检索用原始 query 避免语义污染）
    _t = time.time()
    bm25_query = expand_query(expanded_query, insurance_type)
    logger.info("[RAG-计时] ①查询扩展: %.3fs | 原始=%s | 扩展后=%s",
                time.time() - _t, query, bm25_query)
    
    # 4. 双路检索
    _t = time.time()
    vector_results = vector_search(expanded_query, top_k=50)
    logger.info("[RAG-计时] ②向量检索: %.3fs | 召回%d条", time.time() - _t, len(vector_results))

    _t = time.time()
    bm25_results = bm25_search(bm25_query, top_k=50)
    logger.info("[RAG-计时] ③BM25检索: %.3fs | 召回%d条", time.time() - _t, len(bm25_results))

    # 4.5 PG 故障检测：vector/bm25 内部捕获 psycopg2.Error 会置 _last_pg_error=True，
    #     据此区分“PG 异常”与“正常无结果”（后者是合法业务空返回，不触发熔断）。
    if _last_pg_error:
        raise RAGRetrievalError("PG 检索异常（vector/bm25 连接或查询失败）")
    
    # 5. Metadata 过滤
    _t = time.time()
    vector_filtered = filter_by_insurance_type(vector_results, insurance_type)
    bm25_filtered = filter_by_insurance_type(bm25_results, insurance_type)
    logger.info("[RAG-计时] ④结果过滤: %.3fs | 向量%d→%d | BM25 %d→%d",
                time.time() - _t, len(vector_results), len(vector_filtered),
                len(bm25_results), len(bm25_filtered))
    
    # 6. RRF 融合
    _t = time.time()
    final_results = rrf_fuse(vector_filtered, bm25_filtered, top_k=top_k)
    logger.info("[RAG-计时] ⑤RRF融合: %.3fs | 融合后%d条", time.time() - _t, len(final_results))
    
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