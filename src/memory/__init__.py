# -*- coding: utf-8 -*-
"""长期记忆模块：结构化(sqlite) + 语义(PG 向量) 双存储，及对话后异步提取触发。

对外暴露：
  - structured_store: SqliteStructuredStore 单例（用户 5 字段画像）
  - semantic_store:   PgSemanticStore 单例（用户持久事实短句 + 向量）
  - spawn_memory_extraction(user_id, user_msg, ai_reply): 起 daemon 线程异步提取 + 落库

设计要点：
  - 单例在模块级创建：structured_store 实例化即建 sqlite 表；semantic_store 方法内懒连接、无副作用。
  - spawn 起的后台线程全链路 try/except，任何失败只记日志，绝不影响对话主流程。
  - _run_memory_extraction 内部懒 import extractor / rag，规避 chat.py -> memory -> chains 的循环依赖。
"""
import threading
from typing import List

from src.logger import get_logger
from src.memory.structured_store import SqliteStructuredStore
from src.memory.semantic_store import PgSemanticStore

logger = get_logger(__name__)

# ---------- 全局单例（避免重复初始化）----------
structured_store = SqliteStructuredStore()   # 实例化即建 data/user_memory.db 的 user_memory 表
semantic_store = PgSemanticStore()           # 无副作用，各方法内部各自 get_conn()

# 语义记忆：去重相似度阈值 / 单用户条数上限
_SIMILAR_THRESHOLD = 0.85
_MAX_SEMANTIC_PER_USER = 20


def spawn_memory_extraction(user_id: str, user_msg: str, ai_reply: str = "") -> None:
    """对话回复后触发：起 daemon 后台线程做记忆提取 + 存储，不阻塞主流程。

    user_id 为空（未登录）时直接返回，不做任何事。
    """
    if not user_id:
        return
    try:
        t = threading.Thread(
            target=_run_memory_extraction,
            args=(user_id, user_msg, ai_reply),
            daemon=True,
        )
        t.start()
        logger.debug("[记忆] 已触发后台提取线程 user_id=%s", user_id)
    except Exception as e:
        # 起线程本身失败也不能影响主流程
        logger.error("[记忆] 启动提取线程失败 user_id=%s: %s", user_id, e)


def _run_memory_extraction(user_id: str, user_msg: str, ai_reply: str) -> None:
    """后台线程主体：提取 -> 结构化 upsert -> 语义去重/数量控制/写入。

    每一步独立 try/except，失败记 logger.error 后继续下一步；整体再兜一层，绝不抛出。
    """
    try:
        # 懒 import：extractor 依赖 chains(LLM)，rag 提供 embedding，均此刻才需要
        from src.memory.extractor import extract_memory_from_conversation
        from src.rag import _embed_texts

        result = extract_memory_from_conversation(user_id, user_msg, ai_reply)
        structured = result.get("structured") or {}
        semantic: List[str] = result.get("semantic") or []

        # ---------- 1) 结构化：仅当有非 None 字段才 upsert ----------
        try:
            if any(v is not None for v in structured.values()):
                structured_store.upsert(user_id, structured)
        except Exception as e:
            logger.error("[记忆] 结构化 upsert 失败 user_id=%s: %s", user_id, e)

        # ---------- 2) 语义：逐条 embedding -> 去重 -> 数量控制 -> 写入 ----------
        for content in semantic:
            try:
                content = (content or "").strip()
                if not content:
                    continue
                # 2a) 复用 rag 的 fastembed 模型（懒加载，不重复加载）
                emb = _embed_texts([content])[0]

                # 2b) 去重：相似度 >= 阈值视为同一条
                similar = semantic_store.find_similar(user_id, emb, threshold=_SIMILAR_THRESHOLD)
                if similar:
                    top = similar[0]
                    old = (top.get("content") or "").strip()
                    # 「新信息更丰富」= 新内容更长 -> 覆盖旧条目；否则跳过（去重生效）
                    if len(content) > len(old):
                        semantic_store.update(top["id"], content, emb)
                        logger.info("[记忆-语义] 去重更新 id=%s user_id=%s", top.get("id"), user_id)
                    else:
                        logger.debug("[记忆-语义] 去重跳过（已有相似）user_id=%s", user_id)
                    continue

                # 2c) 数量控制：达到上限先删最旧 1 条
                if semantic_store.count(user_id) >= _MAX_SEMANTIC_PER_USER:
                    semantic_store.delete_oldest(user_id, 1)

                # 2d) 写入
                semantic_store.add(user_id, content, emb)
            except Exception as e:
                logger.error("[记忆-语义] 处理失败 user_id=%s content=%s: %s",
                             user_id, (content or "")[:20], e)
                continue
    except Exception as e:
        logger.error("[记忆] 后台提取异常 user_id=%s: %s", user_id, e)
