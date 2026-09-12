# -*- coding: utf-8 -*-
"""语义记忆存储（ParadeDB / pgvector）：用户持久事实的自然语言短句 + 向量。

设计要点：
  - 复用 src.db.get_conn()，不自建连接方式；embedding 由调用方算好传入（存储层不算向量）。
  - 向量字面量与 rag.py 一致（_vec_literal: '[v1,v2,...]'，SQL 里 %s::vector 传入）。
  - 余弦相似度 sim = 1 - (embedding <=> q)，与 rag.vector_search 完全一致。
  - user_id 为空快速返回；每个方法各自 try/except 记 log、返回安全默认值，绝不向上抛
    （保护调用它的后台线程不影响主流程）。
"""
from typing import List, Tuple, Dict, Optional

import psycopg2

from src.db import get_conn
from src.logger import get_logger

logger = get_logger(__name__)


def _vec_literal(vec: List[float]) -> str:
    """float 列表 → pgvector 字面量 '[v1,v2,...]'（与 rag.py._vec_literal 保持一致）。"""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class PgSemanticStore:
    """PG 语义记忆存储（semantic_memory 表）。所有方法线程安全（各自独立连接）。"""

    def add(self, user_id: str, content: str, embedding_list: List[float]) -> Optional[int]:
        """新增一条语义记忆，返回自增 id（失败返回 None）。"""
        if not user_id or not content:
            return None
        sql = (
            "INSERT INTO semantic_memory (user_id, content, embedding) "
            "VALUES (%s, %s, %s::vector) RETURNING id"
        )
        conn = get_conn()
        try:
            with conn:  # with conn 保证事务提交
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, content, _vec_literal(embedding_list)))
                    row = cur.fetchone()
            new_id = row[0] if row else None
            logger.debug("[记忆-语义] add user_id=%s id=%s", user_id, new_id)
            return new_id
        except psycopg2.Error as e:
            logger.error("[记忆-语义] add 失败 user_id=%s: %s", user_id, e)
            return None
        finally:
            conn.close()

    def search(self, user_id: str, query_embedding_list: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        """按 user_id + 余弦相似度检索，返回 [(content, sim), ...] 降序。"""
        if not user_id:
            return []
        literal = _vec_literal(query_embedding_list)
        sql = (
            "SELECT content, 1 - (embedding <=> %s::vector) AS sim FROM semantic_memory "
            "WHERE user_id = %s ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (literal, user_id, literal, top_k))
                rows = cur.fetchall()
            return [(content, float(sim)) for content, sim in rows]
        except psycopg2.Error as e:
            logger.error("[记忆-语义] search 失败 user_id=%s: %s", user_id, e)
            return []
        finally:
            conn.close()

    def find_similar(self, user_id: str, embedding_list: List[float], threshold: float = 0.85) -> List[Dict]:
        """查找相似度 >= threshold 的记忆（去重用），返回 [{id, content, score}, ...] 降序。"""
        if not user_id:
            return []
        literal = _vec_literal(embedding_list)
        # 用 1 - 余弦距离 >= 阈值 过滤；sim 别名不能直接进 WHERE，故重复表达式
        sql = (
            "SELECT id, content, 1 - (embedding <=> %s::vector) AS sim FROM semantic_memory "
            "WHERE user_id = %s AND 1 - (embedding <=> %s::vector) >= %s "
            "ORDER BY sim DESC"
        )
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (literal, user_id, literal, threshold))
                rows = cur.fetchall()
            return [{"id": rid, "content": content, "score": float(sim)} for rid, content, sim in rows]
        except psycopg2.Error as e:
            logger.error("[记忆-语义] find_similar 失败 user_id=%s: %s", user_id, e)
            return []
        finally:
            conn.close()

    def update(self, memory_id: int, content: str, embedding_list: List[float]) -> None:
        """按 id 更新内容和向量（去重时"新信息更丰富"覆盖旧条目）。"""
        # 用 is None 精确判空：memory_id 来自 BIGSERIAL(≥1)，避免 not 0 被误判跳过
        if memory_id is None or not content:
            return
        sql = "UPDATE semantic_memory SET content = %s, embedding = %s::vector WHERE id = %s"
        conn = get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (content, _vec_literal(embedding_list), memory_id))
            logger.debug("[记忆-语义] update id=%s", memory_id)
        except psycopg2.Error as e:
            logger.error("[记忆-语义] update 失败 id=%s: %s", memory_id, e)
        finally:
            conn.close()

    def count(self, user_id: str) -> int:
        """统计该用户的语义记忆条数（用于数量控制）。"""
        if not user_id:
            return 0
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM semantic_memory WHERE user_id = %s", (user_id,))
                return cur.fetchone()[0]
        except psycopg2.Error as e:
            logger.error("[记忆-语义] count 失败 user_id=%s: %s", user_id, e)
            return 0
        finally:
            conn.close()

    def delete_oldest(self, user_id: str, limit: int) -> None:
        """删除该用户最旧的 limit 条（按 created_at 升序）。"""
        if not user_id or limit <= 0:
            return
        sql = (
            "DELETE FROM semantic_memory WHERE id IN ("
            "SELECT id FROM semantic_memory WHERE user_id = %s ORDER BY created_at ASC LIMIT %s)"
        )
        conn = get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, limit))
            logger.debug("[记忆-语义] delete_oldest user_id=%s limit=%s", user_id, limit)
        except psycopg2.Error as e:
            logger.error("[记忆-语义] delete_oldest 失败 user_id=%s: %s", user_id, e)
        finally:
            conn.close()
