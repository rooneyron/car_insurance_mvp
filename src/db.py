# -*- coding: utf-8 -*-
"""PostgreSQL (ParadeDB) 连接与 schema 初始化。

deploy/aliyun 分支：RAG 存储层从 FAISS + rank_bm25 迁移到 ParadeDB 的
pgvector(HNSW, vector(512)) + pg_search(BM25)。本模块提供：
  - get_conn(): 获取 psycopg2 连接（调用方负责 close；MVP 不做连接池）
  - init_db():  幂等建扩展 + 建表 + 建索引（不重灌数据）

连接串来源：环境变量 DATABASE_URL（本地 .env / 容器 environment 注入）。
本地默认 127.0.0.1:5432；容器内由 docker-compose 覆盖为 db:5432
（load_dotenv 默认 override=False，已存在的容器环境变量优先）。

运行建表：python -m src.db
"""
import os

import psycopg2
from dotenv import load_dotenv

from src.constants import EMBEDDING_DIM

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://insurance_user:insurance_pwd123@127.0.0.1:5432/insurance_db",
)


def get_conn():
    """获取一个 psycopg2 连接。调用方负责 close()。"""
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """幂等初始化 schema：扩展 + documents 表 + HNSW/BM25 两个索引。

    - 全部 IF NOT EXISTS，重复调用安全，不重灌数据。
    - embedding 维度 512（bge-small-zh-v1.5 实测，fastembed/Qdrant ONNX 变体）。
    - BM25 索引对 content_tokens 用 whitespace 分词器（按空格切＝精确还原 jieba 切分）。
    """
    ddl = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE EXTENSION IF NOT EXISTS pg_search",
        f"""
        CREATE TABLE IF NOT EXISTS documents (
            id             BIGINT PRIMARY KEY,
            content        TEXT NOT NULL,
            metadata       JSONB NOT NULL DEFAULT '{{}}',
            content_tokens TEXT NOT NULL DEFAULT '',
            source         TEXT NOT NULL DEFAULT '',
            embedding      vector({EMBEDDING_DIM}),
            created_at     TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents USING hnsw (embedding vector_cosine_ops)
        """,
        """
        CREATE INDEX IF NOT EXISTS documents_bm25_idx
            ON documents USING bm25 (id, content_tokens)
            WITH (key_field='id', text_fields='{"content_tokens":{"tokenizer":{"type":"whitespace"}}}')
        """,
    ]
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                for stmt in ddl:
                    cur.execute(stmt)
        print("[db] init_db 完成：documents 表 + HNSW(vector_cosine_ops) + BM25(whitespace) 索引就绪")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
