# -*- coding: utf-8 -*-
"""一次性灌库脚本：data/chunk_metadata.json → ParadeDB documents 表。

deploy/aliyun 分支 RAG 存储层迁移（FAISS+rank_bm25 → pgvector+pg_search）用。

可复现设计：
  - 数据源为 tracked 的 data/chunk_metadata.json（128 条 {id:{content,metadata}}），
    不依赖未跟踪的 chunks.pkl / faiss_index.bin。
  - 两类检索文本分别复现旧管线口径：
      * embedding      ← full_text  = "{section} {article_no}\\n{content}"（同 scripts/build_faiss_index.build_full_text）
      * content_tokens ← enriched   = "【insurance_type】【section】content" 经 jieba 切分后空格拼接
    enriched 的切分复用 rag._tokenize_for_bm25（与查询侧 bm25_search 同一函数），
    保证 doc/query 词表对称——这是 pg_search whitespace 分词下 BM25 正确匹配的前提。
  - embedding 走 fastembed（Qdrant/bge-small-zh-v1.5 ONNX 变体，实测 dim=512）；
    首次运行需联网下载模型，之后命中缓存。FASTEMBED_CACHE_PATH 默认指向持久的
    ~/.cache/fastembed（避免 %TEMP% 被清理后反复重下），可用环境变量覆盖。

幂等：rag.ingest_documents 用 ON CONFLICT (id) DO NOTHING，重跑第二次 inserted=0 / skipped=128。
本脚本自包含，不 import scripts.*（scripts/ 被 .gitignore 忽略，无法随仓库分发）。

运行：python -m tools.ingest_to_pg
"""
import json
import os
import sys

# fastembed 持久缓存：setdefault 尊重已有环境变量；指向已存在模型的 ~/.cache/fastembed，
# 避免默认 %TEMP%\fastembed_cache 被系统清理后触发重下（需联网/VPN）。
os.environ.setdefault(
    "FASTEMBED_CACHE_PATH",
    os.path.join(os.path.expanduser("~"), ".cache", "fastembed"),
)

import jieba

from src import rag
from src.db import init_db
from src.constants import EMBEDDING_DIM

CHUNK_METADATA_PATH = "data/chunk_metadata.json"


def build_full_text(content: str, metadata: dict) -> str:
    """复现 scripts/build_faiss_index.py 的 build_full_text（向量检索全文）。

    格式：有 section/article_no 时为 "{section} {article_no}\\n{content}"，否则仅 content。
    """
    section = metadata.get("section", "")
    article_no = metadata.get("article_no", "")
    header_parts = []
    if section:
        header_parts.append(section)
    if article_no:
        header_parts.append(article_no)
    if header_parts:
        return f"{' '.join(header_parts)}\n{content}"
    return content


def main() -> int:
    # 1. 读取 tracked chunk 源
    if not os.path.exists(CHUNK_METADATA_PATH):
        print(f"[ingest] 错误：{CHUNK_METADATA_PATH} 不存在", file=sys.stderr)
        return 1
    with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"[ingest] 读取 {CHUNK_METADATA_PATH}: {len(raw)} 条")

    # 2. jieba 自定义词典（与查询侧同一套术语，保证切分口径统一）
    rag._ensure_custom_dict()

    # 3. 组装 chunks（按 id 数值升序，稳定可复现）
    ids_sorted = sorted(raw.keys(), key=int)
    chunks = []
    full_texts = []
    for key in ids_sorted:
        entry = raw[key]
        content = entry["content"]
        metadata = entry.get("metadata", {})
        full_text = build_full_text(content, metadata)
        enriched = rag._build_enriched_text(content, metadata)
        content_tokens = " ".join(rag._tokenize_for_bm25(enriched))
        chunks.append({
            "id": int(key),
            "content": content,
            "metadata": metadata,
            "content_tokens": content_tokens,
            "source": metadata.get("doc_type", ""),
        })
        full_texts.append(full_text)

    # 抽查：打印首条组装结果，人工核对 full_text / content_tokens 口径
    c0 = chunks[0]
    print(f"[ingest] 抽查 id={c0['id']} source={c0['source']!r}")
    print(f"[ingest]   full_text[0]  = {full_texts[0][:60]!r} ...")
    print(f"[ingest]   tokens[0]     = {c0['content_tokens'][:60]!r} ...")

    # 4. 批量向量化（fastembed），断言维度=512 与 vector(512) 列对齐
    print(f"[ingest] 向量化 {len(full_texts)} 条 full_text ...")
    embeddings = rag._embed_texts(full_texts)
    if len(embeddings) != len(chunks):
        print(f"[ingest] 错误：向量数 {len(embeddings)} != chunk 数 {len(chunks)}", file=sys.stderr)
        return 1
    dims = {len(v) for v in embeddings}
    if dims != {EMBEDDING_DIM}:
        print(f"[ingest] 错误：embedding 维度异常 {dims}（应={EMBEDDING_DIM}）", file=sys.stderr)
        return 1
    print(f"[ingest] 向量化完成，dim={EMBEDDING_DIM}")

    # 5. 建表（幂等）+ 灌库（ON CONFLICT DO NOTHING）
    init_db()
    result = rag.ingest_documents(chunks, embeddings)
    print(f"[ingest] 写入结果: {result}")

    # 6. 刷新 rag 元数据缓存（避免读到灌库前的空缓存）
    meta = rag.refresh_chunk_metadata_cache()
    print(f"[ingest] 元数据缓存刷新: {len(meta)} 条")
    print("[ingest] 完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
