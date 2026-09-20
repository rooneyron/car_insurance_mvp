# -*- coding: utf-8 -*-
"""RAG 检索质量评测：用 data/RAG评估测试集.json（30 条，含 relevant_chunk_ids ground truth）
评估迁移到 ParadeDB(pgvector+pg_search) 后的生产检索链路质量。

设计遵循项目规范「评估脚本直接复用生产代码，不自实现检索/精排逻辑」：
  - rag.search_terms(q, top_k=3)      ：完整生产链路（险种识别 + PG向量 + PG BM25 + Metadata过滤 + RRF + 精排）
  - rag.get_last_rag_pipeline_stats() ：取 final_returned_ids（与 chains.py 工具实际所见完全一致）

指标（top_k=3，与生产 search_terms 一致）：
  - Hit@3       : 返回的 top3 中命中任一相关 chunk 的 query 比例
  - MRR         : 首个相关 chunk 排名倒数的均值
  - Precision@3 : |返回 ∩ 相关| / |返回|（返回非空时）
  - Recall@3    : |返回 ∩ 相关| / |相关|
  - Empty_Rate  : 返回 RAG_EMPTY_RESULT 的比例

数据集 30 条全部有非空 relevant_chunk_ids（无「期望空结果」用例），故空返回一律计为 MISS。
运行：python eval_rag_retrieval.py
产物：eval_rag_retrieval_result.txt（utf-8；eval_*.txt 已 gitignore，本地用）
"""
import os
import sys
import json
import logging

# 持久化 fastembed 缓存，避免重复下载模型；本地 CrossEncoder 精排（阈值 0.6），可复现、零 API 成本
os.environ.setdefault("FASTEMBED_CACHE_PATH", os.path.join(os.path.expanduser("~"), ".cache", "fastembed"))
os.environ.setdefault("USE_LOCAL_RERANK", "true")

# 防止 Windows 控制台 GBK 编码导致 print 崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.logger import setup_logging
setup_logging()
# 抑制 RAG 每请求 INFO 日志，保持评测输出干净（ERROR/WARNING 仍保留）
logging.getLogger("src.rag").setLevel(logging.WARNING)

from src.constants import RAG_EMPTY_RESULT
import src.rag as rag

DATASET = os.path.join("data", "RAG评估测试集.json")
OUT_TXT = "eval_rag_retrieval_result.txt"
TOP_K = 3


def _metrics(ret_ids, rel_ids):
    """ret_ids: 生产返回的有序 chunk_id(str) 列表；rel_ids: ground-truth set(str)。
    返回 (hit, reciprocal_rank, precision, recall, n_intersection)。
    """
    ret = list(ret_ids)
    inter = [c for c in ret if c in rel_ids]
    hit = 1 if inter else 0
    rr = 0.0
    for i, c in enumerate(ret):
        if c in rel_ids:
            rr = 1.0 / (i + 1)
            break
    precision = len(inter) / len(ret) if ret else 0.0
    recall = len(inter) / len(rel_ids) if rel_ids else 0.0
    return hit, rr, precision, recall, len(inter)


def main():
    with open(DATASET, encoding="utf-8") as f:
        cases = json.load(f)

    # 一次性预加载全部生产组件（jieba / PG 预检 / Embedding / CrossEncoder / 同义词）
    rag.init_rag_components()

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 84)
    out("RAG 检索质量评测（迁移后 ParadeDB 后端）")
    out(f"  数据集       : {DATASET}")
    out(f"  用例数       : {len(cases)}")
    out(f"  top_k        : {TOP_K}（与生产 search_terms 一致）")
    out(f"  USE_LOCAL_RERANK : {os.environ.get('USE_LOCAL_RERANK')}")
    out(f"  PG 连通(_pg_ready): {rag._pg_ready}")
    out("=" * 84)

    if not rag._pg_ready:
        out("!!! 警告：PG 未连通，检索将全部返回空。请先启动 Docker 里的 ParadeDB 再评测。")

    per_cat = {}
    tot_hit = tot_empty = 0
    tot_rr = tot_prec = tot_rec = 0.0
    miss_cases = []

    for idx, case in enumerate(cases, 1):
        q = case["query"]
        cat = case.get("category", "未分类")
        rel_ids = set(str(x) for x in case.get("relevant_chunk_ids", []))
        expected = case.get("expected_article", "")

        final = rag.search_terms(q, top_k=TOP_K)
        stats = rag.get_last_rag_pipeline_stats()
        ret_ids = [str(x) for x in stats.get("final_returned_ids", [])]
        empty = bool(stats.get("empty_result")) or final == [RAG_EMPTY_RESULT]

        hit, rr, prec, rec, n_inter = _metrics(ret_ids, rel_ids)

        tot_hit += hit
        tot_rr += rr
        tot_prec += prec
        tot_rec += rec
        if empty:
            tot_empty += 1

        d = per_cat.setdefault(cat, {"n": 0, "hit": 0, "rr": 0.0, "prec": 0.0, "rec": 0.0, "empty": 0})
        d["n"] += 1
        d["hit"] += hit
        d["rr"] += rr
        d["prec"] += prec
        d["rec"] += rec
        if empty:
            d["empty"] += 1

        flag = "HIT  " if hit else ("EMPTY" if empty else "MISS ")
        if not hit:
            miss_cases.append((idx, q, cat, sorted(rel_ids, key=lambda x: int(x)), ret_ids, expected))

        out(f"\n[#{idx:02d}][{cat}] {flag} | {q}")
        out(f"    相关id={sorted(rel_ids, key=lambda x: int(x))}  返回id={ret_ids}  命中={n_inter}  RR={rr:.3f} P={prec:.2f} R={rec:.2f}")

    n = len(cases)
    out("\n" + "=" * 84)
    out("总体指标")
    out("=" * 84)
    out(f"  用例数         : {n}")
    out(f"  Hit@{TOP_K}        : {tot_hit}/{n} = {tot_hit / n:.1%}")
    out(f"  MRR            : {tot_rr / n:.3f}")
    out(f"  Precision@{TOP_K}  : {tot_prec / n:.3f}")
    out(f"  Recall@{TOP_K}     : {tot_rec / n:.3f}")
    out(f"  Empty_Rate     : {tot_empty}/{n} = {tot_empty / n:.1%}")

    out("\n" + "=" * 84)
    out("按 category 分组")
    out("=" * 84)
    out(f"  {'类别':<8}{'n':>4}{'Hit@3':>9}{'MRR':>8}{'Prec':>8}{'Rec':>8}{'Empty':>8}")
    for cat, d in per_cat.items():
        cn = d["n"]
        out(f"  {cat:<8}{cn:>4}{d['hit'] / cn:>8.1%}{d['rr'] / cn:>8.3f}{d['prec'] / cn:>8.3f}{d['rec'] / cn:>8.3f}{d['empty'] / cn:>8.1%}")

    if miss_cases:
        out("\n" + "=" * 84)
        out(f"未命中用例明细（{len(miss_cases)} 条，供诊断）")
        out("=" * 84)
        for idx, q, cat, rel, ret, expected in miss_cases:
            out(f"  [#{idx:02d}][{cat}] {q}")
            out(f"      相关id={rel}  返回id={ret}")
            if expected:
                out(f"      期望条款: {expected}")
    else:
        out("\n全部用例命中（Hit@3 = 100%）。")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n结果已写入: {OUT_TXT}")


if __name__ == "__main__":
    main()
