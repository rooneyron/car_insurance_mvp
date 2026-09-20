"""
路由全链路评测脚本

对 data/intent_eval_dataset.json 的每条 query 调用生产路由主函数 route_message
（跑完整 L0 → DST → L1 → L2 → L3 → L4，而非只测 L2/L3），把最终 intent 与标注的
allowed_routes 比对，结果以 JSON 明细写入 tests/route_eval_result.json。

用法（务必从项目根目录执行；脚本已 os.chdir 兜底）：
    python route_eval/eval_route.py             # 全量 84 条
    python route_eval/eval_route.py --limit 3   # 冒烟抽样（省钱先验证通路）
    python route_eval/eval_route.py --verbose   # 额外打印路由层 INFO 日志
"""

import os
import sys

# ---------- 启动引导：必须在 import src.* 之前把项目根加入 sys.path 并 chdir ----------
# l0_safety / l1_keyword 在“模块导入时”就用相对路径 config/config.yaml 读关键词，
# 若不 chdir 到项目根，os.path.exists("config/config.yaml") 会落空 → 关键词静默为空
# → L0/L1 全失效、所有 query 都落到 L2，评测结果失真。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import json
import time
import argparse

from dotenv import load_dotenv

load_dotenv()  # 与 app.py 同款：从 .env 载入 DEEPSEEK_API_KEY / ROUTER_CLASSIFIER_* 等

from src.logger import setup_logging
from src.constants import SOURCE_L0_SAFETY, SOURCE_L4_CLARIFY, SOURCE_L4_HANDOFF

DATASET_PATH = os.path.join("data", "intent_eval_dataset.json")
OUTPUT_PATH = os.path.join("tests", "route_eval_result.json")


# ============================================================
# CountingLLM：包裹真实 LLM，统计 invoke 次数
# L2(l2_classifier.classify) 与 L3(l3_reviewer.review) 内部都走 llm.invoke(prompt)，
# 故包一层即可分别统计两层的真实调用次数。
# ============================================================
class CountingLLM:
    def __init__(self, real):
        self._real = real
        self.count = 0

    def invoke(self, *args, **kwargs):
        self.count += 1
        return self._real.invoke(*args, **kwargs)


def build_llms():
    """忠实复刻 src/chains/chains.py L829-856 的 classifier / reviewer 构造。"""
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("缺少环境变量 DEEPSEEK_API_KEY（检查 .env 是否加载）")
    model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

    classifier_model = os.environ.get("ROUTER_CLASSIFIER_MODEL", model_name)
    reviewer_model = os.environ.get("ROUTER_REVIEWER_MODEL", model_name)
    classifier_base_url = os.environ.get(
        "ROUTER_CLASSIFIER_BASE_URL", "https://api.deepseek.com/v1"
    )

    llm_classifier = ChatOpenAI(
        model=classifier_model,
        api_key=os.environ.get("ROUTER_CLASSIFIER_API_KEY", api_key),
        base_url=classifier_base_url,
        temperature=0.1,
    )
    llm_reviewer = ChatOpenAI(
        model=reviewer_model,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )
    meta = {
        "classifier_model": classifier_model,
        "classifier_base_url": classifier_base_url,
        "reviewer_model": reviewer_model,
    }
    return llm_classifier, llm_reviewer, meta


def sanity_check():
    """防呆：确认 L0/L1 关键词已从 config.yaml 正确加载（cwd 不对会静默为空）。"""
    from src.router.l0_safety import check_safety
    from src.router.l1_keyword import _keyword_config

    if check_safety("转人工") != "handoff":
        raise SystemExit("健全性检查失败：L0 安全关键词未加载（是否未在项目根目录运行？）")
    if not _keyword_config:
        raise SystemExit("健全性检查失败：L1 关键词配置为空（config/config.yaml 未读到）")


def classify_mismatch(source):
    """strict 失配时按决策来源归类（复刻 doc/router_intent_test_report.md 三类分析）。"""
    if source == SOURCE_L0_SAFETY:
        return "l0_handoff"      # 转人工，报告视为生产正确
    if source in (SOURCE_L4_CLARIFY, SOURCE_L4_HANDOFF):
        return "l4_clarify"      # 模糊输入反问，报告视为合理行为
    return "real_error"          # l1/l2/l3 判成业务意图但不在 allowed_routes → 真实判错


def run_eval(limit=None):
    from src.router.router import route_message, RouterConfig
    from src.router.schemas import RouterState

    if not os.path.exists(DATASET_PATH):
        raise SystemExit(f"数据集不存在: {DATASET_PATH}（是否未在项目根目录运行？）")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if limit:
        dataset = dataset[:limit]

    llm_classifier, llm_reviewer, llm_meta = build_llms()
    cnt_l2 = CountingLLM(llm_classifier)
    cnt_l3 = CountingLLM(llm_reviewer)
    cfg = RouterConfig()  # 默认阈值 l2_accept=0.7 / l2_review=0.5 / l3_accept=0.6

    records = []
    total = len(dataset)
    t0 = time.time()
    for i, item in enumerate(dataset, 1):
        query = item["query"]
        allowed = item.get("allowed_routes", [])
        err = None
        try:
            rs = RouterState()  # 全新状态：current_task=None → DST 不触发（单轮隔离）
            r = route_message(query, rs, cnt_l2, cnt_l3, cfg)
            intent, source, action = r.intent, r.source, r.action
            confidence, sentiment = r.confidence, r.sentiment
        except Exception as e:  # 单条兜底，防崩溃中断全量
            intent, source, action = "error", "error", "error"
            confidence, sentiment = 0.0, "neutral"
            err = f"{type(e).__name__}: {e}"

        strict = intent in allowed
        mismatch_type = None if strict else classify_mismatch(source)
        records.append({
            "query": query,
            "category": item.get("category"),
            "allowed_routes": allowed,
            "required_tools": item.get("required_tools", []),
            "intent": intent,
            "source": source,
            "action": action,
            "confidence": round(confidence, 4) if isinstance(confidence, float) else confidence,
            "sentiment": sentiment,
            "strict_correct": strict,
            "mismatch_type": mismatch_type,
            "error": err,
        })
        if i % 10 == 0 or i == total:
            print(f"  进度 {i}/{total} ...")

    elapsed = time.time() - t0
    return records, cnt_l2.count, cnt_l3.count, cfg, llm_meta, total, elapsed


def aggregate(records, l2_calls, l3_calls, cfg, llm_meta, total, elapsed):
    strict_match = sum(1 for r in records if r["strict_correct"])

    mismatch_cls = {"l0_handoff": 0, "l4_clarify": 0, "real_error": 0}
    for r in records:
        if r["mismatch_type"]:
            mismatch_cls[r["mismatch_type"]] += 1
    # 生产有效匹配：严格匹配 + L0 转人工（生产正确）+ L4 澄清（模糊输入合理行为）
    effective_match = strict_match + mismatch_cls["l0_handoff"] + mismatch_cls["l4_clarify"]

    source_dist = {}
    for r in records:
        source_dist[r["source"]] = source_dist.get(r["source"], 0) + 1

    cat_break = {}
    for r in records:
        c = r["category"]
        slot = cat_break.setdefault(c, {"total": 0, "strict_match": 0})
        slot["total"] += 1
        if r["strict_correct"]:
            slot["strict_match"] += 1

    def rate(n):
        return round(n / total, 4) if total else 0.0

    return {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_path": DATASET_PATH.replace(os.sep, "/"),
            "dataset_size": total,
            "entry": "src/router/router.py::route_message",
            "classifier_model": llm_meta["classifier_model"],
            "classifier_base_url": llm_meta["classifier_base_url"],
            "reviewer_model": llm_meta["reviewer_model"],
            "router_config": {
                "l2_accept_threshold": cfg.l2_accept_threshold,
                "l2_review_threshold": cfg.l2_review_threshold,
                "l3_accept_threshold": cfg.l3_accept_threshold,
            },
            "isolation": "fresh RouterState per query (DST skipped)",
            "elapsed_sec": round(elapsed, 1),
        },
        "summary": {
            "total": total,
            "strict_match": strict_match,
            "strict_match_rate": rate(strict_match),
            "effective_match": effective_match,
            "effective_match_rate": rate(effective_match),
            "source_distribution": source_dist,
            "llm_calls": {"l2_classifier": l2_calls, "l3_reviewer": l3_calls},
            "category_breakdown": cat_break,
            "mismatch_classification": mismatch_cls,
        },
        "records": records,
    }


def print_summary(result):
    s = result["summary"]
    line = "=" * 60
    print("\n" + line)
    print("路由全链路评测结果")
    print(line)
    print(f"样本总数        : {s['total']}")
    print(f"严格匹配        : {s['strict_match']}/{s['total']} = {s['strict_match_rate'] * 100:.1f}%")
    print(f"生产有效匹配    : {s['effective_match']}/{s['total']} = {s['effective_match_rate'] * 100:.1f}%")
    print(f"LLM 调用        : L2={s['llm_calls']['l2_classifier']}  L3={s['llm_calls']['l3_reviewer']}")
    print(f"决策来源分布    : {s['source_distribution']}")
    print(f"失配分类        : {s['mismatch_classification']}")
    print("-" * 60)
    print("分类别严格匹配  :")
    for c, v in s["category_breakdown"].items():
        print(f"    {c:<18} {v['strict_match']}/{v['total']}")

    real_errors = [r for r in result["records"] if r["mismatch_type"] == "real_error"]
    if real_errors:
        print("-" * 60)
        print(f"真实判错（{len(real_errors)} 条）:")
        for r in real_errors:
            print(f"    [{r['category']}] {r['query']}  →  intent={r['intent']} "
                  f"(应属 {r['allowed_routes']}) src={r['source']} conf={r['confidence']}")

    errs = [r for r in result["records"] if r.get("error")]
    if errs:
        print("-" * 60)
        print(f"调用异常（{len(errs)} 条）:")
        for r in errs:
            print(f"    {r['query']}  →  {r['error']}")
    print(line)


def main():
    parser = argparse.ArgumentParser(description="路由全链路评测（route_message L0→L4）")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条（冒烟抽样，省成本）")
    parser.add_argument("--verbose", action="store_true", help="打印路由层 INFO 日志（默认只 WARNING+）")
    args = parser.parse_args()

    setup_logging("INFO" if args.verbose else "WARNING")
    sanity_check()

    scope = f"前 {args.limit} 条" if args.limit else "全量"
    print(f"数据集: {DATASET_PATH}")
    print(f"开始评测（{scope}）...")

    records, l2c, l3c, cfg, meta, total, elapsed = run_eval(args.limit)
    result = aggregate(records, l2c, l3c, cfg, meta, total, elapsed)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print_summary(result)
    print(f"\n结果已写入: {OUTPUT_PATH}  （耗时 {elapsed:.1f}s）")


if __name__ == "__main__":
    main()
