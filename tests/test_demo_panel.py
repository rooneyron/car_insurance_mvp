"""
演示面板自动化测试脚本
测试 12 个演示按钮对应的输入，覆盖：工具调用、短期/长期记忆、边界防护。

用法：
    python test_demo_panel.py              # 运行测试，结果写入 tests/demo_test_result.txt
    python test_demo_panel.py --verbose    # 同时打印到终端

设计要点：
    - 同一 session_id → 短期记忆（对话历史 MemorySaver）
    - 同一 user_id   → 长期记忆（结构化 sqlite + 语义 PG 向量）
    - 栏2 后半段用新 session_id 模拟 Gradio「清空对话」，验证长期记忆独立生效
"""

import os
import sys
import json
import time
import uuid
import argparse

# ---------- 项目根目录引导 ----------
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from dotenv import load_dotenv
load_dotenv()

from src.logger import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# ---------- 测试配置 ----------
TEST_USER_ID = "demo_test_user"
RESULT_FILE = os.path.join("tests", "demo_test_result.txt")
# 记忆提取等待秒数（spawn_memory_extraction 是 daemon 线程，需等待落库）
MEMORY_EXTRACT_WAIT = 3


# ============================================================
# 初始化（复刻 app.py 的预加载流程）
# ============================================================
def init_app():
    """初始化 graph + RAG + 记忆存储（与 app.py 一致）"""
    logger.info("正在初始化测试环境...")

    from src.chains.chains import init_graph
    from src.rag import init_rag_components
    from src.state import set_graph

    # StateGraph
    graph, llm, llm_classifier = init_graph()
    set_graph(graph)

    # RAG 组件
    try:
        init_rag_components()
    except Exception as e:
        logger.warning("RAG 预加载失败（不影响测试）: %s", e)

    # 数据库 + 记忆存储
    try:
        from src.db import init_db
        init_db()
    except Exception as e:
        logger.warning("init_db 失败: %s", e)

    try:
        from src.memory import structured_store  # noqa: F401
    except Exception as e:
        logger.warning("记忆存储初始化失败: %s", e)

    logger.info("测试环境初始化完成")
    return graph


# ============================================================
# 测试用例定义
# ============================================================
def build_test_cases():
    """
    返回测试用例列表，每个元素:
        (label, session_id, user_id, message, action)
    action:
        "send"  - 发送消息并记录回复
        "clear" - 模拟「清空对话」（仅打印分隔，下一条用新 session_id）
    """
    cases = []

    # ---- 栏 1：工具调用（6 个输入，同一 session_id）----
    s1 = f"test_tool_{uuid.uuid4().hex[:8]}"
    cases.append(("栏1-1 保费计算①", s1, "", "帮我算一下特斯拉的保费", "send"))
    cases.append(("栏1-2 保费计算②", s1, "", "37岁，驾龄5年", "send"))
    cases.append(("栏1-3 保单查询①", s1, "", "帮我查保单，保单号 POL20260001", "send"))
    cases.append(("栏1-4 保单查询②", s1, "", "110101199001011234", "send"))
    cases.append(("栏1-5 条款检索",   s1, "", "车损险保障哪些情况？", "send"))
    cases.append(("栏1-6 转人工",     s1, "", "我要投诉，转人工", "send"))

    # ---- 栏 2：记忆测试（3 个输入，需登录 user_id）----
    # 短期记忆：同一 session_id
    s2 = f"test_mem_{uuid.uuid4().hex[:8]}"
    cases.append(("栏2-1 告诉信息", s2, TEST_USER_ID, "我叫张三，身份证 110101199001011234，我的车去年出险了", "send"))
    cases.append(("栏2-2 短期记忆验证", s2, TEST_USER_ID, "我叫什么名字？", "send"))

    # 模拟「清空对话」：新 session_id，同 user_id（长期记忆）
    s2b = f"test_mem_cleared_{uuid.uuid4().hex[:8]}"
    cases.append(("--- 清空对话（新 session） ---", s2b, TEST_USER_ID, None, "clear"))
    cases.append(("栏2-3 长期记忆验证(结构化+语义)", s2b, TEST_USER_ID, "我叫什么名字？我出过险吗？", "send"))

    # ---- 栏 3：前端输入（1 个输入）----
    s3 = f"test_edge_{uuid.uuid4().hex[:8]}"
    cases.append(("栏3-1 模糊意图(L4)", s3, "", "你帮我看看", "send"))

    return cases


# ============================================================
# 预期结果定义（用于自动化对比）
# ============================================================
EXPECTED = {
    "栏1-1 保费计算①": {
        "route": "sale",
        "reply_contains": ["特斯拉"],
        "desc": "应路由 sale，调用 calculate_premium，因缺参数会反问",
    },
    "栏1-2 保费计算②": {
        "route": "sale",
        "reply_contains": ["保费", "元"],
        "desc": "DST 承接上轮，补全参数后应返回保费结果",
    },
    "栏1-3 保单查询①": {
        "route": "service",
        "reply_contains": ["保单", "身份证", "证件"],
        "desc": "应路由 service，调用 query_policy，因缺身份证会反问",
    },
    "栏1-4 保单查询②": {
        "route": "service",
        "reply_contains": ["保单"],
        "desc": "DST 承接上轮，补全身份证后应返回保单结果",
    },
    "栏1-5 条款检索": {
        "route": ["sale", "service"],
        "reply_contains": ["条款", "保险", "车损"],
        "desc": "应调用 search_insurance_terms（RAG 检索）",
    },
    "栏1-6 转人工": {
        "route": "general",  # L0 安全拦截 → intent=handoff → agent_type 回退 general
        "reply_contains": ["人工", "转接", "工单"],
        "desc": "L0 安全拦截 → 转人工短路直返",
    },
    "栏2-1 告诉信息": {
        "route": ["general", "service"],  # "出险" 可能触发 service 意图
        "reply_contains": ["张三"],
        "desc": "应确认用户身份和车辆信息，触发记忆提取",
    },
    "栏2-2 短期记忆验证": {
        "route": "general",
        "reply_contains": ["张三"],
        "desc": "应从对话历史回忆姓名=张三",
    },
    "栏2-3 长期记忆验证(结构化+语义)": {
        "route": ["general", "service"],  # "出过险" 可能触发 service 意图
        "reply_contains": ["张三", "出险", "出过险"],
        "desc": "清空对话后，应从结构化记忆(sqlite)回忆姓名=张三，从语义记忆(PG向量)回忆出险信息",
    },
    "栏3-1 模糊意图(L4)": {
        "route": "general",
        "reply_contains": [],
        "desc": "应触发 L4 澄清反问或路由到 general",
    },
}


def check_expected(label, reply, route, transfer):
    """对比实际结果与预期，返回 (pass: bool, issues: list[str])"""
    exp = EXPECTED.get(label)
    if not exp:
        return True, []

    issues = []
    reply_lower = (reply or "").lower()

    # 路由检查
    exp_route = exp.get("route")
    if exp_route:
        if isinstance(exp_route, list):
            if route not in exp_route:
                issues.append(f"路由: 期望 {exp_route} 之一, 实际 {route}")
        elif route != exp_route:
            issues.append(f"路由: 期望 {exp_route}, 实际 {route}")

    # 转人工检查
    exp_transfer = exp.get("transfer")
    if exp_transfer is not None and exp_transfer != transfer:
        issues.append(f"转人工: 期望 {exp_transfer}, 实际 {transfer}")

    # 回复内容检查
    reply_keywords = exp.get("reply_contains", [])
    if reply_keywords:
        found_any = any(kw.lower() in reply_lower for kw in reply_keywords)
        if not found_any:
            issues.append(f"回复关键词: 期望包含 {reply_keywords} 之一, 实际未找到")

    return len(issues) == 0, issues


# ============================================================
# 执行测试
# ============================================================
def run_tests(verbose=False):
    from src.chat import chat_api
    from src.memory import structured_store, semantic_store

    init_app()
    cases = build_test_cases()

    results = []
    summary = []  # (label, pass/fail, issues)
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"演示面板自动化测试  |  user_id={TEST_USER_ID}")
    print(f"{'='*60}\n")

    msg_idx = 0
    for i, (label, session_id, user_id, message, action) in enumerate(cases, 1):
        # ---- 清空对话分隔 ----
        if action == "clear":
            sep = f"\n{'─'*50}\n  [清空对话] session_id 已切换: {session_id}\n{'─'*50}\n"
            results.append(sep)
            if verbose:
                print(sep)
            continue

        msg_idx += 1
        # ---- 发送消息 ----
        print(f"  [{msg_idx:02d}] {label}  →  {message[:40]}...")
        t0 = time.time()
        try:
            resp = chat_api(session_id, message, user_id=user_id)
        except Exception as e:
            resp = {"success": -1, "content": {"reply": f"❌ 异常: {e}"}, "route": "error", "elapsed_ms": -1}
        elapsed = time.time() - t0

        reply = resp.get("content", {}).get("reply", "")
        route = resp.get("route", "?")
        transfer = resp.get("content", {}).get("transfer", False)
        api_ms = resp.get("elapsed_ms", -1)

        # ---- 对比预期 ----
        passed, issues = check_expected(label, reply, route, transfer)
        status = "✅" if passed else "❌"
        summary.append((label, passed, issues))

        # 格式化结果
        transfer_tag = " 🔄转人工" if transfer else ""
        exp_desc = EXPECTED.get(label, {}).get("desc", "")
        result_block = (
            f"[{msg_idx:02d}] {status} {label}\n"
            f"  预期: {exp_desc}\n"
            f"  输入: {message}\n"
            f"  路由: {route}{transfer_tag}\n"
            f"  耗时: {api_ms:.0f}ms (API) / {elapsed*1000:.0f}ms (含锁)\n"
            f"  回复: {reply[:300]}{'...' if len(reply) > 300 else ''}\n"
        )
        if issues:
            result_block += f"  ⚠️ 问题: {'; '.join(issues)}\n"
        results.append(result_block)

        if verbose:
            print(result_block)

        # ---- 栏2 消息后等待记忆提取 ----
        if label.startswith("栏2-") and action == "send":
            logger.info("等待记忆提取 (%ds)...", MEMORY_EXTRACT_WAIT)
            time.sleep(MEMORY_EXTRACT_WAIT)

    # ---- 最终验证：检查长期记忆是否落库 ----
    verify_block = "\n" + "="*60 + "\n长期记忆验证\n" + "="*60 + "\n"

    # 结构化记忆
    structured = structured_store.get(TEST_USER_ID)
    if structured:
        verify_block += f"  结构化记忆 (sqlite): {json.dumps(structured, ensure_ascii=False)}\n"
        # 验证关键字段
        name = structured.get("name", "")
        id_card = structured.get("id_card", "")
        if "张三" in str(name):
            verify_block += "    ✅ 姓名=张三\n"
        else:
            verify_block += f"    ❌ 姓名未匹配: {name}\n"
    else:
        verify_block += "  结构化记忆 (sqlite): ❌ 未找到\n"

    # 语义记忆
    try:
        semantic_rows = semantic_store.list_by_user(TEST_USER_ID)
        if semantic_rows:
            verify_block += f"  语义记忆 (PG向量): {len(semantic_rows)} 条\n"
            for row in semantic_rows:
                verify_block += f"    - {row.get('content', '')[:60]}\n"
            # 验证是否包含出险相关（宽松匹配："出险" 或 "出过险"）
            has_claim = any("出" in row.get("content", "") and "险" in row.get("content", "") for row in semantic_rows)
            if has_claim:
                verify_block += "    ✅ 包含出险相关记忆\n"
            else:
                verify_block += "    ❌ 未找到出险相关记忆\n"
        else:
            verify_block += "  语义记忆 (PG向量): ❌ 未找到\n"
    except Exception as e:
        verify_block += f"  语义记忆 (PG向量): ❌ 查询失败: {e}\n"

    results.append(verify_block)

    # ---- 汇总报告 ----
    total_cases = len(summary)
    passed_cases = sum(1 for _, p, _ in summary if p)
    failed_cases = [(l, iss) for l, p, iss in summary if not p]

    report_block = "\n" + "="*60 + "\n测试汇总\n" + "="*60 + "\n"
    report_block += f"  总用例: {total_cases}\n"
    report_block += f"  通过:   {passed_cases}\n"
    report_block += f"  失败:   {len(failed_cases)}\n"
    if failed_cases:
        report_block += "\n  失败详情:\n"
        for label, issues in failed_cases:
            report_block += f"    ❌ {label}\n"
            for iss in issues:
                report_block += f"       - {iss}\n"
    results.append(report_block)

    # ---- 写入文件 ----
    total_time = time.time() - t_start
    header = (
        f"{'='*60}\n"
        f"演示面板自动化测试结果\n"
        f"{'='*60}\n"
        f"执行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"测试用户: {TEST_USER_ID}\n"
        f"总耗时:   {total_time:.1f}s\n"
        f"用例总数: {len([c for c in cases if c[4] == 'send'])} 条消息\n"
        f"{'='*60}\n\n"
    )

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(results))

    print(f"\n{'='*60}")
    print(f"测试完成 | 通过 {passed_cases}/{total_cases} | 总耗时 {total_time:.1f}s")
    print(f"结果写入: {RESULT_FILE}")
    if failed_cases:
        print(f"\n失败用例 ({len(failed_cases)}):")
        for label, issues in failed_cases:
            print(f"  ❌ {label}: {'; '.join(issues)}")
    print(f"{'='*60}\n")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="演示面板自动化测试")
    parser.add_argument("--verbose", "-v", action="store_true", help="同时打印到终端")
    args = parser.parse_args()
    run_tests(verbose=args.verbose)
