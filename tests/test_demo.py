r"""
演示评估测试脚本

按面试演示话术逐场景执行，记录每轮的输入和输出，
结果写入 test_result.txt，由人工判断是否符合预期。

使用方式：
    cd d:\car_insurance_mvp
    python test_demo.py
"""

import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.logger import setup_logging, get_logger
setup_logging()

from src.chains.chains import init_graph
from src.state import set_graph
from src.rag import init_rag
from src.chat import chat_api

# ============================================================
# 初始化
# ============================================================
print("正在初始化（加载模型 + FAISS 索引）...")
graph, _ = init_graph()
set_graph(graph)
init_rag()
print("初始化完成，开始测试。\n")

# ============================================================
# 测试场景定义
# 每个场景 = (场景名, session组, [(输入, 说明), ...])
# session组相同的场景共享同一个 session_id（用于验证 route 切换）
# session组为 None 的场景使用独立 session
# ============================================================

SCENARIOS = [
    ("1. 问候", None, [
        ("你好", None),
    ]),
    ("2. 记忆", "A", [
        ("我叫张三，身份证 110101199001011234", "存入姓名+身份证"),
        ("我叫什么名字", "验证是否记住姓名"),
    ]),
    ("3. 保费多轮", "A", [
        ("帮我算一下特斯拉的保费", "信息不完整，AI应反问 → route: sale"),
        ("37岁，特斯拉Y，驾龄5年", "补全信息，AI应算出保费"),
    ]),
    ("4. 保单查询", "A", [
        ("我改变主意了，不计算保费了，帮我查一下保单，保单号 POL20260001",
         "route 应从 sale 切到 service，利用记忆中的身份证号查询"),
    ]),
    ("5. RAG条款", None, [
        ("车损险保障哪些情况？", "应触发RAG检索，返回条款内容"),
    ]),
    ("6. 太空飞船", None, [
        ("太空飞船的保险赔不赔？", "RAG无结果，AI自行回答"),
    ]),
    ("7. 转人工", None, [
        ("我要投诉，转人工", "应触发转人工，生成工单号"),
    ]),
    ("8. 输入超长", None, [
        ("测" * 1001, "应返回输入过长错误"),
    ]),
]


# ============================================================
# 执行测试
# ============================================================
def run_test():
    output_lines = []

    def log(text):
        print(text)
        output_lines.append(text)

    log("=" * 70)
    log("车险智能客服 MVP — 演示测试记录")
    log(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"共 {len(SCENARIOS)} 个场景")
    log("=" * 70)

    session_groups = {}  # session组名 -> session_id

    for scenario_name, session_group, turns in SCENARIOS:
        # 确定 session_id：同组共享 session，None 则独立新建
        if session_group and session_group in session_groups:
            session_id = session_groups[session_group]
        else:
            session_id = f"test_{uuid.uuid4().hex[:8]}"
            if session_group:
                session_groups[session_group] = session_id

        log(f"\n{'─' * 70}")
        log(f"【{scenario_name}】")
        log(f"{'─' * 70}")

        for turn_idx, (user_input, note) in enumerate(turns, 1):
            input_display = user_input[:60] + "..." if len(user_input) > 60 else user_input

            start = time.time()
            result = chat_api(session_id, user_input)
            elapsed = time.time() - start

            reply = result.get("content", {}).get("reply", "")
            error_msg = result.get("error_msg", "")
            route = result.get("route", "")
            transfer = result.get("content", {}).get("transfer", False)
            ticket_id = result.get("content", {}).get("ticket_id", "")

            output_text = reply or error_msg

            log(f"  轮次 {turn_idx}: {input_display}")
            if note:
                log(f"  说明: {note}")
            log(f"  路由: {route} | 转人工: {transfer} | 耗时: {elapsed:.1f}s")
            if ticket_id:
                log(f"  工单号: {ticket_id}")
            log(f"  输出: {output_text[:200]}{'...' if len(output_text) > 200 else ''}")
            log("")

    # 写入结果文件
    log(f"\n{'=' * 70}")
    log("测试记录结束")
    log(f"{'=' * 70}")

    output_path = os.path.join(os.path.dirname(__file__), "test_result.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\n结果已写入: {output_path}")


if __name__ == "__main__":
    run_test()
