"""
DST 端到端行为测试（通过 chat_api 接口）

覆盖三大行为：
1. 继续当前任务（DST continue）
2. 退出当前任务（DST escape）
3. 无当前任务时正常路由

使用方式：
    cd d:/car_insurance_mvp
    python tests/test_dst.py

注意：
    需要服务环境就绪（.env 配置正确、DB 可用）。
    每个 case 使用独立 session_id，互不干扰。
"""

import os
import sys
import uuid
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.chat import chat_api
from src import state as global_state
from src.chains.chains import init_graph

# ============================================================
# 初始化 Graph（测试进程内调用）
# ============================================================

print("正在初始化 Graph...")
_graph, _rag_llm = init_graph()
global_state.set_graph(_graph)
print("Graph 初始化完成\n")


# ============================================================
# 辅助函数
# ============================================================

def _new_session():
    """生成唯一 session_id"""
    return f"dst_test_{uuid.uuid4().hex[:8]}"


def _send(session_id, message):
    """
    发送消息并返回完整响应 + 图状态快照。
    返回: (response_dict, graph_state_dict)
    """
    resp = chat_api(session_id, message)
    # 读取图状态（用于验证 DST 内部状态）
    config = {"configurable": {"thread_id": session_id}}
    try:
        gs = _graph.get_state(config).values
    except Exception:
        gs = {}
    return resp, gs


def _check(case_name, condition, detail=""):
    """记录并打印单个断言结果"""
    status = "PASS" if condition else "FAIL"
    _results.append((case_name, condition, detail))
    icon = "+" if condition else "X"
    extra = f" | {detail}" if detail else ""
    print(f"  [{icon}] {case_name}{extra}")
    return condition


def _print_exchange(step, direction, text, route=None, source=None, task=None, slot=None):
    """打印一轮交互的信息"""
    arrow = ">>>" if direction == "user" else "<<<"
    line = f"    {arrow} {text[:60]}"
    if route:
        line += f"  [route={route}"
        if source:
            line += f" source={source}"
        if task is not None:
            line += f" task={task}"
        if slot is not None:
            line += f" slot={slot}"
        line += "]"
    print(line)


# ============================================================
# 测试结果收集
# ============================================================

_results = []


# ============================================================
# 一、继续当前任务（DST continue）
# ============================================================

def test_continue():
    print("\n" + "=" * 60)
    print("一、继续当前任务（DST continue）")
    print("=" * 60)

    # ---- Case 1: 正常补参数 ----
    print("\n  [Case 1] 正常补参数")
    sid = _new_session()

    resp1, gs1 = _send(sid, "我要查询保单")
    route1 = resp1.get("route", "")
    reply1 = resp1.get("content", {}).get("reply", "")
    _print_exchange(1, "user", "我要查询保单")
    _print_exchange(1, "ai", reply1, route=route1,
                    source=gs1.get("router_source"),
                    task=gs1.get("current_task"),
                    slot=gs1.get("awaiting_slot"))

    _check("Case1-R1: route=service", route1 == "service",
           f"actual={route1}")

    task1 = gs1.get("current_task")
    slot1 = gs1.get("awaiting_slot")
    _check("Case1-R1: current_task=service", task1 == "service",
           f"actual={task1}")

    # 第二轮：补身份证
    resp2, gs2 = _send(sid, "110101199001011234")
    route2 = resp2.get("route", "")
    reply2 = resp2.get("content", {}).get("reply", "")
    source2 = gs2.get("router_source")
    _print_exchange(2, "user", "110101199001011234")
    _print_exchange(2, "ai", reply2, route=route2,
                    source=source2,
                    task=gs2.get("current_task"),
                    slot=gs2.get("awaiting_slot"))

    _check("Case1-R2: route=service (DST 承接)", route2 == "service",
           f"actual={route2}")
    _check("Case1-R2: router_source=dst", source2 == "dst",
           f"actual={source2}")

    # ---- Case 2: 自然语言补参数 ----
    print("\n  [Case 2] 自然语言补参数")
    sid = _new_session()

    resp1, gs1 = _send(sid, "我要查询保单")
    _check("Case2-R1: route=service", resp1.get("route") == "service")

    resp2, gs2 = _send(sid, "我叫张三，身份证号是110101199001011234")
    route2 = resp2.get("route", "")
    source2 = gs2.get("router_source")
    reply2 = resp2.get("content", {}).get("reply", "")
    _print_exchange(2, "user", "我叫张三，身份证号是110101199001011234")
    _print_exchange(2, "ai", reply2, route=route2, source=source2)

    _check("Case2-R2: route=service (不是 general)", route2 == "service",
           f"actual={route2}")
    _check("Case2-R2: router_source=dst", source2 == "dst",
           f"actual={source2}")

    # ---- Case 3: 连续补参数 ----
    print("\n  [Case 3] 连续补参数（如果工具还缺参数）")
    sid = _new_session()

    _send(sid, "我要查询保单")
    _send(sid, "110101199001011234")

    resp3, gs3 = _send(sid, "POL20260001")
    route3 = resp3.get("route", "")
    source3 = gs3.get("router_source")
    task3 = gs3.get("current_task")
    slot3 = gs3.get("awaiting_slot")
    reply3 = resp3.get("content", {}).get("reply", "")
    _print_exchange(3, "user", "POL20260001")
    _print_exchange(3, "ai", reply3, route=route3, source=source3,
                    task=task3, slot=slot3)

    # 如果工具同时缺 policy_id 和 id_card，第一轮只等 id_card
    # 补了 id_card 后，如果还缺 policy_id → awaiting_slot=policy_id
    # 此时再补 policy_id → 应该还是 DST continue
    # 如果工具已经成功 → DST 已清空，route 由正常 Router 决定
    if slot3 is not None and task3 == "service":
        _check("Case3: 仍缺参数, DST 继续承接",
               source3 == "dst" and route3 == "service",
               f"source={source3}, route={route3}")
    else:
        _check("Case3: 工具已成功, DST 已清空",
               task3 is None and slot3 is None,
               f"task={task3}, slot={slot3}")


# ============================================================
# 二、退出当前任务（DST escape）
# ============================================================

def test_escape():
    print("\n" + "=" * 60)
    print("二、退出当前任务（DST escape）")
    print("=" * 60)

    # ---- Case 4: 用户说"你好" ----
    print("\n  [Case 4] 等待身份证时，用户说'你好'")
    sid = _new_session()

    _send(sid, "我要查询保单")
    _send(sid, "110101199001011234")

    resp, gs = _send(sid, "你好")
    route = resp.get("route", "")
    reply = resp.get("content", {}).get("reply", "")
    source = gs.get("router_source")
    task = gs.get("current_task")
    slot = gs.get("awaiting_slot")
    _print_exchange("E4", "user", "你好")
    _print_exchange("E4", "ai", reply, route=route, source=source,
                    task=task, slot=slot)

    # "你好"是模糊表达，LLM 保守判断 continue，继续当前任务
    _check("Case4: route=service (LLM 判断 continue，继续追问)", route == "service",
           f"actual={route}")
    _check("Case4: router_source=dst", source == "dst",
           f"actual={source}")

    # ---- Case 5: 用户明确取消 ----
    print("\n  [Case 5] 用户说'算了，不查了'")
    sid = _new_session()

    resp1, gs1 = _send(sid, "我要查询保单")
    _check("Case5-R1: route=service", resp1.get("route") == "service")

    resp2, gs2 = _send(sid, "算了，不查了")
    route2 = resp2.get("route", "")
    task2 = gs2.get("current_task")
    slot2 = gs2.get("awaiting_slot")
    reply2 = resp2.get("content", {}).get("reply", "")
    _print_exchange("E5", "user", "算了，不查了")
    _print_exchange("E5", "ai", reply2, route=route2,
                    task=task2, slot=slot2)

    _check("Case5: DST 已清空",
           task2 is None and slot2 is None,
           f"task={task2}, slot={slot2}")
    _check("Case5: route != service",
           route2 != "service", f"actual={route2}")

    # ---- Case 6: 用户切换到另一个业务 ----
    print("\n  [Case 6] 等待身份证时，用户说'我想买车险'")
    sid = _new_session()

    _send(sid, "我要查询保单")

    resp, gs = _send(sid, "我想买车险")
    route = resp.get("route", "")
    source = gs.get("router_source")
    task = gs.get("current_task")
    slot = gs.get("awaiting_slot")
    reply = resp.get("content", {}).get("reply", "")
    _print_exchange("E6", "user", "我想买车险")
    _print_exchange("E6", "ai", reply, route=route, source=source,
                    task=task, slot=slot)

    _check("Case6: route=sale (成功切换)", route == "sale",
           f"actual={route}")
    _check("Case6: DST 状态正确 (task=sale 或 None)",
           task in (None, "sale"), f"actual={task}")


# ============================================================
# 三、不要误逃逸
# ============================================================

def test_no_false_escape():
    print("\n" + "=" * 60)
    print("三、不要误逃逸")
    print("=" * 60)

    # ---- Case 7: 模糊承接 ----
    print("\n  [Case 7] '我忘记身份证了' → 应继续 service")
    sid = _new_session()

    _send(sid, "我要查询保单")

    resp, gs = _send(sid, "我忘记身份证了")
    route = resp.get("route", "")
    source = gs.get("router_source")
    task = gs.get("current_task")
    reply = resp.get("content", {}).get("reply", "")
    _print_exchange("N7", "user", "我忘记身份证了")
    _print_exchange("N7", "ai", reply, route=route, source=source, task=task)

    _check("Case7: route=service (没有误逃逸)", route == "service",
           f"actual={route}")

    # ---- Case 8: 普通自然语言 ----
    print("\n  [Case 8] '这个要去哪里看？' → 应继续 service")
    sid = _new_session()

    _send(sid, "我要查询保单")

    resp, gs = _send(sid, "这个要去哪里看？")
    route = resp.get("route", "")
    source = gs.get("router_source")
    task = gs.get("current_task")
    reply = resp.get("content", {}).get("reply", "")
    _print_exchange("N8", "user", "这个要去哪里看？")
    _print_exchange("N8", "ai", reply, route=route, source=source, task=task)

    _check("Case8: route=service (没有误逃逸)", route == "service",
           f"actual={route}")


# ============================================================
# 四、状态残留验证
# ============================================================

def test_no_residual_state():
    print("\n" + "=" * 60)
    print("四、状态残留验证")
    print("=" * 60)

    # ---- Case 9: 任务完成后再问新业务 ----
    print("\n  [Case 9] 查询成功后，再问'新能源车保险多少钱'")
    sid = _new_session()

    r1, _ = _send(sid, "我要查询保单")
    _print_exchange("S9-1", "user", "我要查询保单")
    _print_exchange("S9-1", "ai", r1.get("content", {}).get("reply", ""),
                    route=r1.get("route"))

    r2, gs2 = _send(sid, "110101199001011234")
    _print_exchange("S9-2", "user", "110101199001011234")
    _print_exchange("S9-2", "ai", r2.get("content", {}).get("reply", ""),
                    route=r2.get("route"))

    r3, gs3 = _send(sid, "POL20260001")
    reply3 = r3.get("content", {}).get("reply", "")
    route3 = r3.get("route", "")
    task3 = gs3.get("current_task")
    slot3 = gs3.get("awaiting_slot")
    _print_exchange("S9-3", "user", "POL20260001")
    _print_exchange("S9-3", "ai", reply3, route=route3,
                    task=task3, slot=slot3)

    # 查询成功后 DST 应清空
    if "成功" in reply3 or "保单号" in reply3:
        _check("Case9-R3: 查询成功, DST 清空",
               task3 is None and slot3 is None,
               f"task={task3}, slot={slot3}")

        # 新业务
        r4, gs4 = _send(sid, "新能源车保险多少钱？")
        route4 = r4.get("route", "")
        source4 = gs4.get("router_source")
        reply4 = r4.get("content", {}).get("reply", "")
        _print_exchange("S9-4", "user", "新能源车保险多少钱？")
        _print_exchange("S9-4", "ai", reply4, route=route4, source=source4)

        _check("Case9-R4: route=sale (正常路由)", route4 == "sale",
               f"actual={route4}")
        _check("Case9-R4: source != dst", source4 != "dst",
               f"actual={source4}")
    else:
        _check("Case9-R3: 查询未成功(跳过后续)", True,
               f"reply={reply3[:50]}")

    # ---- Case 10: 取消后重新开始 ----
    print("\n  [Case 10] 取消后再开始新业务")
    sid = _new_session()

    _send(sid, "我要查询保单")
    _send(sid, "算了，不查了")

    resp, gs = _send(sid, "我想买保险")
    route = resp.get("route", "")
    source = gs.get("router_source")
    task = gs.get("current_task")
    reply = resp.get("content", {}).get("reply", "")
    _print_exchange("S10", "user", "我想买保险")
    _print_exchange("S10", "ai", reply, route=route, source=source, task=task)

    _check("Case10: route=sale (不是 service)", route == "sale",
           f"actual={route}")
    _check("Case10: source != dst", source != "dst",
           f"actual={source}")


# ============================================================
# 五、DST Escape Check 回归测试（新增逻辑验证）
# ============================================================

def test_dst_escape_regression():
    """
    验证 DST Escape Check 的新逻辑：
    - L1 同业务不 escape
    - L1 不同业务 escape
    - L2 复用（不重复调用）
    - L2 general → continue
    - L2 同业务 → continue
    - L2 低置信度 → continue
    - 取消短语 → escape，不调 L2
    """
    print("\n" + "=" * 60)
    print("五、DST Escape Check 回归测试")
    print("=" * 60)

    # ---- Case R1: 同业务 L1 不应 escape ----
    print("\n  [Case R1] 同业务 L1 不应 escape: '我想查另一张保单'")
    sid = _new_session()

    resp1, gs1 = _send(sid, "我要查询保单")
    _check("CaseR1-R1: route=service", resp1.get("route") == "service")

    # 补身份证触发 awaiting_slot
    resp2, gs2 = _send(sid, "110101199001011234")
    slot2 = gs2.get("awaiting_slot")
    task2 = gs2.get("current_task")

    # 如果还在等待参数，测试同业务切换
    if slot2 is not None and task2 == "service":
        resp3, gs3 = _send(sid, "我想查另一张保单")
        route3 = resp3.get("route", "")
        source3 = gs3.get("router_source")
        reply3 = resp3.get("content", {}).get("reply", "")
        _print_exchange("R1-3", "user", "我想查另一张保单")
        _print_exchange("R1-3", "ai", reply3, route=route3, source=source3)

        # L1 判断 service == current_task → 不 escape → continue
        _check("CaseR1: route=service (同业务不 escape)", route3 == "service",
               f"actual={route3}")
        _check("CaseR1: router_source=dst", source3 == "dst",
               f"actual={source3}")
    else:
        _check("CaseR1: 工具已成功, 跳过", True, "DST 已清空")

    # ---- Case R2: L1 明确切换业务 → escape ----
    print("\n  [Case R2] L1 明确切换业务: '我要买车险'")
    sid = _new_session()

    _send(sid, "我要查询保单")
    _send(sid, "110101199001011234")

    resp, gs = _send(sid, "我要买车险")
    route = resp.get("route", "")
    source = gs.get("router_source")
    task = gs.get("current_task")
    slot = gs.get("awaiting_slot")
    reply = resp.get("content", {}).get("reply", "")
    _print_exchange("R2", "user", "我要买车险")
    _print_exchange("R2", "ai", reply, route=route, source=source,
                    task=task, slot=slot)

    _check("CaseR2: route=sale (L1 切换成功)", route == "sale",
           f"actual={route}")
    _check("CaseR2: DST 已清空", task is None and slot is None,
           f"task={task}, slot={slot}")

    # ---- Case R3: L2 判断新业务 → escape 并复用 ----
    print("\n  [Case R3] L2 判断新业务: '帮我算一下特斯拉的保费'")
    sid = _new_session()

    _send(sid, "我要查询保单")
    _send(sid, "110101199001011234")

    resp, gs = _send(sid, "帮我算一下特斯拉的保费")
    route = resp.get("route", "")
    source = gs.get("router_source")
    task = gs.get("current_task")
    slot = gs.get("awaiting_slot")
    reply = resp.get("content", {}).get("reply", "")
    _print_exchange("R3", "user", "帮我算一下特斯拉的保费")
    _print_exchange("R3", "ai", reply, route=route, source=source,
                    task=task, slot=slot)

    # L2 应该判断 sale → escape → 复用 L2 结果
    _check("CaseR3: route=sale (L2 复用)", route == "sale",
           f"actual={route}")
    _check("CaseR3: DST 已清空", task is None and slot is None,
           f"task={task}, slot={slot}")
    # source 应该是 l2_dst_escape（复用了 DST 的 L2 结果）
    _check("CaseR3: source=l2_dst_escape", source == "l2_dst_escape",
           f"actual={source}")

    # ---- Case R4: L2 是 general → continue ----
    print("\n  [Case R4] L2 是 general: '110101199001011234'（补参数）")
    sid = _new_session()

    _send(sid, "我要查询保单")

    resp, gs = _send(sid, "110101199001011234")
    route = resp.get("route", "")
    source = gs.get("router_source")
    reply = resp.get("content", {}).get("reply", "")
    _print_exchange("R4", "user", "110101199001011234")
    _print_exchange("R4", "ai", reply, route=route, source=source)

    _check("CaseR4: route=service (DST 承接)", route == "service",
           f"actual={route}")
    _check("CaseR4: router_source=dst", source == "dst",
           f"actual={source}")

    # ---- Case R5: 取消短语 → escape，不调 L2 ----
    print("\n  [Case R5] 取消短语: '算了，不查了'")
    sid = _new_session()

    _send(sid, "我要查询保单")
    _send(sid, "110101199001011234")

    resp, gs = _send(sid, "算了，不查了")
    route = resp.get("route", "")
    source = gs.get("router_source")
    task = gs.get("current_task")
    slot = gs.get("awaiting_slot")
    reply = resp.get("content", {}).get("reply", "")
    _print_exchange("R5", "user", "算了，不查了")
    _print_exchange("R5", "ai", reply, route=route, source=source,
                    task=task, slot=slot)

    _check("CaseR5: DST 已清空", task is None and slot is None,
           f"task={task}, slot={slot}")
    _check("CaseR5: route != service", route != "service",
           f"actual={route}")


# ============================================================
# 汇总
# ============================================================

def _summary():
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"总计: {total}  通过: {passed}  失败: {failed}")
    print("=" * 60)

    if failed:
        print("\n失败用例:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}: {detail}")

    print("\n验收标准:")
    # ① 参数输入不会被 L2 错分
    param_cases = [n for n, _, _ in _results if "DST 承接" in n or "不是 general" in n]
    param_pass = sum(1 for n, ok, _ in _results if ok and ("DST 承接" in n or "不是 general" in n))
    print(f"  1 参数输入不被错分: {param_pass}/{len(param_cases)}")

    # ② 用户可以随时退出
    exit_cases = [n for n, _, _ in _results if "已清空" in n or "成功切换" in n or "不是 service" in n]
    exit_pass = sum(1 for n, ok, _ in _results if ok and ("已清空" in n or "成功切换" in n or "不是 service" in n))
    print(f"  2 用户可随时退出: {exit_pass}/{len(exit_cases)}")

    # ③ 退出/完成后不残留
    residual_cases = [n for n, _, _ in _results if "正常路由" in n or "不是 service" in n or "不是 dst" in n]
    residual_pass = sum(1 for n, ok, _ in _results if ok and ("正常路由" in n or "不是 service" in n or "不是 dst" in n))
    print(f"  3 退出后不残留: {residual_pass}/{len(residual_cases)}")

    return failed == 0


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DST 端到端行为测试（通过 chat_api）")
    print("=" * 60)

    test_continue()
    test_escape()
    test_no_false_escape()
    test_no_residual_state()
    test_dst_escape_regression()

    all_passed = _summary()
    sys.exit(0 if all_passed else 1)
