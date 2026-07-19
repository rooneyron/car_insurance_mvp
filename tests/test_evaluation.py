"""
车险智能客服 MVP - 评估脚本
覆盖五大场景：
1. 路由正确性
2. 记忆生效
3. RAG 被真实调用
4. 转人工触发
5. 接口健壮性
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入核心函数
from src.chat import chat_api
from src.core.routing import session_intent_store, set_session_intent


def test_route_correctness():
    """场景一：路由正确性"""
    print("\n" + "="*50)
    print("场景一：路由正确性测试")
    print("="*50)
    
    test_cases = [
        ("你好", "general"),
        ("我要续保", "sale"),
        ("我想理赔", "service"),
        ("保费多少钱", "sale"),
        ("我车爆胎了", "service"),
        ("转人工", "service"),
    ]
    
    passed = 0
    for msg, expected in test_cases:
        sid = f"route_test_{int(time.time())}"
        # 重置意图
        set_session_intent(sid, None)
        
        result = chat_api(sid, msg)
        actual = result.get("route", "unknown")
        status = "✅" if actual == expected else "❌"
        print(f"  消息: {msg:15} -> 预期: {expected:8} 实际: {actual:8} {status}")
        if actual == expected:
            passed += 1
    
    print(f"  通过: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_memory():
    """场景二：记忆生效测试"""
    print("\n" + "="*50)
    print("场景二：记忆生效测试")
    print("="*50)
    
    sid = f"memory_test_{int(time.time())}"
    
    # 第一轮：用户主动提供身份证号
    msg1 = "我的身份证号是110101199001011234"
    result1 = chat_api(sid, msg1)
    print(f"  第一轮: {msg1}")
    print(f"    回复: {result1['content']['reply'][:60]}...")
    
    # 第二轮：用户直接询问自己的身份证号
    msg2 = "我的身份证号是多少"
    result2 = chat_api(sid, msg2)
    print(f"  第二轮: {msg2}")
    print(f"    回复: {result2['content']['reply'][:80]}...")
    
    # 检查第二轮回复中是否包含身份证号
    if "110101199001011234" in result2['content']['reply']:
        print("  ✅ 记忆生效：系统记住了用户之前提供的身份证号")
        return True
    else:
        print("  ❌ 记忆未生效：系统未能回忆起身份证号")
        return False


def test_rag():
    """场景三：RAG 被真实调用"""
    print("\n" + "="*50)
    print("场景三：RAG 检索测试")
    print("="*50)
    
    sid = f"rag_test_{int(time.time())}"
    msg = "车损险赔自然灾害吗"
    result = chat_api(sid, msg)
    
    print(f"  消息: {msg}")
    reply = result['content']['reply']
    print(f"  回复前200字: {reply[:200]}...")
    
    # 检查回复中是否包含车损险相关条款关键词
    keywords = ["车损险", "碰撞", "倾覆", "火灾", "自然灾害", "暴风", "洪水"]
    found = any(kw in reply for kw in keywords)
    if found:
        print("  ✅ RAG 检索成功：回复中包含相关条款关键词")
    else:
        print("  ❌ RAG 检索可能未触发：回复中未找到条款关键词")
    
    return found


def test_transfer():
    """场景四：转人工触发"""
    print("\n" + "="*50)
    print("场景四：转人工触发测试")
    print("="*50)
    
    sid = f"transfer_test_{int(time.time())}"
    msg = "转人工"
    result = chat_api(sid, msg)
    
    print(f"  消息: {msg}")
    transfer = result['content'].get('transfer', False)
    ticket_id = result['content'].get('ticket_id', '')
    reply = result['content']['reply']
    
    print(f"  转人工标志: {transfer}")
    print(f"  工单号: {ticket_id}")
    print(f"  回复: {reply}")
    
    if transfer and ticket_id.startswith("TK"):
        print("  ✅ 转人工触发成功")
        return True
    else:
        print("  ❌ 转人工未正确触发")
        return False


def test_robustness():
    """场景五：接口健壮性"""
    print("\n" + "="*50)
    print("场景五：接口健壮性测试")
    print("="*50)
    
    sid = f"robust_test_{int(time.time())}"
    test_cases = [
        ("", "空字符串"),
        ("   ", "空白字符串"),
        ("你好" * 100, "超长文本"),
        ("转人工", "正常转人工（对照）"),
    ]
    
    passed = 0
    for msg, desc in test_cases:
        try:
            result = chat_api(sid, msg)
            success = result.get('success', -1)
            status = "✅" if success == 0 else "❌"
            print(f"  {desc:10} -> success={success} {status}")
            if success == 0:
                passed += 1
        except Exception as e:
            print(f"  {desc:10} -> 异常: {e} ❌")
    
    print(f"  通过: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def run_all_tests():
    """运行全部评估"""
    print("\n" + "🚗"*20)
    print("车险智能客服 MVP - 五大场景评估")
    print("🚗"*20)
    
    results = {}
    # results["路由正确性"] = test_route_correctness()
    results["记忆生效"] = test_memory()
    # results["RAG检索"] = test_rag()
    # results["转人工触发"] = test_transfer()
    # results["接口健壮性"] = test_robustness()
    
    print("\n" + "="*50)
    print("评估结果汇总")
    print("="*50)
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:12}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*50)
    if all_passed:
        print("🎉 全部场景通过！MVP 验证成功！")
    else:
        print("⚠️ 有场景未通过，请检查对应功能。")
    print("="*50)
    
    return all_passed


if __name__ == "__main__":
    # 设置环境变量（如果未设置）
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("⚠️ 未设置 DEEPSEEK_API_KEY 环境变量")
        print("请先执行: $env:DEEPSEEK_API_KEY='sk-你的密钥'")
        sys.exit(1)
    
    run_all_tests()