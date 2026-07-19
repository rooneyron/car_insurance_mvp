"""
Session 内存锁并发测试

验证同一 session_id 并发调用时，第二个请求立即返回 SESSION_BUSY 错误。
"""

import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.logger import setup_logging
setup_logging()

from src.chains.chains import init_graph
from src.state import set_graph
from src.rag import init_rag
from src.chat import chat_api

print("正在初始化...")
graph, _ = init_graph()
set_graph(graph)
init_rag()
print("初始化完成，开始并发测试。\n")


def test_concurrent_lock():
    """同一 session_id 并发发 2 个请求，验证第二个被拦截"""
    session_id = f"test_lock_{uuid.uuid4().hex[:8]}"
    
    results = {}
    
    def call_api(idx):
        result = chat_api(session_id, "你好")
        return idx, result
    
    # 用 2 个线程并发调用
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_api, i) for i in range(2)]
        for f in as_completed(futures):
            idx, result = f.result()
            results[idx] = result
    
    # 验证结果
    success_count = 0
    busy_count = 0
    
    for idx, result in results.items():
        if result.get("success") == 0:
            success_count += 1
            print(f"  请求 {idx}: ✅ 正常返回")
        elif "处理中" in result.get("error_msg", ""):
            busy_count += 1
            print(f"  请求 {idx}: 🔒 被锁拦截 (SESSION_BUSY)")
        else:
            print(f"  请求 {idx}: ❓ 未知结果: {result}")
    
    print(f"\n结果: 成功={success_count}, 被拦截={busy_count}")
    
    if success_count == 1 and busy_count == 1:
        print("✅ 内存锁测试通过！")
        return True
    else:
        print("❌ 内存锁测试失败！")
        return False


def test_different_sessions():
    """不同 session_id 不应互相影响"""
    session1 = f"test_{uuid.uuid4().hex[:8]}"
    session2 = f"test_{uuid.uuid4().hex[:8]}"
    
    results = {}
    
    def call_api(session_id, idx):
        result = chat_api(session_id, "你好")
        return idx, result
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(call_api, session1, 1),
            executor.submit(call_api, session2, 2),
        ]
        for f in as_completed(futures):
            idx, result = f.result()
            results[idx] = result
    
    success_count = sum(1 for r in results.values() if r.get("success") == 0)
    
    if success_count == 2:
        print("✅ 不同 session 互不影响！")
        return True
    else:
        print(f"❌ 不同 session 测试失败！成功={success_count}/2")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("测试 1: 同一 session 并发")
    print("=" * 50)
    test1 = test_concurrent_lock()
    
    print("\n" + "=" * 50)
    print("测试 2: 不同 session 并发")
    print("=" * 50)
    test2 = test_different_sessions()
    
    print("\n" + "=" * 50)
    if test1 and test2:
        print("全部测试通过 ✅")
    else:
        print("部分测试失败 ❌")
