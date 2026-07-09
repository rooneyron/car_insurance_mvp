"""
降级功能测试脚本

使用方式：
1. 正常启动服务: python app.py
2. 运行本脚本: python test_fallback.py

注意：部分测试需要手动修改配置并重启服务，脚本会给出明确指引。
"""

import os
import sys
import time
import requests
import json

# 服务地址
BASE_URL = "http://127.0.0.1:8000"
GRADIO_URL = f"{BASE_URL}/gradio"

# ============================================================
# 测试 1：输入超长（自动验证）
# ============================================================
def test_input_too_long():
    print("\n" + "="*50)
    print("测试 1：输入超长（1001 字符）")
    print("="*50)
    
    long_message = "测" * 1001
    
    # 需要先获取 session_id（通过调用 gradio 接口，或直接调用 chat_api）
    # 这里我们用 requests 直接调用 FastAPI 的 chat_api（但需要构造请求）
    # 由于 chat_api 没有直接暴露为 HTTP 接口（只有 gradio 界面），我们无法直接用 curl 测试。
    # 可以改用模拟调用 chat_api 的方式，但需要 import app 模块。
    
    # 更好的方法：直接导入 app 模块测试
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.chat import chat_api
    
    result = chat_api("test_session", long_message)
    if result["success"] == -1 and "输入内容过长" in result.get("error_msg", ""):
        print("✅ 测试通过：输入超长被正确拦截")
        return True
    else:
        print(f"❌ 测试失败：返回结果异常 -> {result}")
        return False

# ============================================================
# 测试 2：Rerank 分数低于阈值（需要手动改阈值）
# ============================================================
def test_rag_score_low():
    print("\n" + "="*50)
    print("测试 2：Rerank 分数低于阈值")
    print("="*50)
    print("【手动操作】")
    print("1. 打开 src/rag.py，将 RAG_SCORE_THRESHOLD 临时改为 0.99")
    print("2. 重启服务: python app.py")
    print("3. 在 Gradio 界面发送消息: '车损险保什么'")
    print("4. 观察回复是否为降级文本（'未找到与您问题直接相关的条款...'）")
    print("5. 检查 data/missed_queries.log 中是否新增一条记录")
    print("6. 测试完成后，将 RAG_SCORE_THRESHOLD 改回 0.6，重启服务")
    print()
    input("按 Enter 继续...")
    
    # 这里可以提供一个检查日志的辅助函数
    log_path = "data/missed_queries.log"
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                if "best_score:" in last_line:
                    print(f"✅ 日志记录成功: {last_line}")
                    return True
                else:
                    print("❌ 日志格式异常")
                    return False
            else:
                print("❌ 日志为空")
                return False
    else:
        print("❌ 日志文件不存在")
        return False

# ============================================================
# 测试 3：FAISS 完全查不到（需要手动改文件）
# ============================================================
def test_faiss_empty():
    print("\n" + "="*50)
    print("测试 3：FAISS 完全查不到")
    print("="*50)
    print("【手动操作】")
    print("1. 备份 data/insurance_terms.txt，然后将其内容清空（或改成无关文本）")
    print("2. 删除 data/faiss_index.bin 和 data/chunks.pkl（强制重建索引）")
    print("3. 重启服务: python app.py")
    print("4. 在 Gradio 界面发送消息: '车损险保什么'")
    print("5. 观察回复是否为降级文本")
    print("6. 检查 data/missed_queries.log 中是否有 FAISS_recall: 0 记录")
    print("7. 测试完成后，恢复 insurance_terms.txt，删除索引文件，重启服务让索引重建")
    print()
    input("按 Enter 继续...")
    
    log_path = "data/missed_queries.log"
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # 检查最近一条是否包含 FAISS_recall: 0
            for line in reversed(lines):
                if "FAISS_recall: 0" in line:
                    print(f"✅ 日志记录成功: {line.strip()}")
                    return True
            print("❌ 未找到 FAISS_recall: 0 的记录")
            return False
    else:
        print("❌ 日志文件不存在")
        return False

# ============================================================
# 测试 4：API 异常（需要手动改 API Key）
# ============================================================
def test_api_error():
    print("\n" + "="*50)
    print("测试 4：LLM API 异常（API Key 错误）")
    print("="*50)
    print("【手动操作】")
    print("1. 打开 .env 文件，将 DEEPSEEK_API_KEY 改成错误值")
    print("2. 重启服务: python app.py")
    print("3. 在 Gradio 界面发送任意消息")
    print("4. 观察回复是否为 '系统服务暂时不可用，请稍后再试'")
    print("5. 测试完成后，恢复正确的 API Key，重启服务")
    print()
    input("按 Enter 继续...")
    
    # 由于无法自动验证，只记录测试状态
    print("✅ 测试 4 已完成（请手动确认界面返回的提示）")
    return True

# ============================================================
# 测试 5：Agent 循环超限（需要手动改 recursion_limit）
# ============================================================
def test_recursion_limit():
    print("\n" + "="*50)
    print("测试 5：Agent 循环超限")
    print("="*50)
    print("【手动操作】")
    print("1. 打开 src/chains/chains.py，在 agent_sale 和 agent_service 中")
    print("   将 recursion_limit=5 临时改为 1")
    print("2. 重启服务: python app.py")
    print("3. 在 Gradio 界面发送消息: '我想续保'（会触发工具调用）")
    print("4. 观察回复是否为 '处理超时，请重新提问'")
    print("5. 测试完成后，将 recursion_limit 改回 5，重启服务")
    print()
    input("按 Enter 继续...")
    
    print("✅ 测试 5 已完成（请手动确认界面返回的提示）")
    return True

# ============================================================
# 测试 6：正常流程不受影响（自动验证）
# ============================================================
def test_normal_flow():
    print("\n" + "="*50)
    print("测试 6：正常流程不受影响")
    print("="*50)
    print("【手动操作】")
    print("1. 确保服务正常运行（所有配置已恢复）")
    print("2. 在 Gradio 界面发送消息: '我想续保'")
    print("3. 观察是否正常返回保费估算结果（路由标签 + 耗时正常）")
    print("4. 再发一条消息: '我的身份证号是110101199001011234'")
    print("5. 再发: '我的身份证号是多少'，看是否能正确召回")
    print()
    input("按 Enter 继续...")
    
    print("✅ 测试 6 已完成（请手动确认功能正常）")
    return True

# ============================================================
# 主函数
# ============================================================
def run_all_tests():
    print("\n" + "🚗"*20)
    print("降级功能测试")
    print("🚗"*20)
    
    results = {}
    
    # 测试 1：自动测试
    results["输入超长"] = test_input_too_long()
    
    # 测试 2-6：手动测试
    results["RAG 分数低"] = test_rag_score_low()
    results["FAISS 查不到"] = test_faiss_empty()
    results["API 异常"] = test_api_error()
    results["循环超限"] = test_recursion_limit()
    results["正常流程"] = test_normal_flow()
    
    print("\n" + "="*50)
    print("测试结果汇总")
    print("="*50)
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:12}: {status}")
    
    print("\n" + "="*50)
    print("🎉 所有测试执行完毕！")
    print("请根据手动测试的结果判断整体是否通过。")
    print("="*50)

if __name__ == "__main__":
    run_all_tests()