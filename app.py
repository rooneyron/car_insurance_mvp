"""
车险智能客服 MVP - 主入口
提供 Gradio 交互界面 + /health 健康检查接口
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="车险智能客服 MVP", version="0.1.0")


# ============================================================
# 计时工具
# ============================================================
class Timer:
    def __init__(self):
        self.logs = []
    
    def start(self, label: str):
        self.logs.append({"label": label, "start": time.time(), "end": None})
    
    def stop(self, label: str = None):
        if label:
            for log in self.logs:
                if log["label"] == label and log["end"] is None:
                    log["end"] = time.time()
                    break
        else:
            for log in reversed(self.logs):
                if log["end"] is None:
                    log["end"] = time.time()
                    break
    
    def get_report(self) -> str:
        report = []
        for log in self.logs:
            if log["end"] is not None:
                elapsed = (log["end"] - log["start"]) * 1000
                report.append(f"  ├── {log['label']}: {elapsed:.0f}ms")
        return "\n".join(report)
    
    def get_total_ms(self) -> float:
        if self.logs and self.logs[-1]["end"] is not None:
            return (self.logs[-1]["end"] - self.logs[0]["start"]) * 1000
        return 0


# ============================================================
# API 路由
# ============================================================
@app.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "service": "car_insurance_mvp",
            "version": "0.1.0",
            "timestamp": int(time.time())
        }
    )

@app.get("/")
async def root():
    return {
        "message": "车险智能客服 MVP",
        "docs": "/docs",
        "health": "/health",
        "gradio": "/gradio"
    }


# ============================================================
# 核心调用函数
# ============================================================
def chat_api(session_id: str, message: str) -> dict:
    from src.core.routing import decide_route
    from src.chains.chains import init_chains
    from src.rag import init_rag

    timer = Timer()
    timer.start("总耗时")
    
    print(f"[DEBUG] 收到消息: {message[:30]}...", flush=True)
    
    timer.start("初始化 Chain")
    general_chain, agent_sale, agent_service, memory = init_chains()
    timer.stop("初始化 Chain")
    
    timer.start("初始化 RAG")
    init_rag()
    timer.stop("初始化 RAG")
    
    timer.start("路由决策")
    route = decide_route(session_id, message)
    timer.stop("路由决策")
    print(f"[路由决策] session={session_id}, message={message[:20]}..., route={route}", flush=True)
    
    config = {"configurable": {"thread_id": session_id}}
    
    def check_transfer_flag(result_dict):
        messages = result_dict.get("messages", [])
        for msg in messages:
            if hasattr(msg, 'content') and "__TRANSFER__" in str(msg.content):
                return True
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if isinstance(tc, dict) and tc.get('name') == 'transfer_to_human':
                        return True
                    elif hasattr(tc, 'get') and tc.get('name') == 'transfer_to_human':
                        return True
            if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                for tc in msg.additional_kwargs['tool_calls']:
                    if isinstance(tc, dict) and tc.get('function', {}).get('name') == 'transfer_to_human':
                        return True
        return False

    try:
        timer.start("Agent 调用")
        
        if route == "general":
            result = general_chain.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config=config
            )
            last_msg = result["messages"][-1]
            reply = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            transfer_flag = False
        
        elif route == "sale":
            result = agent_sale.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config=config
            )
            last_msg = result["messages"][-1]
            reply = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            transfer_flag = check_transfer_flag(result)
        
        elif route == "service":
            result = agent_service.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config=config
            )
            last_msg = result["messages"][-1]
            reply = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            transfer_flag = check_transfer_flag(result)
        
        else:
            return {"success": -1, "error_msg": f"未知路由: {route}", "content": {}}
        
        timer.stop("Agent 调用")
        timer.stop("总耗时")
        
        # 打印性能日志
        print(f"\n[性能日志] 总耗时: {timer.get_total_ms():.0f}ms")
        print(timer.get_report())
        print("-" * 40)
        
        if transfer_flag:
            ticket_id = f"TK{int(time.time())}{session_id[-4:]}"
            clean_reply = reply.replace("__TRANSFER__", "").strip()
            return {
                "success": 0,
                "content": {
                    "reply": clean_reply or "正在为您转接人工客服，请稍候...",
                    "transfer": True,
                    "ticket_id": ticket_id
                },
                "route": route,
                "elapsed_ms": timer.get_total_ms()
            }
        
        return {
            "success": 0,
            "content": {
                "reply": reply,
                "transfer": False
            },
            "route": route,
            "elapsed_ms": timer.get_total_ms()
        }
    
    except Exception as e:
        print(f"[ERROR] chat_api 异常: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"success": -1, "error_msg": str(e), "content": {}}


# ============================================================
# Gradio 界面
# ============================================================
def create_gradio_interface():
    def chat_response(message, history, session_id):
        print(f"[Gradio] 用户输入: {message}", flush=True)
        result = chat_api(session_id, message)
        if result["success"] != 0:
            return f"❌ 系统异常: {result.get('error_msg', '未知错误')}", session_id
        
        route = result.get("route", "unknown")
        route_label = {
            "general": "💬 闲聊",
            "sale": "💰 报价链",
            "service": "🛠️ 理赔链"
        }.get(route, f"🔀 {route}")
        
        elapsed_ms = result.get("elapsed_ms", 0)
        elapsed_str = f"⏱️ {elapsed_ms/1000:.1f}s" if elapsed_ms > 0 else ""
        
        reply = result["content"]["reply"]
        if result["content"].get("transfer", False):
            ticket_id = result["content"].get("ticket_id", "")
            reply += f"\n\n🔄 已为您转接人工客服，工单号：{ticket_id}"
        
        return f"<small>{route_label}  {elapsed_str}</small>\n\n{reply}", session_id

    with gr.Blocks(title="车险智能客服 MVP") as demo:
        gr.Markdown("# 🚗 车险智能客服 MVP")
        gr.Markdown("基于路由决策 + LangChain 构建")
        
        # 用 State 保存 session_id
        session_state = gr.State(value=f"gradio_{int(time.time())}")
        # 不用 type 参数，Gradio 6.x 默认接受字典列表
        chatbot = gr.Chatbot(label="对话窗口")
        msg = gr.Textbox(label="输入消息", placeholder="请输入你的问题...")
        clear = gr.Button("清空对话")
        
        def respond(message, chat_history, session_id):
            if not message:
                return "", chat_history, session_id
            bot_message, session_id = chat_response(message, chat_history, session_id)
            # 直接传字典列表，Gradio 6.x 原生支持
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_message})
            return "", chat_history, session_id
        
        msg.submit(respond, [msg, chatbot, session_state], [msg, chatbot, session_state])
        clear.click(
            lambda: (None, [], f"gradio_{int(time.time())}"),
            None,
            [chatbot, session_state],
            queue=False
        )
    
    return demo


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    # ---------- 预加载：启动时加载所有模型 ----------
    print("⏳ 正在预加载模型...")
    from src.chains.chains import init_chains
    from src.rag import init_rag
    try:
        init_chains()
        init_rag()
        print("✅ 预加载完成")
    except Exception as e:
        print(f"⚠️ 预加载失败: {e}")
        print("   服务仍会启动，但第一条消息可能较慢")
    
    demo = create_gradio_interface()
    app = gr.mount_gradio_app(app, demo, path="/gradio")
    
    print("=" * 50)
    print("🚗 车险智能客服 MVP 已启动")
    print("=" * 50)
    print(f"📖 API 文档: http://127.0.0.1:8000/docs")
    print(f"❤️ 健康检查: http://127.0.0.1:8000/health")
    print(f"💬 Gradio 界面: http://127.0.0.1:8000/gradio")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)