"""
车险智能客服 MVP - 主入口
提供 Gradio 交互界面 + /health 健康检查接口
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import time
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="车险智能客服 MVP", version="0.1.0")

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


# ---------- 核心调用函数 ----------
def chat_api(session_id: str, message: str) -> dict:
    from src.core.routing import decide_route
    from src.chains.chains import init_chains
    from src.rag import init_rag

    print(f"[DEBUG] 收到消息: {message[:30]}...", flush=True)
    
    general_chain, agent_sale, agent_service, memory = init_chains()
    init_rag()
    
    route = decide_route(session_id, message)
    print(f"[路由决策] session={session_id}, message={message[:20]}..., route={route}", flush=True)
    
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # 统一的转人工检测函数
        def check_transfer_flag(result_dict):
            """检测 Agent 返回结果中是否包含转人工标志或工具调用"""
            messages = result_dict.get("messages", [])
            for msg in messages:
                # 检查消息内容
                if hasattr(msg, 'content') and "__TRANSFER__" in str(msg.content):
                    return True
                # 检查工具调用记录
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if isinstance(tc, dict) and tc.get('name') == 'transfer_to_human':
                            return True
                        elif hasattr(tc, 'get') and tc.get('name') == 'transfer_to_human':
                            return True
                # 检查 additional_kwargs 中的工具调用
                if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                    for tc in msg.additional_kwargs['tool_calls']:
                        if isinstance(tc, dict) and tc.get('function', {}).get('name') == 'transfer_to_human':
                            return True
            return False

        if route == "general":
            # 普通链：不带工具，但也是 Agent（带 Memory）
            result = general_chain.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config=config
            )
            last_msg = result["messages"][-1]
            reply = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            transfer_flag = False  # general 链没有转人工工具
        
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
        
        # 处理转人工
        if transfer_flag:
            ticket_id = f"TK{int(time.time())}{session_id[-4:]}"
            # 清理回复中的标志（如果有）
            clean_reply = reply.replace("__TRANSFER__", "").strip()
            return {
                "success": 0,
                "content": {
                    "reply": clean_reply or "正在为您转接人工客服，请稍候...",
                    "transfer": True,
                    "ticket_id": ticket_id
                },
                "route": route
            }
        
        return {
            "success": 0,
            "content": {
                "reply": reply,
                "transfer": False
            },
            "route": route
        }
    
    except Exception as e:
        print(f"[ERROR] chat_api 异常: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"success": -1, "error_msg": str(e), "content": {}}


# ---------- Gradio 界面 ----------
def create_gradio_interface():
    def chat_response(message, history):
        print(f"[Gradio] 用户输入: {message}", flush=True)
        session_id = f"gradio_{int(time.time())}"
        result = chat_api(session_id, message)
        if result["success"] != 0:
            return f"❌ 系统异常: {result.get('error_msg', '未知错误')}"
        
        # 提取路由信息
        route = result.get("route", "unknown")
        route_label = {
            "general": "💬 闲聊",
            "sale": "💰 报价链",
            "service": "🛠️ 理赔链"
        }.get(route, f"🔀 {route}")
        
        reply = result["content"]["reply"]
        if result["content"].get("transfer", False):
            ticket_id = result["content"].get("ticket_id", "")
            reply += f"\n\n🔄 已为您转接人工客服，工单号：{ticket_id}"
        
        # 在回复前面加上路由标签
        return f"<small>{route_label}</small>\n\n{reply}"

    demo = gr.ChatInterface(
        fn=chat_response,
        title="🚗 车险智能客服 MVP",
        description="基于路由决策 + LangChain 构建"
    )
    return demo


# ---------- 启动入口 ----------
if __name__ == "__main__":
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