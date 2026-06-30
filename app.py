"""
车险智能客服 MVP - 主入口
提供 Gradio 交互界面 + /health 健康检查接口
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import threading
import time
import gradio as gr

# ---------- 创建 FastAPI 应用 ----------
app = FastAPI(title="车险智能客服 MVP", version="0.1.0")

@app.get("/health")
async def health_check():
    """
    健康检查接口
    用于 Render 等部署平台检测服务是否存活
    """
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
    """根路径，返回服务信息"""
    return {
        "message": "车险智能客服 MVP",
        "docs": "/docs",
        "health": "/health"
    }


# ---------- Gradio 界面（后续填充） ----------
def create_gradio_interface():
    """
    创建 Gradio 交互界面
    TODO: 在 Step 4 中实现完整对话逻辑
    """
    def chat_response(message, history):
        # 临时占位，后续会接入真正的路由 + Agent
        return f"收到你的消息：{message}\n（路由逻辑已就绪，Agent 开发中...）"
    
    with gr.Blocks(title="车险智能客服 MVP") as demo:
        gr.Markdown("# 🚗 车险智能客服 MVP")
        gr.Markdown("基于路由决策 + LangChain 构建（开发中）")
        
        chatbot = gr.Chatbot(label="对话窗口")
        msg = gr.Textbox(label="输入消息", placeholder="请输入你的问题...")
        clear = gr.Button("清空对话")
        
        def respond(message, chat_history):
            bot_message = chat_response(message, chat_history)
            chat_history.append((message, bot_message))
            return "", chat_history
        
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
    
    return demo


# ---------- 启动入口 ----------
if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "gradio":
        # 单独启动 Gradio 界面（开发用）
        demo = create_gradio_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    else:
        # 默认启动 FastAPI（包含 /health 和 Gradio 挂载）
        demo = create_gradio_interface()
        
        # 将 Gradio 挂载到 FastAPI 的 /gradio 路径
        app = gr.mount_gradio_app(app, demo, path="/gradio")
        
        print("=" * 50)
        print("🚗 车险智能客服 MVP 已启动")
        print("=" * 50)
        print(f"📖 API 文档: http://127.0.0.1:8000/docs")
        print(f"❤️ 健康检查: http://127.0.0.1:8000/health")
        print(f"💬 Gradio 界面: http://127.0.0.1:8000/gradio")
        print("=" * 50)
        
        uvicorn.run(app, host="0.0.0.0", port=8000)