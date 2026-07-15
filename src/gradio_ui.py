"""
Gradio 交互界面
提供 Web UI 供用户与智能客服对话。
"""

import uuid
import gradio as gr
from src.chat import chat_api_stream
from src.route_types import Route, ROUTE_LABELS
from src.logger import get_logger

logger = get_logger(__name__)


def create_gradio_interface():
    with gr.Blocks(title="车险智能客服 MVP") as demo:
        gr.Markdown("# 🚗 车险智能客服 MVP")
        gr.Markdown("基于路由决策 + LangChain 构建")

        # 用 State 保存 session_id（初始为空，页面加载时动态生成）
        session_state = gr.State(value="")
        chatbot = gr.Chatbot(label="对话窗口")
        msg = gr.Textbox(label="输入消息", placeholder="请输入你的问题...")
        clear = gr.Button("清空对话")

        # 流式响应函数（async generator）
        async def respond(message, chat_history, session_id):
            # 空输入保护：直接返回，不修改任何状态
            if not message or not message.strip():
                yield message, chat_history, session_id
                return

            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": message})
            # 先显示用户消息 + "正在思考..." 占位
            chat_history.append({"role": "assistant", "content": "正在思考..."})
            yield "", chat_history, session_id

            assistant_idx = len(chat_history) - 1

            last_metadata = None
            async for partial_text, metadata in chat_api_stream(session_id, message):
                if metadata and metadata.get("error"):
                    chat_history[assistant_idx]["content"] = f"❌ {partial_text}"
                    yield "", chat_history, session_id
                    return
                if metadata:
                    last_metadata = metadata
                # 空文本说明工具调用中，显示"正在思考..."
                if not partial_text:
                    chat_history[assistant_idx]["content"] = "正在思考..."
                else:
                    chat_history[assistant_idx]["content"] = partial_text
                yield "", chat_history, session_id

            # 流结束后前置路由标签
            if last_metadata and last_metadata.get("route"):
                route = Route(last_metadata["route"])
                label = ROUTE_LABELS.get(route, "")
                if label:
                    current = chat_history[assistant_idx]["content"]
                    chat_history[assistant_idx]["content"] = f"<small>{label}</small>\n\n{current}"
                    yield "", chat_history, session_id

        def _new_session_id():
            return f"gradio_{uuid.uuid4().hex[:12]}"

        # 每次页面加载/刷新时，生成全新 session_id（等效换用户）
        demo.load(lambda: _new_session_id(), None, session_state)

        msg.submit(respond, [msg, chatbot, session_state], [msg, chatbot, session_state])
        clear.click(
            lambda: (None, [], _new_session_id()),
            None,
            [chatbot, session_state],
            queue=False
        )

    return demo
