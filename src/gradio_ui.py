"""
Gradio 交互界面
提供 Web UI 供用户与智能客服对话。
"""

import uuid
import gradio as gr
from src.route_types import ROUTE_LABELS
from src.error_types import DEFAULT_ERROR_MESSAGE
from src.chat import chat_api
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

        # 核心响应函数：先显示状态，再显示回复
        def respond(message, chat_history, session_id):
            if not message or not message.strip():
                yield "", chat_history, session_id
                return

            # 第一步：立即显示"正在理解问题..."
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": "🤔 正在理解问题..."})
            yield "", chat_history, session_id

            # 第二步：调用 chat_api，获取结果
            result = chat_api(session_id, message)

            # 第三步：更新为最终回复
            if result["success"] != 0:
                final_reply = f"❌ {result.get('error_msg', DEFAULT_ERROR_MESSAGE)}"
            else:
                route = result.get("route", "unknown")
                route_label = ROUTE_LABELS.get(route, f"🔀 {route}")
                elapsed_ms = result.get("elapsed_ms", 0)
                elapsed_str = f"⏱️ {elapsed_ms/1000:.1f}s" if elapsed_ms > 0 else ""
                final_reply = result["content"]["reply"]
                if result["content"].get("transfer", False):
                    ticket_id = result["content"].get("ticket_id", "")
                    final_reply += f"\n\n🔄 已为您转接人工客服，工单号：{ticket_id}"
                # 加上路由标签和耗时
                final_reply = f"<small>{route_label}  {elapsed_str}</small>\n\n{final_reply}"

            # 替换最后一条 assistant 消息
            chat_history[-1] = {"role": "assistant", "content": final_reply}
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
