"""
Gradio 交互界面
提供 Web UI 供用户与智能客服对话。
包含面试演示快捷面板。
"""

import uuid
import gradio as gr
from src.chat import chat_api_stream
from src.route_types import Route, ROUTE_LABELS
from src.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 演示话术定义（三栏布局：左-路由与记忆 / 中-RAG与兜底 / 右-边界防护）
# 格式：每栏 list of rows，每行 list of (按钮文字, 完整话术)
#       同行多按钮用 → 文字表示连续操作
# ============================================================
ROUTE_MEMORY = [
    [("你好", "你好")],
    [("我叫张三", "我叫张三，身份证 110101199001011234"), ("→ 叫什么名", "我叫什么名字")],
    [("帮我算保费", "帮我算一下特斯拉的保费"), ("→ 37岁特斯拉Y", "37岁，特斯拉Y，驾龄5年")],
    [("我改变主意，查保单", "我改变主意了，不计算保费了，帮我查一下保单，保单号 POL20260001")],
]

RAG_FALLBACK = [
    [("车损险保障哪些？", "车损险保障哪些情况？")],
    [("太空飞船赔不赔？", "太空飞船的保险赔不赔？")],
    [("我要投诉，转人工", "我要投诉，转人工")],
]

BOUNDARY = [
    [("测×1001", "测" * 1001)],
]


def create_gradio_interface():
    with gr.Blocks(title="车险智能客服 MVP") as demo:
        gr.Markdown("# 🚗 车险智能客服 MVP")
        gr.Markdown("基于路由决策 + LangChain 构建")
        
        # Token 有效期显示
        token_expiry_md = gr.HTML('<div id="token-expiry" style="display:none; color:#666; font-size:0.9em;"></div>', visible=True)

        # 状态
        session_state = gr.State(value="")
        chatbot = gr.Chatbot(label="对话窗口", height=400)
        
        # 输入区域
        with gr.Row():
            msg = gr.Textbox(
                label="输入消息", 
                placeholder="请输入你的问题...",
                scale=5,
                interactive=True,
                elem_id="msg-input",
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)
        
        # 操作按钮
        with gr.Row():
            clear_btn = gr.Button("🗑️ 清空对话", size="sm")
            reset_btn = gr.Button("🔄 重置会话（新 Session）", variant="stop", size="sm")

        gr.Markdown("---")
        gr.Markdown("### 📋 演示快捷面板")

        all_demo_btns = []  # 收集所有按钮用于事件绑定

        with gr.Row():
            # 左栏：路由与记忆
            with gr.Column(scale=1):
                gr.Markdown("**路由与记忆**")
                for ri, row in enumerate(ROUTE_MEMORY):
                    with gr.Row():
                        for ci, (label, full_msg) in enumerate(row):
                            btn = gr.Button(label, size="sm", variant="secondary",
                                          elem_id=f"demo_rm_{ri}_{ci}")
                            all_demo_btns.append((btn, full_msg))

            # 中栏：RAG与兜底
            with gr.Column(scale=1):
                gr.Markdown("**RAG与兜底**")
                for ri, row in enumerate(RAG_FALLBACK):
                    with gr.Row():
                        for ci, (label, full_msg) in enumerate(row):
                            btn = gr.Button(label, size="sm", variant="secondary",
                                          elem_id=f"demo_rf_{ri}_{ci}")
                            all_demo_btns.append((btn, full_msg))

            # 右栏：边界防护
            with gr.Column(scale=1):
                gr.Markdown("**边界防护**")
                for ri, row in enumerate(BOUNDARY):
                    with gr.Row():
                        for ci, (label, full_msg) in enumerate(row):
                            btn = gr.Button(label, size="sm", variant="secondary",
                                          elem_id=f"demo_bd_{ri}_{ci}")
                            all_demo_btns.append((btn, full_msg))

        # ============================================================
        # 核心响应函数
        # ============================================================
        async def respond(message, chat_history, session_id):
            """流式响应，AI 思考期间锁定输入"""
            # 空输入保护
            if not message or not message.strip():
                yield message, chat_history, session_id, gr.update(interactive=True), gr.update(interactive=True)
                return

            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": "正在思考..."})
            
            # 锁定输入
            yield "", chat_history, session_id, gr.update(interactive=False), gr.update(interactive=False)

            assistant_idx = len(chat_history) - 1
            last_metadata = None
            
            async for partial_text, metadata in chat_api_stream(session_id, message):
                if metadata and metadata.get("error"):
                    chat_history[assistant_idx]["content"] = f"❌ {partial_text}"
                    yield "", chat_history, session_id, gr.update(interactive=True), gr.update(interactive=True)
                    return
                if metadata:
                    last_metadata = metadata
                if not partial_text:
                    chat_history[assistant_idx]["content"] = "正在思考..."
                else:
                    chat_history[assistant_idx]["content"] = partial_text
                yield "", chat_history, session_id, gr.update(interactive=False), gr.update(interactive=False)

            # 流结束后前置路由标签
            if last_metadata and last_metadata.get("route"):
                route = Route(last_metadata["route"])
                label = ROUTE_LABELS.get(route, "")
                if label:
                    current = chat_history[assistant_idx]["content"]
                    chat_history[assistant_idx]["content"] = f"<small>{label}</small>\n\n{current}"

            # 解锁输入
            yield "", chat_history, session_id, gr.update(interactive=True), gr.update(interactive=True)

        def _new_session_id():
            return f"gradio_{uuid.uuid4().hex[:12]}"

        def reset_session(chat_history):
            """重置会话：清空对话 + 新 session_id"""
            return [], _new_session_id(), gr.update(interactive=True), gr.update(interactive=True)

        def make_demo_handler(full_message):
            """生成演示按钮的处理函数"""
            async def handler(chat_history, session_id):
                async for item in respond(full_message, chat_history, session_id):
                    yield item
            return handler

        # ============================================================
        # 事件绑定
        # ============================================================
        
        # 页面加载时生成 session_id
        demo.load(_new_session_id, None, session_state)

        # 注入 tooltip JavaScript 和 token 有效期显示
        tooltip_js = ""
        # Token 有效期显示
        tooltip_js += """
        (function(){
            var params = new URLSearchParams(window.location.search);
            var token = params.get('token');
            if(token){
                try{
                    var parts = token.split('.');
                    var payload = JSON.parse(atob(parts[1]));
                    if(payload.exp){
                        var expDate = new Date(payload.exp * 1000);
                        var expStr = expDate.toLocaleString('zh-CN', {year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit'});
                        var md = document.querySelector('#token-expiry');
                        if(md){
                            md.innerHTML = '🔑 Token 有效期至: ' + expStr;
                            md.style.display = 'block';
                        }
                    }
                }catch(e){console.error('Token parse error:', e);}
            }
        })();
        """
        for btn, full_msg in all_demo_btns:
            elem_id = btn.elem_id
            if elem_id:
                escaped = full_msg.replace('\\', '\\\\').replace('"', '\\"')[:80]
                tooltip_js += f'var b=document.getElementById("{elem_id}");if(b)b.title="{escaped}";'
        gr.HTML("<!-- tooltip -->", visible=True, js_on_load=tooltip_js)

        # 自动聚焦 JS：响应结束后自动聚焦输入框
        _autofocus_js = """() => {
            setTimeout(() => {
                const el = document.getElementById('msg-input');
                if (el) {
                    const ta = el.querySelector('textarea');
                    if (ta) ta.focus();
                }
            }, 100);
        }"""

        # 输入框提交
        msg.submit(
            respond, 
            [msg, chatbot, session_state], 
            [msg, chatbot, session_state, msg, send_btn]
        ).then(js=_autofocus_js)
        send_btn.click(
            respond,
            [msg, chatbot, session_state],
            [msg, chatbot, session_state, msg, send_btn]
        ).then(js=_autofocus_js)

        # 清空对话（保留 session）
        clear_btn.click(
            lambda: ([], gr.update(interactive=True), gr.update(interactive=True)),
            None,
            [chatbot, msg, send_btn],
            queue=False
        )

        # 重置会话（新 session）
        reset_btn.click(
            reset_session,
            [chatbot],
            [chatbot, session_state, msg, send_btn],
            queue=False
        )

        # 演示快捷面板按钮
        for btn, full_msg in all_demo_btns:
            btn.click(
                make_demo_handler(full_msg),
                [chatbot, session_state],
                [msg, chatbot, session_state, msg, send_btn]
            )

    return demo
