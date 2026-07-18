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
# 演示话术定义
# ============================================================
# 组A：共享 session（验证记忆 + route 切换）
# 格式：(按钮短文字, 完整话术, tooltip全文)
GROUP_A = [
    ("1. 你好", "你好", "你好"),
    ("2. 我叫张三...", "我叫张三，身份证 110101199001011234", "我叫张三，身份证 110101199001011234"),
    ("2. 我叫什么名...", "我叫什么名字", "我叫什么名字"),
    ("3. 帮我算一下...", "帮我算一下特斯拉的保费", "帮我算一下特斯拉的保费"),
    ("3. 37岁特斯拉Y...", "37岁，特斯拉Y，驾龄5年", "37岁，特斯拉Y，驾龄5年"),
    ("4. 我改变主意...", "我改变主意了，不计算保费了，帮我查一下保单，保单号 POL20260001", "我改变主意了，不计算保费了，帮我查一下保单，保单号 POL20260001"),
]

# 独立场景：每个按钮点击前建议重置 session
INDEPENDENT_SCENARIOS = [
    ("5. 车损险保障哪...", "车损险保障哪些情况？", "车损险保障哪些情况？"),
    ("6. 太空飞船的保...", "太空飞船的保险赔不赔？", "太空飞船的保险赔不赔？"),
    ("7. 我要投诉转...", "我要投诉，转人工", "我要投诉，转人工"),
    ("8. 测×1001", "测" * 1001, "输入超长测试（1001个字符）"),
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
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)
        
        # 操作按钮
        with gr.Row():
            clear_btn = gr.Button("🗑️ 清空对话", size="sm")
            reset_btn = gr.Button("🔄 重置会话（新 Session）", variant="stop", size="sm")

        gr.Markdown("---")
        gr.Markdown("### 📋 演示快捷面板")
        
        # 组A：记忆 + 路由切换
        gr.Markdown("**组A：记忆 + 路由切换**（连续点击，共享同一 Session）")
        with gr.Row():
            group_a_btns = []
            for i, (label, _, tooltip) in enumerate(GROUP_A):
                btn = gr.Button(
                    label, size="sm", variant="secondary",
                    elem_id=f"demo_a_{i}"
                )
                group_a_btns.append((btn, tooltip))
        
        # 独立场景
        gr.Markdown("**独立场景**（建议先点「重置会话」）")
        with gr.Row():
            indep_btns = []
            for i, (label, _, tooltip) in enumerate(INDEPENDENT_SCENARIOS):
                btn = gr.Button(
                    label, size="sm", variant="secondary",
                    elem_id=f"demo_i_{i}"
                )
                indep_btns.append((btn, tooltip))

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
        for i, (_, _, tooltip) in enumerate(GROUP_A):
            escaped = tooltip.replace('\\', '\\\\').replace('"', '\\"')
            tooltip_js += f'var b=document.getElementById("demo_a_{i}");if(b)b.title="{escaped}";'
        for i, (_, _, tooltip) in enumerate(INDEPENDENT_SCENARIOS):
            escaped = tooltip.replace('\\', '\\\\').replace('"', '\\"')
            tooltip_js += f'var b=document.getElementById("demo_i_{i}");if(b)b.title="{escaped}";'
        gr.HTML("<!-- tooltip -->", visible=True, js_on_load=tooltip_js)

        # 输入框提交
        msg.submit(
            respond, 
            [msg, chatbot, session_state], 
            [msg, chatbot, session_state, msg, send_btn]
        )
        send_btn.click(
            respond,
            [msg, chatbot, session_state],
            [msg, chatbot, session_state, msg, send_btn]
        )

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

        # 组A 演示按钮
        for (btn, _), (_, full_msg, _) in zip(group_a_btns, GROUP_A):
            btn.click(
                make_demo_handler(full_msg),
                [chatbot, session_state],
                [msg, chatbot, session_state, msg, send_btn]
            )

        # 独立场景按钮
        for (btn, _), (_, full_msg, _) in zip(indep_btns, INDEPENDENT_SCENARIOS):
            btn.click(
                make_demo_handler(full_msg),
                [chatbot, session_state],
                [msg, chatbot, session_state, msg, send_btn]
            )

    return demo
