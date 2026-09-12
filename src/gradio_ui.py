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
from src.auth import validate_user_id

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
        # ============================================================
        # 顶部：左侧标题 + 右侧紧凑登录区（右上角内联）
        # 登录区组件「始终可见」，仅切换 value/interactive，规避 Gradio 6.x
        # 的 visible=False→True 首次不渲染缺陷（issue #12511）。
        # ============================================================
        with gr.Row():
            # 左：标题 + Token 有效期
            with gr.Column(scale=3):
                gr.Markdown("# 🚗 车险智能客服 MVP")
                gr.Markdown("基于路由决策 + LangChain 构建")
                token_expiry_md = gr.HTML('<div id="token-expiry" style="display:none; color:#666; font-size:0.9em;"></div>', visible=True)
            # 右：用户标识登录区（极简登录，为后续长期记忆功能预留 user_id）
            with gr.Column(scale=2):
                user_status_md = gr.Markdown("🔓 未登录（可选，用于保存对话记忆）")
                with gr.Row():
                    user_id_input = gr.Textbox(
                        label="用户标识",
                        placeholder="字母/数字/下划线/中文",
                        scale=2,
                    )
                    # 单按钮两用：未登录=登录、已登录=退出（on_user_action 按当前状态判断）
                    user_action_btn = gr.Button("登录", variant="primary", scale=1, size="sm")
        # 浏览器端持久化组件（Gradio 6.x 用 storage_key，非 5.x 的 key）；刷新页面不丢失
        user_id_state = gr.BrowserState(default_value="", storage_key="user_id")

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
        async def respond(message, chat_history, session_id, user_id):
            """流式响应，AI 思考期间锁定输入。user_id 为当前登录用户标识（空串=未登录）"""
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
            lines = []              # 已定型展示行：[("thinking"/"tool"/"answer", text)]，多轮思考全部保留
            cur_stream = ""         # 当前 agent 轮正在流式的文本（角色待定）
            
            async for partial_text, metadata in chat_api_stream(session_id, message, user_id):
                if metadata and metadata.get("error"):
                    chat_history[assistant_idx]["content"] = f"❌ {partial_text}"
                    yield "", chat_history, session_id, gr.update(interactive=True), gr.update(interactive=True)
                    return
                role = metadata.get("role") if metadata else None
                tool_status = metadata.get("tool_status") if metadata else None
                if metadata:
                    last_metadata = metadata

                if role == "streaming":
                    # 当前轮 content 流式累积（partial_text 即该轮文本）
                    cur_stream = partial_text or ""
                elif role == "thinking":
                    # on_tool_start：当前轮定性为思考 → 定型为 💭 行，再追加工具状态行
                    if cur_stream:
                        lines.append(("thinking", cur_stream))
                        cur_stream = ""
                    if tool_status:
                        lines.append(("tool", "🔧 " + tool_status))
                elif role == "tool":
                    # on_tool_end：把最近的工具状态行更新为"调用完成"
                    if tool_status:
                        if lines and lines[-1][0] == "tool":
                            lines[-1] = ("tool", "🔧 " + tool_status)
                        else:
                            lines.append(("tool", "🔧 " + tool_status))
                elif role == "answer":
                    # 收尾：最后一轮定性为最终答案
                    final = partial_text or cur_stream
                    if final:
                        lines.append(("answer", final))
                    cur_stream = ""

                # 渲染：已定型行（思考加 💭）+ 进行中的当前轮
                parts = [("💭 " + text) if typ == "thinking" else text for typ, text in lines]
                if cur_stream:
                    parts.append(cur_stream)
                chat_history[assistant_idx]["content"] = "\n\n".join(parts) if parts else "正在思考..."
                yield "", chat_history, session_id, gr.update(interactive=False), gr.update(interactive=False)

            # 流结束后前置路由标签
            if last_metadata and last_metadata.get("route"):
                route = Route(last_metadata["route"])
                label = ROUTE_LABELS.get(route, "")
                # 显示决策层 + 分数
                source = last_metadata.get("router_source", "")
                confidence = last_metadata.get("router_confidence", 0.0)
                layer_map = {
                    "l0_safety": "L0",
                    "l1_keyword": "L1",
                    "l2": "L2",
                    "l3": "L3",
                    "l4_clarify": "L4",
                    "l4_handoff": "L4",
                }
                layer = layer_map.get(source, "")
                if layer:
                    label = f"{label} ({layer}, {confidence:.2f})"
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

        # ============================================================
        # 用户标识登录：加载 / 登录 / 退出（单按钮两用）
        # 状态文字/按钮始终可见（切 value/variant）；输入框登录后隐藏、退出后显示。
        # 一律返回组件实例（非 gr.update），确保属性更新可靠。
        # ============================================================
        def _login_ui(user_id: str, error: str = ""):
            """按登录态返回 [状态文字, 输入框, 操作按钮] 的更新。
            登录后隐藏输入框（含其 label），只留状态文字 + 退出按钮；未登录反之。
            error 非空时状态文字显示错误提示（代替 Toast，避免 Toast 遮挡按钮致点击失效）。"""
            if user_id:
                return (
                    gr.Markdown(f"👤 当前用户：**{user_id}**"),
                    gr.Textbox(visible=False),
                    gr.Button("退出", variant="stop"),
                )
            status = f"⚠️ {error}" if error else "🔓 未登录（可选，用于保存对话记忆）"
            return (
                gr.Markdown(status),
                gr.Textbox(value="", interactive=True, visible=True, placeholder="字母/数字/下划线/中文"),
                gr.Button("登录", variant="primary"),
            )

        def on_user_load(saved_user_id):
            """页面加载：从 BrowserState 恢复登录态显示（刷新不丢失）；仅刷新显示，不改 state。"""
            return _login_ui(validate_user_id(saved_user_id or ""))

        def on_user_action(raw_input, current_user_id):
            """单按钮两用：未登录→登录；已登录→退出并重置会话。

            返回 8 元组，按位置对应 _user_outputs：
              [0]   user_id_state（BrowserState 新值）
              [1:4] 登录态 UI —— _login_ui 产出的 [状态文字, 输入框, 操作按钮]
              [4:8] 对话区 —— chat_reset（退出：清空+新 session+解锁）或 chat_keep（登录：全 skip 不动历史）
            反馈一律走状态文字（不用 gr.Info/Warning），避免 Toast 遮挡按钮导致首次点击失效。"""
            # 对话区 4 项（chatbot, session_state, msg, send_btn）的两种取值，具名替代裸元组拼接
            chat_keep = (gr.skip(), gr.skip(), gr.skip(), gr.skip())   # 登录：保持不变，不误清历史
            chat_reset = ([], _new_session_id(),                        # 退出：清空对话 + 新 session_id
                          gr.update(interactive=True), gr.update(interactive=True))  # + 解锁输入/发送
            current = validate_user_id(current_user_id or "")
            if current:
                # 已登录 → 退出：清登录态（user_id_state=""，UI 复位）+ 重置会话
                return ("",) + _login_ui("") + chat_reset
            # 未登录 → 登录：只更新登录态，对话区保持不变
            uid = validate_user_id(raw_input)
            if not uid:
                login_ui = _login_ui("", error="用户标识无效：仅支持字母、数字、下划线、中文")
                return ("",) + login_ui + chat_keep
            return (uid,) + _login_ui(uid) + chat_keep

        def make_demo_handler(full_message):
            """生成演示按钮的处理函数"""
            async def handler(chat_history, session_id, user_id):
                async for item in respond(full_message, chat_history, session_id, user_id):
                    yield item
            return handler

        # ============================================================
        # 事件绑定
        # ============================================================
        
        # 页面加载时生成 session_id
        demo.load(_new_session_id, None, session_state)

        # 页面加载时从浏览器 localStorage 恢复用户标识登录态显示（刷新不丢失）
        demo.load(
            on_user_load,
            [user_id_state],
            [user_status_md, user_id_input, user_action_btn],
        )

        # 单按钮两用（登录/退出）+ 输入框回车登录
        # 退出时顺带重置会话，故 outputs 追加 [对话, session, 输入框, 发送按钮]（登录分支对这 4 项用 gr.skip 跳过）
        _user_outputs = [user_id_state, user_status_md, user_id_input, user_action_btn, chatbot, session_state, msg, send_btn]
        user_action_btn.click(on_user_action, [user_id_input, user_id_state], _user_outputs)
        user_id_input.submit(on_user_action, [user_id_input, user_id_state], _user_outputs)

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

        # 输入框提交（user_id_state 作为输入，把当前登录用户标识传给后端）
        msg.submit(
            respond, 
            [msg, chatbot, session_state, user_id_state], 
            [msg, chatbot, session_state, msg, send_btn]
        ).then(js=_autofocus_js)
        send_btn.click(
            respond,
            [msg, chatbot, session_state, user_id_state],
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
                [chatbot, session_state, user_id_state],
                [msg, chatbot, session_state, msg, send_btn]
            )

    return demo
