"""
车险智能客服 MVP - 主入口
提供 Gradio 交互界面 + /health 健康检查接口
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import gradio as gr
from dotenv import load_dotenv
from src.route_types import Route, ROUTE_LABELS
from src.error_types import ErrorCode, USER_ERROR_MESSAGES, DEFAULT_ERROR_MESSAGE
from langgraph.errors import GraphRecursionError
from src.constants import TOOL_FINISHED_PREFIX, TRANSFER_SIGNAL
from src.token_usage import add_tokens, get_today_usage, is_budget_exceeded
import jwt
from src.token_usage import add_tokens, get_today_usage, is_budget_exceeded, DAILY_TOKEN_LIMIT

load_dotenv()

app = FastAPI(title="车险智能客服 MVP", version="0.1.0")

# ---------- Token 验证配置 ----------
ACCESS_TOKEN_SECRET = os.environ.get("ACCESS_TOKEN_SECRET", "your-secret-key-change-me-in-production")

# ---------- Token 验证中间件 ----------
@app.middleware("http")
async def verify_token(request: Request, call_next):
    path = request.url.path

    # 放行健康检查、根路径、清单、favicon、Token 查询接口
    if path in ("/health", "/", "/manifest.json", "/favicon.ico", "/queryToken", "/docs"):
        return await call_next(request)

        # 放行 OpenAPI JSON（FastAPI 文档依赖）
    if path == "/openapi.json":
        return await call_next(request)

    # 放行所有以 /gradio/ 开头的子资源（静态文件、API 等）
    if path.startswith("/gradio/"):
        return await call_next(request)

    # /gradio 页面本身需要 token
    if path == "/gradio":
        token = request.query_params.get("token")
        if not token:
            return JSONResponse(status_code=401, content={"detail": "Missing token"})
        try:
            jwt.decode(token, ACCESS_TOKEN_SECRET, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return JSONResponse(status_code=401, content={"detail": "Token expired"})
        except jwt.InvalidTokenError:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})
        return await call_next(request)

    # 其他路径（如 /docs, /openapi.json 等）也需要 token
    token = request.query_params.get("token")
    if not token:
        return JSONResponse(status_code=401, content={"detail": "Missing token"})
    try:
        jwt.decode(token, ACCESS_TOKEN_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return JSONResponse(status_code=401, content={"detail": "Token expired"})
    except jwt.InvalidTokenError:
        return JSONResponse(status_code=401, content={"detail": "Invalid token"})

    return await call_next(request)

# ---------- 全局变量（预加载时初始化） ----------
_general_chain = None
_agent_sale = None
_agent_service = None

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

@app.get("/queryToken")
async def query_token():
    """查询今日 Token 使用量和费用"""
    usage = get_today_usage()
    return {
        "date": usage["date"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "daily_token_limit": DAILY_TOKEN_LIMIT,
    }


# ============================================================
# 核心调用函数
# ============================================================
def chat_api(session_id: str, message: str) -> dict:
    from src.core.routing import decide_route

    # ---------- 输入长度校验 ----------
    if len(message) > 1000:
        return {
            "success": -1,
            "error_msg": USER_ERROR_MESSAGES.get(ErrorCode.INPUT_TOO_LONG, DEFAULT_ERROR_MESSAGE),
            "content": {}
        }

    # ---------- 每日 Token 限额检查 ----------
    if is_budget_exceeded():
        return {
            "success": -1,
            "error_msg": "今日 Token 配额已用完，请明天再试",
            "content": {}
        }

    timer = Timer()
    timer.start("总耗时")
    
    print(f"[DEBUG] 收到消息: {message[:30]}...", flush=True)
    
    timer.start("路由决策")
    route = decide_route(session_id, message)
    timer.stop("路由决策")
    print(f"[路由决策] session={session_id}, message={message[:20]}..., route={route}", flush=True)
    
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}

    def _extract_usage(result_dict):
        """从 LangGraph 结果中提取 usage_metadata"""
        # 尝试顶层
        if "usage_metadata" in result_dict:
            return result_dict["usage_metadata"]
        # 尝试从最后一条消息
        if "messages" in result_dict and result_dict["messages"]:
            last_msg = result_dict["messages"][-1]
            if hasattr(last_msg, "usage_metadata"):
                return last_msg.usage_metadata
            if isinstance(last_msg, dict) and "usage_metadata" in last_msg:
                return last_msg["usage_metadata"]
        return {}

    def check_transfer_flag(result_dict):
        messages = result_dict.get("messages", [])
        for msg in messages:
            if hasattr(msg, 'content') and TRANSFER_SIGNAL in str(msg.content):
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
        result = None
        
        if route == Route.GENERAL:
            result = _general_chain.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config=config
            )
            last_msg = result["messages"][-1]
            reply = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            transfer_flag = False
        
        elif route == Route.SALE:
            result = _agent_sale.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config=config
            )
            last_msg = result["messages"][-1]
            reply = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            transfer_flag = check_transfer_flag(result)
            # 检查工具终止信号
            if reply.startswith("TOOL_FINISHED:"):
                clean_reply = reply.replace("TOOL_FINISHED:", "").strip()
                usage = _extract_usage(result)
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cached_tokens = usage.get("input_token_details", {}).get("cache_read", 0)
                add_tokens(input_tokens, output_tokens)
                import json
                log_entry = {
                    "timestamp": time.time(),
                    "session_id": session_id,
                    "route": route.value if hasattr(route, 'value') else str(route),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "daily_usage": get_today_usage()
                }
                print(f"[Token日志] {json.dumps(log_entry, ensure_ascii=False)}")
                timer.stop("Agent 调用")
                timer.stop("总耗时")
                print(f"\n[性能日志] 总耗时: {timer.get_total_ms():.0f}ms")
                print(timer.get_report())
                print("-" * 40)
                return {
                    "success": 0,
                    "content": {
                        "reply": clean_reply or "暂未找到相关信息，建议转人工咨询。",
                        "transfer": False,
                    },
                    "route": route,
                    "elapsed_ms": timer.get_total_ms(),
                }
    
        elif route == Route.SERVICE:
            result = _agent_service.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config=config
            )
            last_msg = result["messages"][-1]
            reply = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            transfer_flag = check_transfer_flag(result)
        
        else:
            return {"success": -1, "error_msg": DEFAULT_ERROR_MESSAGE, "content": {}}
        
        timer.stop("Agent 调用")
        timer.stop("总耗时")
        
        # ---------- 提取 Token 使用量 ----------
        usage = _extract_usage(result) if result is not None else {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cached_tokens = usage.get("input_token_details", {}).get("cache_read", 0)
        add_tokens(input_tokens, output_tokens)
        
        # 打印性能日志
        print(f"\n[性能日志] 总耗时: {timer.get_total_ms():.0f}ms")
        print(timer.get_report())
        print("-" * 40)
        
        # ---------- 打印 Token JSON 日志 ----------
        import json
        log_entry = {
            "timestamp": time.time(),
            "session_id": session_id,
            "route": route.value if hasattr(route, 'value') else str(route),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": input_tokens + output_tokens,
            "daily_usage": get_today_usage()
        }
        print(f"[Token日志] {json.dumps(log_entry, ensure_ascii=False)}")
        
        # ---------- 处理转人工 ----------
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
        
        # ---------- 正常返回 ----------
        return {
            "success": 0,
            "content": {
                "reply": reply,
                "transfer": False
            },
            "route": route,
            "elapsed_ms": timer.get_total_ms()
        }
    
    except GraphRecursionError as e:
        print(f"[ERROR] GraphRecursionError: {e}", flush=True)
        return {
            "success": -1,
            "error_msg": USER_ERROR_MESSAGES.get(ErrorCode.GRAPH_RECURSION_LIMIT, DEFAULT_ERROR_MESSAGE),
            "content": {}
        }
    
    except Exception as e:
        print(f"[ERROR] chat_api 异常: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {
            "success": -1,
            "error_msg": USER_ERROR_MESSAGES.get(ErrorCode.UNKNOWN, DEFAULT_ERROR_MESSAGE),
            "content": {}
        }


# ============================================================
# Gradio 界面
# ============================================================
def create_gradio_interface():
    with gr.Blocks(title="车险智能客服 MVP") as demo:
        gr.Markdown("# 🚗 车险智能客服 MVP")
        gr.Markdown("基于路由决策 + LangChain 构建")

        # 用 State 保存 session_id
        session_state = gr.State(value=f"gradio_{int(time.time())}")
        chatbot = gr.Chatbot(label="对话窗口")
        msg = gr.Textbox(label="输入消息", placeholder="请输入你的问题...")
        clear = gr.Button("清空对话")

        # 核心响应函数：先显示状态，再显示回复
        def respond(message, chat_history, session_id):
            if not message:
                return "", chat_history, session_id

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
        # 初始化 Chain，获取三个 Agent 和 Memory
        _general_chain, _agent_sale, _agent_service, _ = init_chains()
        # 初始化 RAG（加载 FAISS 索引和模型）
        init_rag()
        print("✅ 预加载完成")
    except Exception as e:
        print(f"⚠️ 预加载失败: {e}")
        print("   服务仍会启动，但第一条消息可能较慢")
        # 如果预加载失败，设置默认值避免 None 引用错误
        _general_chain = None
        _agent_sale = None
        _agent_service = None
    
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