# -*- coding: utf-8 -*-
"""飞书机器人 —— in-process 直连车险核心(chat_api)，把 LLM 回复以 Markdown 卡片渲染回飞书。

【架构：in-process 方案】
  本模块由 app.py 在预加载完成后，用后台 daemon 线程启动（start_in_background）。
  与 Gradio 一样，飞书事件回调直接 import 并调用核心函数 chat_api（不走 HTTP、不需要 JWT），
  复用 app.py 已加载的 graph/RAG/记忆，省一份模型加载。

    手机飞书发消息
      -> 长连接事件 P2ImMessageReceiveV1（在 WS 后台线程里同步回调）
      -> chat_api(session_id=chat_id, message=text, user_id=open_id, stream=False)  <- 直连核心
      -> 解析 content.reply / transfer / ticket_id
      -> 构造 interactive 卡片 -> im.v1.message.reply(message_id, 卡片)

【运行】只需启动主服务：  python app.py     （API + Gradio + 飞书机器人 一起起）
  - 未配置 FEISHU_APP_ID/SECRET 时自动跳过飞书，不影响 API/Gradio。
  - 单独调试本模块：python feishu_bot.py（会自行初始化核心后启动长连接，较慢）。

【环境变量】（load_dotenv 读项目根 .env，也可用系统环境变量，均不硬编码）
  FEISHU_APP_ID / FEISHU_APP_SECRET   飞书应用凭证（缺失则跳过飞书机器人）

【飞书开发者后台前置配置】（缺一步都收不到/发不出消息）
  1.「事件与回调」订阅方式选「使用长连接接收事件」；
  2. 添加事件「接收消息 im.message.receive_v1」；
  3.「权限管理」开通读取消息 + 以应用身份发消息（im:message:send_as_bot）权限；
  4.「版本管理与发布」创建版本并发布使配置生效；
  5. 把机器人加入某个群，或与机器人单聊。

依赖：pip install lark-oapi python-dotenv
"""
import os
import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

import requests   # 飞书 SDK 底层依赖；显式 import 用于精准捕获网络层异常做重试
import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    P2ImMessageReceiveV1,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

# 保证可 import src.*（standalone 运行时用；被 app.py import 时 sys.path 通常已就绪，重复插入无害）
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 先加载项目根 .env（凭证配在这里），再读环境变量；系统环境变量若已设则优先。
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from src.logger import get_logger

logger = get_logger(__name__)

# 直连核心：与 Gradio(src/gradio_ui.py) 同款，import 核心对话函数。
# chat_api 是非流式入口，返回完整 dict：{success, content:{reply,transfer,ticket_id?}, route, elapsed_ms}
from src.chat import chat_api

# 消息去重（sqlite 持久化）：只读判重 + 回复成功后登记，防飞书重投/重连补推导致重复回复
from src.memory.dedup_store import is_duplicate_message, mark_message_processed

# ---------- 配置（全部来自环境变量，不硬编码任何密钥）----------
APP_ID = os.environ.get("FEISHU_APP_ID")
APP_SECRET = os.environ.get("FEISHU_APP_SECRET")

# 输入长度上限，与 src.constants.MAX_INPUT_LENGTH 对齐
MAX_INPUT_LEN = 1000

# 发消息（回复）用的普通 API Client，在 run_forever() 里初始化；
# tenant_access_token 的获取与过期刷新由 SDK 内部完成，业务代码无需关心。
_api_client = None

# 业务处理线程池：让事件 handler 快速返回（SDK 及时回写 ack，避免飞书因 ack 超时重投事件），
# 把耗时的 chat_api(LLM 多轮)+reply 交给线程池异步跑。max_workers 支持多会话并发；
# 同一会话(chat_id)的串行由 chat_api 内部 session 锁保证，故此处并发安全。
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="feishu-worker")


# ============================================================
# 1. 构造飞书 interactive 卡片（Markdown 渲染）
# ============================================================

def _build_card(reply_text: str, transfer: bool = False, ticket_id: str = "") -> str:
    """把回复文本包成飞书 interactive 卡片 JSON 字符串。

    用 {"tag":"markdown"} 元素渲染 **加粗**/emoji/换行/列表；
    若你的飞书客户端版本渲染异常，可把该元素换成
    {"tag":"div","text":{"tag":"lark_md","content": reply_text}}。
    """
    elements = [{"tag": "markdown", "content": reply_text or ""}]
    if transfer:
        tail = "已为您转接人工客服"
        if ticket_id:
            tail += f"（工单 {ticket_id}）"
        elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": tail})
    card = {"config": {"wide_screen_mode": True}, "elements": elements}
    return json.dumps(card, ensure_ascii=False)


# ============================================================
# 2. 回复消息（interactive 卡片）
# ============================================================

def _reply_card(message_id: str, card_str: str, max_retries: int = 2) -> None:
    """用「回复消息」API 回复一条 interactive 卡片。失败抛异常，由调用方兜底。

    用 reply 而非 create：message_id 已定位会话与用户，校验限制更少（echo 版已验证）。

    【网络抖动重试】飞书 SDK 底层是 requests + urllib3 连接池(keep-alive)。机器人消息不频繁时，
    池中连接长时间闲置会被服务端/NAT 静默关闭，下次复用这条「陈旧连接」发请求即收到
    ConnectionReset(10054) —— 表现为「第一次 reply 失败、兜底重发却成功」。故对网络层异常
    (requests.RequestException：连接重置/连接超时/读超时)自动重试 max_retries 次（重试会重建新
    连接，通常一次即救回）；业务错误(response.success()==False，如权限/卡片格式)重试无意义，直接抛。
    """
    body = (
        ReplyMessageRequestBody.builder()
        .msg_type("interactive")   # 卡片消息
        .content(card_str)         # content 必须是 JSON 字符串
        .build()
    )
    request = (
        ReplyMessageRequest.builder()
        .message_id(message_id)    # message_id 是路径参数，定位被回复的会话
        .request_body(body)
        .build()
    )
    for attempt in range(max_retries + 1):
        try:
            print(f"[reply] 调用飞书回复 API(第{attempt + 1}次): message_id={message_id}, "
                  f"msg_type=interactive, card_len={len(card_str)}")
            response = _api_client.im.v1.message.reply(request)
            # 无论成败都打印飞书 API 响应码，便于定位「能收消息但发不出回复」（发送权限/机器人能力/卡片格式）
            print(f"[reply] 飞书响应: success={response.success()}, code={response.code}, msg={response.msg}")
            if not response.success():
                # 业务错误（发送权限/机器人能力/卡片格式/message_id 失效）：重试也是同样结果，直接抛
                raise RuntimeError(f"回复失败: code={response.code}, msg={response.msg}")
            return   # 回复成功
        except requests.exceptions.RequestException as e:
            # 网络层异常（ConnectionReset 10054 / 连接超时 / 读超时 / 陈旧连接）：退避后重试可救回
            if attempt < max_retries:
                wait = 0.5 * (attempt + 1)
                print(f"[reply] 网络异常({type(e).__name__})，{wait}s 后重试({attempt + 1}/{max_retries}): {e}")
                time.sleep(wait)
                continue
            print(f"[reply] 网络异常，重试 {max_retries} 次仍失败，抛出: {type(e).__name__}: {e}")
            raise


# ============================================================
# 3. 接收消息事件处理器（直连 chat_api，in-process）
# ============================================================

def do_p2_im_message_receive_v1(data: P2ImMessageReceiveV1) -> None:
    """收到飞书消息事件 -> 轻量提取 + 只读判重 -> 提交线程池异步处理 -> 立即返回。

    【为何必须快速返回】lark SDK 的 _handle_data_frame 里「同步调用本 handler，且 handler
    返回后才回写 ack 帧」。若在此同步跑 chat_api(LLM 多轮，几十秒)，会阻塞 ack，飞书服务端
    ack 超时即按退避策略重投事件(实测间隔 3~6min)。故本函数只做提取+判重后立即返回让 SDK 秒 ack，
    耗时的 chat_api+reply+登记交给 _executor 线程池（见 _process_and_reply）。
    异常只打印、不抛出，保证单条消息处理失败不会中断长连接。
    """
    try:
        event = data.event
        message = event.message
        # 只处理文本消息，其他类型（图片/语音/富文本/表情包等）直接忽略
        if message.message_type != "text":
            print(f"[忽略] 非文本消息: message_type={message.message_type}")
            return

        open_id = event.sender.sender_id.open_id
        chat_id = message.chat_id          # oc_ 开头，会话标识（单聊/群各自独立）
        message_id = message.message_id    # om_ 开头，用于回复这条消息
        # message.content 是 JSON 字符串，形如 {"text":"你好"}
        text = json.loads(message.content).get("text", "").strip()

        # 只读判重：已成功回复过的消息，飞书重投/重连补推时直接跳过（不回复、不调 chat、不存记忆）。
        # 注意此处只查不登记——登记由 _process_and_reply 在 reply 成功后 mark_message_processed 完成，
        # 避免回复失败的消息被误标记、重投时跳过导致用户永远收不到回复。
        if is_duplicate_message(message_id):
            logger.info("重复消息，跳过: %s", message_id)
            return

        print(f"[收到消息] open_id={open_id}, chat_id={chat_id}, message_id={message_id}, 内容={text}")

        # 提交线程池异步处理，本函数立即返回 -> SDK 及时 ack -> 飞书不因超时重投
        session_id = chat_id or open_id    # 每会话独立多轮上下文；user_id=open_id（长期记忆主键）
        _executor.submit(_process_and_reply, message_id, text, session_id, open_id)
    except Exception as e:
        # 提取/判重阶段异常（此时可能拿不到 message_id，无法异步处理）；打印不抛出，保住长连接
        print(f"[handler异常] {type(e).__name__}: {e}")


def _process_and_reply(message_id: str, text: str, session_id: str, user_id: str) -> None:
    """线程池内异步执行：chat_api -> 卡片回复 -> 「reply 成功后」才登记去重。

    登记时机（关键）：只有 _reply_card 未抛异常(=飞书确认收到回复)才 mark_message_processed；
    reply 失败(网络重试后仍失败)或 chat_api 异常则不登记，飞书若重投/重连补推可重新处理，
    避免「回复失败却已登记 -> 重投被跳过 -> 用户永远收不到回复」。
    """
    try:
        if not text:
            _reply_card(message_id, _build_card("你好像发了条空消息，请输入你的问题～"))
            mark_message_processed(message_id)
            return

        # 输入截断，与核心侧 MAX_INPUT_LENGTH 对齐
        if len(text) > MAX_INPUT_LEN:
            print(f"[截断] 输入 {len(text)} 字超过 {MAX_INPUT_LEN}，已截断")
            text = text[:MAX_INPUT_LEN]

        # ---- 直连核心（in-process），非流式一次性拿完整回复 ----
        try:
            result = chat_api(session_id, text, user_id, stream=False)
        except Exception as e:
            print(f"[chat_api异常] {type(e).__name__}: {e}")
            _reply_card(message_id, _build_card("服务开小差了，请稍后再试。"))
            # chat_api 异常（多为偶发 LLM 超时）：不登记，飞书重投/重连可重试
            return

        # ---- 解析响应 ----
        if result.get("success") == 0:
            content = result.get("content") or {}
            reply_text = content.get("reply") or "(空回复)"
            transfer = bool(content.get("transfer"))
            ticket_id = content.get("ticket_id", "")
            print(f"[chat_api返回] success=0, route={result.get('route')}, "
                  f"transfer={transfer}, reply_len={len(reply_text)}, elapsed_ms={result.get('elapsed_ms')}")
            _reply_card(message_id, _build_card(reply_text, transfer, ticket_id))
            mark_message_processed(message_id)   # 【关键】reply 成功后才登记为已处理
            print(f"[回复成功] message_id={message_id}")
        else:
            # success == -1：输入空/超长、预算超限、session busy 等业务错误（确定性回复）
            err_msg = result.get("error_msg") or "服务开小差了，请稍后再试。"
            print(f"[业务错误] success={result.get('success')}, msg={err_msg}")
            _reply_card(message_id, _build_card(err_msg))
            mark_message_processed(message_id)   # 业务错误也是确定性回复，登记避免重投反复打扰
    except Exception as e:
        # reply 失败(网络重试后仍失败)等：兜底提示，但【不登记】-> 飞书重投/重连可重新处理
        print(f"[处理异常] {type(e).__name__}: {e}")
        try:
            _reply_card(message_id, _build_card("处理消息时出错了，请稍后再试。"))
        except Exception:
            pass


# ============================================================
# 4. 启动：阻塞运行 + 后台线程封装
# ============================================================

def run_forever() -> None:
    """阻塞式运行飞书长连接（含断线自动重连）。

    必须在「核心已初始化(state.graph 就绪)」的进程中调用——生产中由 app.py 预加载后
    在后台线程调用本函数；standalone 运行时先调 _init_core_standalone()。
    """
    global _api_client
    _api_client = lark.Client.builder().app_id(APP_ID).app_secret(APP_SECRET).build()

    # 注册事件处理器：长连接模式下 builder 的两个参数
    # （encrypt_key, verification_token）必须填空字符串。
    event_handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(do_p2_im_message_receive_v1)
        .build()
    )

    print("[飞书] 长连接启动中，等待消息...")
    # ws.Client.start() 阻塞运行，SDK 内部已实现断线自动重连；
    # 外层再包一层循环兜底：若 start() 整体异常退出，5 秒后重建连接。
    while True:
        try:
            ws_client = lark.ws.Client(
                APP_ID,
                APP_SECRET,
                event_handler=event_handler,
                log_level=lark.LogLevel.INFO,
            )
            ws_client.start()
        except KeyboardInterrupt:
            print("\n[飞书] 收到 Ctrl+C，退出长连接。")
            break
        except Exception as e:
            print(f"[飞书][长连接异常] {type(e).__name__}: {e}，5 秒后重连...")
            time.sleep(5)


def start_in_background() -> bool:
    """供 app.py 调用：在后台 daemon 线程启动飞书长连接（in-process 直连 chat_api）。

    - 缺少飞书凭证则跳过（返回 False），不影响 API/Gradio 主服务；
    - 线程内任何异常都被吞掉并打印，绝不冒泡影响 uvicorn 主进程。
    """
    if not APP_ID or not APP_SECRET:
        print("[飞书] 未配置 FEISHU_APP_ID / FEISHU_APP_SECRET，跳过飞书机器人（API/Gradio 不受影响）。")
        return False

    def _runner():
        try:
            run_forever()
        except Exception as e:
            print(f"[飞书] 后台线程异常退出: {type(e).__name__}: {e}")

    threading.Thread(target=_runner, name="feishu-bot", daemon=True).start()
    print("[飞书] 机器人已在后台线程启动（in-process 直连 chat_api，与 Gradio 同源）。")
    return True


def _init_core_standalone() -> None:
    """单独运行 feishu_bot.py 时初始化核心（chat_api 依赖 state.graph）。

    生产中由 app.py 预加载，无需调用本函数。这里做最小可用初始化：graph + RAG + db + 记忆存储。
    """
    print("[飞书][standalone] 初始化核心（graph/RAG/db）...")
    from src.chains.chains import init_graph
    from src.state import set_graph
    graph, _llm, _clf = init_graph()
    set_graph(graph)
    try:
        from src.rag import init_rag_components
        init_rag_components()
    except Exception as e:
        print(f"[飞书][standalone] RAG 初始化失败（一般问答仍可用）: {e}")
    try:
        from src.db import init_db
        init_db()
    except Exception as e:
        print(f"[飞书][standalone] init_db 失败（不影响启动）: {e}")
    try:
        from src.memory import structured_store  # noqa: F401  触发 sqlite 建表
    except Exception as e:
        print(f"[飞书][standalone] 记忆存储初始化失败（不影响启动）: {e}")
    print("[飞书][standalone] 核心初始化完成")


if __name__ == "__main__":
    # 独立运行模式：自行初始化核心后阻塞跑长连接。
    # 生产推荐由 app.py 的 start_in_background() 在后台线程启动，复用其预加载的核心。
    if not APP_ID or not APP_SECRET:
        print("错误：未设置环境变量 FEISHU_APP_ID / FEISHU_APP_SECRET，请先设置后重试。")
        sys.exit(1)
    _init_core_standalone()
    run_forever()
