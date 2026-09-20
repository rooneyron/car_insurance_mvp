# -*- coding: utf-8 -*-
"""飞书机器人长连接最小可运行脚本 —— 链路验证：手机飞书发消息，机器人原样回复。

【运行步骤】
  1. 配置凭证（脚本用 load_dotenv 读项目根 .env；也可用系统环境变量，均不硬编码）：
       方式A（推荐）：在 .env 文件里加两行
           FEISHU_APP_ID=cli_xxxxxxxx
           FEISHU_APP_SECRET=xxxxxxxxxxxxxxxx
       方式B：设系统环境变量
           PowerShell:  $env:FEISHU_APP_ID="cli_xxx"; $env:FEISHU_APP_SECRET="xxx"
           Linux/macOS: export FEISHU_APP_ID=cli_xxx FEISHU_APP_SECRET=xxx
  2. 运行：
           python feishu_ws_echo.py

【飞书开发者后台前置配置】（缺一步都收不到/发不出消息）
  1. 「事件与回调」→ 订阅方式选「使用长连接接收事件」；
  2. 添加事件「接收消息 im.message.receive_v1」；
  3. 「权限管理」开通发消息权限（如 im:message:send_as_bot）与读取消息权限；
  4. 「版本管理与发布」创建版本并发布，使上述配置生效；
  5. 把机器人加入某个群，或与机器人单聊。

依赖：pip install lark-oapi
"""
import os
import sys
import json
import time
from pathlib import Path

from dotenv import load_dotenv

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    P2ImMessageReceiveV1,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

# 先加载项目根 .env（凭证配在这里），再读环境变量；显式用脚本所在目录定位 .env，
# 避免受运行时工作目录影响。系统环境变量若已设则优先（load_dotenv 默认不覆盖）。
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# 凭证从环境变量读取（值来自 .env 或系统环境变量，不硬编码）
APP_ID = os.environ.get("FEISHU_APP_ID")
APP_SECRET = os.environ.get("FEISHU_APP_SECRET")

# 发消息（回复）用的普通 API Client，在 main() 校验环境变量后初始化；
# tenant_access_token 的获取与过期刷新由 SDK 内部完成，业务代码无需关心。
_api_client = None


def _reply_message(message_id: str, text: str) -> None:
    """回复指定消息（message_id）一条文本消息。失败抛异常，由调用方 try/except 兜底。

    用「回复消息」API 而非「发送消息」：message_id 已定位会话与用户，
    无需 receive_id / receive_id_type，校验限制比主动发送更少。
    """
    # content 必须是 JSON 字符串；ensure_ascii=False 保证中文不被转义
    content = json.dumps({"text": text}, ensure_ascii=False)
    # 发送前打印消息内容，便于排查发送失败问题
    print(f"[回复消息] message_id={message_id}, msg_type=text, content={content}")
    body = (
        ReplyMessageRequestBody.builder()
        .msg_type("text")
        .content(content)
        .build()
    )
    request = (
        ReplyMessageRequest.builder()
        .message_id(message_id)   # message_id 是路径参数，定位被回复的会话
        .request_body(body)
        .build()
    )
    response = _api_client.im.v1.message.reply(request)
    if not response.success():
        raise RuntimeError(f"回复失败: code={response.code}, msg={response.msg}")


def do_p2_im_message_receive_v1(data: P2ImMessageReceiveV1) -> None:
    """接收消息事件处理器：只处理文本消息，原样回复「收到：<原文>」。

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
        message_id = message.message_id   # om_ 开头，用于回复这条消息
        # message.content 是 JSON 字符串，形如 {"text":"你好"}
        text = json.loads(message.content).get("text", "")
        print(f"[收到消息] open_id={open_id}, message_id={message_id}, 内容={text}")

        _reply_message(message_id, "收到：" + text)
        print(f"[回复成功] message_id={message_id}")
    except Exception as e:
        # 打印错误但不崩溃（长连接保持运行）
        print(f"[处理异常] {type(e).__name__}: {e}")


def main() -> None:
    global _api_client
    if not APP_ID or not APP_SECRET:
        print("错误：未设置环境变量 FEISHU_APP_ID / FEISHU_APP_SECRET，请先设置后重试。")
        sys.exit(1)

    _api_client = lark.Client.builder().app_id(APP_ID).app_secret(APP_SECRET).build()

    # 注册事件处理器：长连接模式下 builder 的两个参数
    # （encrypt_key, verification_token）必须填空字符串。
    event_handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(do_p2_im_message_receive_v1)
        .build()
    )

    print("飞书长连接已启动，等待消息...")
    # ws.Client.start() 阻塞运行，SDK 内部已实现断线自动重连；
    # 外层再包一层循环兜底：若 start() 整体异常退出，5 秒后重建连接。
    # 调试字段结构时可把 log_level 改成 lark.LogLevel.DEBUG 查看完整事件体。
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
            print("\n[停止] 收到 Ctrl+C，退出长连接。")
            break
        except Exception as e:
            print(f"[长连接异常] {type(e).__name__}: {e}，5 秒后重连...")
            time.sleep(5)


if __name__ == "__main__":
    main()
