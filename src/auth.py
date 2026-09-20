"""
极简用户标识（user_id）校验与规范化。

设计目标：为后续「长期记忆」功能提供统一的用户身份入口。
- 不涉及账号密码、注册、数据库存储；用户直接输入一个标识字符串即可。
- 登录相关逻辑集中于本模块，避免散落到各业务文件。

user_id 贯穿链路：Gradio 界面 → chat_api(_stream) → 图 input_data/state → 各节点读取。
未登录时 user_id 为空字符串，所有下游功能保持完全兼容（不报错、不拦截）。
"""

import re

# 只保留：数字、字母、下划线、中文（CJK 统一表意文字 \u4e00-\u9fff）；其余字符一律过滤
_ILLEGAL_CHARS = re.compile(r"[^0-9A-Za-z_\u4e00-\u9fff]")

# user_id 最大长度（超出部分截断）
MAX_USER_ID_LEN = 64


def validate_user_id(user_id: str) -> str:
    """校验并规范化用户标识 user_id。

    规则（按顺序执行）：
    1. 空 / None → 返回 ""
    2. strip 去首尾空格
    3. 只保留字母、数字、下划线、中文，过滤其他所有字符
    4. 最长 MAX_USER_ID_LEN(64) 字符，超出截断

    示例：
        "张 三!@#"    → "张三"
        "  alice_01 " → "alice_01"
        "a" * 200     → "a" * 64
        ""  / None    → ""
    """
    if not user_id:
        return ""
    cleaned = user_id.strip()
    if not cleaned:
        return ""
    # 过滤非法字符（保留字母/数字/下划线/中文）
    cleaned = _ILLEGAL_CHARS.sub("", cleaned)
    # 长度截断
    return cleaned[:MAX_USER_ID_LEN]
