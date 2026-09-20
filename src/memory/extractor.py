# -*- coding: utf-8 -*-
"""长期记忆提取器：用 LLM 从一轮对话中提取用户持久信息。

职责边界：
  - 只从「用户输入」提取（产品决策）；ai_reply 仅作轻量上下文帮 LLM 理解指代，不从中提取用户信息。
  - 复用项目已有 LLM 单例（chains._get_rag_llm），不新建客户端。
  - 结构化 5 字段在返回前完成清洗校验；语义事实 0~3 条。
  - 全函数 try/except 兜底：LLM 未初始化 / 超时 / 解析失败等一律返回空结果，绝不抛给调用方。

返回结构：
  成功: {"structured": {"name","age","id_card","phone","plate"}, "semantic": [str, ...]}
  失败: {"structured": {}, "semantic": []}
"""
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from src.logger import get_logger

logger = get_logger(__name__)

# 提取失败 / 无可提取信息时返回的空结果
_EMPTY_RESULT: Dict[str, Any] = {"structured": {}, "semantic": []}

# 结构化字段的规范键顺序（清洗后始终返回这 5 个键，缺失为 None）
_STRUCT_KEYS = ("name", "age", "id_card", "phone", "plate")

_SYSTEM_PROMPT = """你是车险客服系统的「用户长期记忆提取器」。任务：从一轮对话中提取值得长期记住的用户持久信息，只输出一个 JSON 对象。

【总原则】
1. 只提取对话中明确出现的确定事实，绝不猜测、不推断、不补全。
2. 提取不到就填 null（结构化）或不输出（语义），严禁编造。

【结构化字段 structured】只提取以下 5 个字段，提取不到填 null：
- name: 姓名
- age: 年龄（整数，例如「37岁」填 37）
- id_card: 身份证号
- phone: 手机号
- plate: 车牌号

【语义事实 semantic】提取 0~3 条用户持久事实，每条是一个自然语言短句，主语统一用「用户」。可提取的类型：
- 历史事件（如出险、理赔、事故记录）
- 业务状态（如已投保某险种、保单即将到期）
- 用户偏好（如关注保费价格、偏好线上办理）
- 车辆特征（如车型、车龄、使用性质）

【不要提取】
- 问候、寒暄、礼貌用语（如「你好」「谢谢」）
- 一次性请求或临时问题（如「帮我查一下XX」「这个怎么算」）
- 临时上下文、指代不清、无法长期复用的内容
- 任何猜测性、不确定的内容
- 上面 5 个结构化字段已包含的信息（不要重复放进 semantic）

【时间处理】
- 相对时间必须转为绝对时间。输入会给出「当前年份」，换算规则：去年=当前年份-1，前年=当前年份-2，明年=当前年份+1。例如当前年份 2026、用户说「去年出过追尾险」，应写成「用户2025年出过追尾险」。

【输出格式】只输出一个 JSON 对象，不要任何解释、前后缀或 markdown 代码围栏，形如：
{"structured": {"name": null, "age": null, "id_card": null, "phone": null, "plate": null}, "semantic": []}
"""

_HUMAN_TEMPLATE = """当前年份：{year}

【用户本轮输入】
{user_msg}

【客服本轮回复】（仅供理解指代关系，不要从中提取用户信息）
{ai_reply}

请严格按系统要求，只输出 JSON。"""


# ============================================================
# 结构化字段清洗（存前洗，纯函数，便于单测）
# ============================================================
def _clean_phone(v: Any) -> Optional[str]:
    """手机号：只留数字，长度必须 11 位。'138-1234-5678'->'13812345678'，'123'->None。"""
    if v is None:
        return None
    digits = re.sub(r"\D", "", str(v))
    return digits if len(digits) == 11 else None


def _clean_id_card(v: Any) -> Optional[str]:
    """身份证：18 位（末位可为 X）或 15 位纯数字，否则 None。统一转大写。"""
    if v is None:
        return None
    s = str(v).strip().upper()
    if re.fullmatch(r"\d{17}[\dX]", s):
        return s
    if re.fullmatch(r"\d{15}", s):
        return s
    return None


def _clean_plate(v: Any) -> Optional[str]:
    """车牌：首位汉字 + 5~6 位字母数字 + 可选特殊尾字，总长 7~8，否则 None。统一转大写。"""
    if v is None:
        return None
    s = str(v).strip().upper()
    if 7 <= len(s) <= 8 and re.fullmatch(r"[\u4e00-\u9fa5][A-Z0-9]{5,6}[A-Z0-9挂学警港澳]?", s):
        return s
    return None


def _clean_age(v: Any) -> Optional[int]:
    """年龄：先用正则提取数字（兼容 '37岁'/'37 岁'/37），转 int，0<age<=120 才保留。"""
    if v is None:
        return None
    m = re.search(r"\d+", str(v))
    if not m:
        return None
    try:
        age = int(m.group())
    except (ValueError, TypeError):
        return None
    return age if 0 < age <= 120 else None


def _clean_name(v: Any) -> Optional[str]:
    """姓名：strip 后非空且长度 <=20 才保留。"""
    if v is None:
        return None
    s = str(v).strip()
    return s if s and len(s) <= 20 else None


def _clean_structured(raw: Any) -> Dict[str, Any]:
    """把 LLM 返回的 structured 逐字段清洗，始终返回含 5 个键的 dict（缺失为 None）。"""
    if not isinstance(raw, dict):
        return {k: None for k in _STRUCT_KEYS}
    return {
        "name": _clean_name(raw.get("name")),
        "age": _clean_age(raw.get("age")),
        "id_card": _clean_id_card(raw.get("id_card")),
        "phone": _clean_phone(raw.get("phone")),
        "plate": _clean_plate(raw.get("plate")),
    }


def _clean_semantic(raw: Any) -> List[str]:
    """语义列表清洗：只留非空字符串，最多 3 条。"""
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
        if len(out) >= 3:
            break
    return out


def _parse_json(text: str) -> Optional[dict]:
    """从 LLM 文本输出中解析出 JSON 对象；兼容 ```json 围栏与前后杂文本。失败返回 None。"""
    if not text:
        return None
    s = str(text).strip()
    # 剥离 markdown 代码围栏 ```json ... ``` 或 ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    # 退一步：截取第一个 '{' 到最后一个 '}' 的子串再试（容忍前后解释性文字）
    left, right = s.find("{"), s.rfind("}")
    if left != -1 and right != -1 and right > left:
        try:
            obj = json.loads(s[left:right + 1])
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


# ============================================================
# 对外主函数
# ============================================================
def extract_memory_from_conversation(user_id: str, user_msg: str, ai_reply: str = "") -> Dict[str, Any]:
    """从一轮对话提取长期记忆。任何异常都返回空结果，不抛出。

    参数：
      user_id : 用户标识（仅用于日志）
      user_msg: 本轮用户输入（提取主来源）
      ai_reply: 本轮客服回复（仅帮 LLM 理解指代，不从中提取用户信息）
    返回：
      {"structured": {5 字段，缺失为 None}, "semantic": [0~3 条持久事实]}
      失败/无信息时为 {"structured": {}, "semantic": []}
    """
    try:
        if not user_msg or not str(user_msg).strip():
            return dict(_EMPTY_RESULT)

        # 懒 import：chains 依赖较重且此刻才需要，避免任何 import 期耦合
        from src.chains.chains import _get_rag_llm
        llm = _get_rag_llm()  # 未初始化会抛 RuntimeError，被外层 except 捕获

        year = datetime.now().year
        human = _HUMAN_TEMPLATE.format(year=year, user_msg=str(user_msg).strip(), ai_reply=(ai_reply or "(无)").strip())

        resp = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human)])
        content = resp.content if hasattr(resp, "content") else str(resp)

        data = _parse_json(content)
        if data is None:
            logger.warning("[记忆-提取] LLM 输出解析失败，返回空结果；raw=%s", str(content)[:120])
            return dict(_EMPTY_RESULT)

        structured = _clean_structured(data.get("structured"))
        semantic = _clean_semantic(data.get("semantic"))

        hit = {k: v for k, v in structured.items() if v is not None}
        logger.info("[记忆-提取] user_id=%s 结构化命中=%s 语义=%d条", user_id, hit, len(semantic))
        return {"structured": structured, "semantic": semantic}
    except Exception as e:
        # LLM 未初始化 / 超时 / 网络 / 任意异常：静默返回空结果，绝不影响主流程
        logger.error("[记忆-提取] 异常，返回空结果 user_id=%s: %s", user_id, e)
        return dict(_EMPTY_RESULT)
