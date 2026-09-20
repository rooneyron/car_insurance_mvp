# -*- coding: utf-8 -*-
"""敏感信息掩码（纯函数，无副作用，无 store 依赖）。

两类用途：
  1. 展示掩码：把结构化记忆里的证件字段遮成掩码后注入 LLM prompt
     （姓名/年龄明文；身份证/手机号/车牌掩码）。
  2. 工具结果兜底：mask_sensitive_text 用正则识别任意文本里的真实证件号并遮罩。
     本批仅用于 query_policy 的工具结果回灌 LLM 之前——工具执行前 id_card 被解码成明文，
     若"查不到"错误信息回显了明文身份证，这里把它遮回掩码，保证 LLM 永远看不到明文。

替换顺序：先长（身份证18位）后短（身份证15位/手机号），避免长串的子串被短规则二次误伤；
替换结果含 * 后不再被后续规则命中。
"""
import re

# 省份简称（车牌首位）——与存储侧 _clean_plate 的中文首字保持一致
_PLATE_PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"

# 工具结果兜底正则（未锚定，用于在整段文本里搜索；顺序即替换优先级：长 → 短）
_RE_ID18 = re.compile(r"\d{17}[\dXx]")
_RE_ID15 = re.compile(r"\d{15}")
_RE_PHONE = re.compile(r"1[3-9]\d{9}")
_RE_PLATE = re.compile(r"[" + _PLATE_PROVINCES + r"][A-Z][A-Z0-9]{5,6}")


def is_masked(value) -> bool:
    """判断入参值是否已是掩码：含 '*' 即视为掩码。空值返回 False。"""
    if value is None:
        return False
    return "*" in str(value)


def mask_name(name) -> str:
    """姓名：明文，不掩码。strip 后返回；空/None → ""。"""
    if name is None:
        return ""
    return str(name).strip()


def mask_age(age) -> str:
    """年龄：明文，不掩码。str(age)；None/空 → ""。"""
    if age is None or str(age).strip() == "":
        return ""
    return str(age)


def mask_phone(phone) -> str:
    """手机号：前3后4，中间固定4个*。13812345678 → 138****5678。非11位/空 → ""。"""
    if phone is None:
        return ""
    digits = str(phone).strip()
    if len(digits) != 11 or not digits.isdigit():
        return ""
    return digits[:3] + "****" + digits[7:]


def mask_id_card(id_card) -> str:
    """身份证：前3后4，中间*。18位→11个*；15位→8个*。其它格式/空 → ""。"""
    if id_card is None:
        return ""
    s = str(id_card).strip().upper()
    if re.fullmatch(r"\d{17}[\dX]", s):
        return s[:3] + "*" * 11 + s[14:]
    if re.fullmatch(r"\d{15}", s):
        return s[:3] + "*" * 8 + s[11:]
    return ""


def mask_plate(plate) -> str:
    """车牌：前2后1，中间固定3个*。京A12345 → 京A***5；新能源8位同理。空/过短 → ""。"""
    if plate is None:
        return ""
    s = str(plate).strip().upper()
    if len(s) < 3:
        return ""
    return s[:2] + "***" + s[-1]


def mask_sensitive_text(text) -> str:
    """工具结果兜底：识别文本里的真实身份证/手机号/车牌并遮罩，返回处理后文本。

    按"从长到短"顺序替换（身份证18 → 身份证15 → 手机号 → 车牌）：
    先替换长的，可避免其子串被短规则误伤；替换后含 * 也不再被后续规则命中。
    空文本/None → ""。
    """
    if not text:
        return ""
    s = str(text)
    s = _RE_ID18.sub(lambda m: m.group()[:3] + "*" * 11 + m.group()[14:], s)
    s = _RE_ID15.sub(lambda m: m.group()[:3] + "*" * 8 + m.group()[11:], s)
    s = _RE_PHONE.sub(lambda m: m.group()[:3] + "****" + m.group()[7:], s)
    s = _RE_PLATE.sub(lambda m: m.group()[:2] + "***" + m.group()[-1], s)
    return s
