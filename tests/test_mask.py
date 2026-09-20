# -*- coding: utf-8 -*-
"""L1 单元测试：src/memory/mask.py 掩码纯函数（离线、无外部依赖）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.mask import (  # noqa: E402
    is_masked, mask_name, mask_age, mask_phone, mask_id_card, mask_plate, mask_sensitive_text,
)


# ---- is_masked ----
def test_is_masked():
    assert is_masked("110***********1234") is True
    assert is_masked("110101199001011234") is False
    assert is_masked("") is False
    assert is_masked(None) is False


# ---- 姓名 / 年龄：明文不遮 ----
def test_mask_name():
    assert mask_name("  张三 ") == "张三"
    assert mask_name("") == ""
    assert mask_name(None) == ""


def test_mask_age():
    assert mask_age(37) == "37"
    assert mask_age(None) == ""
    assert mask_age("") == ""


# ---- 手机号 ----
def test_mask_phone():
    assert mask_phone("13812345678") == "138****5678"
    assert mask_phone("1381234") == ""        # 非 11 位
    assert mask_phone("1381234567a") == ""    # 含字母
    assert mask_phone("") == ""
    assert mask_phone(None) == ""


# ---- 身份证 ----
def test_mask_id_card():
    assert mask_id_card("110101199001011234") == "110***********1234"   # 18 位
    assert mask_id_card("11010119900101123X") == "110***********123X"   # 末位 X
    assert mask_id_card("11010119900101123x") == "110***********123X"   # 末位 x→大写
    assert mask_id_card("123456789012345") == "123********2345"         # 15 位
    assert mask_id_card("12345") == ""                                   # 非法长度
    assert mask_id_card("") == ""
    assert mask_id_card(None) == ""


# ---- 车牌 ----
def test_mask_plate():
    assert mask_plate("京A12345") == "京A***5"      # 普通 7 位
    assert mask_plate("京AD12345") == "京A***5"     # 新能源 8 位
    assert mask_plate("京") == ""                    # 过短
    assert mask_plate("") == ""
    assert mask_plate(None) == ""


# ---- mask_sensitive_text：工具结果兜底 ----
def test_mask_sensitive_text_basic():
    assert mask_sensitive_text("") == ""
    assert mask_sensitive_text(None) == ""
    assert mask_sensitive_text("身份证110101199001011234哦") == "身份证110***********1234哦"
    assert mask_sensitive_text("旧证123456789012345") == "旧证123********2345"
    assert mask_sensitive_text("手机13812345678呀") == "手机138****5678呀"
    assert mask_sensitive_text("车牌京A12345的") == "车牌京A***5的"


def test_mask_sensitive_text_long_before_short():
    """从长到短替换：18 位身份证不被 15 位 / 手机号规则误伤。"""
    assert mask_sensitive_text("110101199001011234") == "110***********1234"
    assert mask_sensitive_text("证110101199001011234机13812345678牌京A12345") == \
        "证110***********1234机138****5678牌京A***5"


def test_mask_sensitive_text_success_json_untouched():
    """query_policy 成功 JSON（含保单号 / 金额 / 日期等数字）不应被误伤。"""
    ok = '{"status":"success","data":{"保单号":"POL20260001","车主":"张三","保额":"500,000 元","到期日":"2027-06-30"}}'
    assert mask_sensitive_text(ok) == ok


def test_mask_sensitive_text_error_json_masked():
    """query_policy 查不到时 error 回显明文身份证，必须被遮回掩码。"""
    err_in = '{"status":"error","message":"未找到保单（保单号：POL1，身份证号：110101199001011234）"}'
    err_out = '{"status":"error","message":"未找到保单（保单号：POL1，身份证号：110***********1234）"}'
    assert mask_sensitive_text(err_in) == err_out
