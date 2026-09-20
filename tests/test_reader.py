# -*- coding: utf-8 -*-
"""L1 + L2 测试：src/memory/reader.py。

- build_memory_prompt：纯函数（L1，离线）。
- resolve_masked_id_card：真实 sqlite 往返（L2，用 temp_user fixture）。
- load_user_memory：结构化走真实 sqlite，语义走 mock（隔离 PG / fastembed 依赖，L2）。
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.reader import (  # noqa: E402
    build_memory_prompt, resolve_masked_id_card, load_user_memory,
)

_FULL = {"name": "张三", "age": 37, "id_card": "110101199001011234",
         "phone": "13812345678", "plate": "京A12345"}
_FAKE_EMB = [[0.1] * 512]  # 假 embedding，避开真实 fastembed 模型加载


# ========== L1: build_memory_prompt（纯函数）==========
def test_build_prompt_both_empty():
    assert build_memory_prompt(None, []) == ""
    assert build_memory_prompt({}, []) == ""


def test_build_prompt_full_fields_masked():
    p = build_memory_prompt(_FULL, ["用户2025年出过追尾险"])
    assert "【用户留存信息】" in p
    assert "【用户相关历史】" in p
    assert "姓名：张三" in p                     # 姓名明文
    assert "年龄：37" in p                       # 年龄明文
    assert "身份证：110***********1234" in p      # 掩码
    assert "手机号：138****5678" in p             # 掩码
    assert "车牌：京A***5" in p                   # 掩码
    assert "- 用户2025年出过追尾险" in p


def test_build_prompt_no_plaintext():
    """安全断言：prompt 里绝不出现明文身份证 / 手机。"""
    p = build_memory_prompt(_FULL, [])
    assert "110101199001011234" not in p
    assert "13812345678" not in p


def test_build_prompt_confirm_wording():
    """方向A：强制“用后告知确认”措辞存在，且不再是软性“如需”。"""
    p = build_memory_prompt(_FULL, [])
    assert "必须" in p
    assert "请用户确认" in p
    assert "优先级高于" in p
    assert "如需" not in p


def test_build_prompt_semantic_only():
    p = build_memory_prompt(None, ["用户关注保费价格"])
    assert "【用户相关历史】" in p
    assert "【用户留存信息】" not in p


def test_build_prompt_structured_only_partial():
    """只有部分字段时只出对应行（无 id_card 就不出身份证行）。"""
    p = build_memory_prompt({"name": "李四"}, [])
    assert "【用户留存信息】" in p
    assert "姓名：李四" in p
    # 用带冒号的“字段行”判定：方向A措辞的示例里含“身份证 110***”（空格、无冒号），
    # 故 "身份证："（中文冒号）不出现即代表未输出身份证字段行。
    assert "身份证：" not in p


# ========== L2: resolve_masked_id_card（真实 sqlite）==========
def test_resolve_plaintext_passthrough():
    """明文（不含 *）原样返回，不查库。"""
    assert resolve_masked_id_card("_any_user", "110101199001011234") == "110101199001011234"


def test_resolve_empty_userid_masked_passthrough():
    assert resolve_masked_id_card("", "110***********1234") == "110***********1234"


def test_resolve_masked_to_plaintext(temp_user):
    """掩码 + 库中有该用户 → 解码为明文。"""
    assert resolve_masked_id_card(temp_user, "110***********1234") == "110101199001011234"


def test_resolve_masked_no_user_passthrough():
    """掩码 + 库中无该用户 → 原样返回掩码（优雅降级，不抛异常）。"""
    assert resolve_masked_id_card("_no_such_user_xyz", "110***********1234") == "110***********1234"


# ========== L2: load_user_memory（结构化真实 + 语义 mock）==========
def test_load_not_logged_in():
    assert load_user_memory("", "帮我查保单") == {"prompt_text": "", "semantic": []}


def test_load_structured_masked(temp_user):
    with patch("src.rag._embed_texts", return_value=_FAKE_EMB), \
         patch("src.memory.reader.semantic_store") as mock_ss:
        mock_ss.search.return_value = []
        m = load_user_memory(temp_user, "帮我查保单")
    assert set(m.keys()) == {"prompt_text", "semantic"}
    assert "张三" in m["prompt_text"]
    assert "110***********1234" in m["prompt_text"]
    assert "110101199001011234" not in m["prompt_text"]   # 不含明文
    assert m["semantic"] == []


def test_load_semantic_threshold_filter(temp_user):
    """语义按阈值 0.45 过滤：>=0.45 保留，<0.45 丢弃。"""
    with patch("src.rag._embed_texts", return_value=_FAKE_EMB), \
         patch("src.memory.reader.semantic_store") as mock_ss:
        mock_ss.search.return_value = [("高相似事实", 0.80), ("低相似事实", 0.30)]
        m = load_user_memory(temp_user, "相关问题")
    assert m["semantic"] == ["高相似事实"]
    assert "高相似事实" in m["prompt_text"]
    assert "低相似事实" not in m["prompt_text"]


def test_load_pg_down_graceful(temp_user):
    """PG / 语义检索异常 → 语义降级为空，结构化不受影响，不抛异常。"""
    with patch("src.rag._embed_texts", return_value=_FAKE_EMB), \
         patch("src.memory.reader.semantic_store") as mock_ss:
        mock_ss.search.side_effect = Exception("connection refused")
        m = load_user_memory(temp_user, "帮我查保单")
    assert m["semantic"] == []
    assert "张三" in m["prompt_text"]          # 结构化仍正常


def test_load_empty_msg_skip_semantic(temp_user):
    """user_msg 为空 → 跳过语义检索（不调 embedding），只出结构化。"""
    with patch("src.rag._embed_texts") as mock_emb, \
         patch("src.memory.reader.semantic_store"):
        m = load_user_memory(temp_user, "")
    assert m["semantic"] == []
    assert "张三" in m["prompt_text"]
    mock_emb.assert_not_called()   # 空消息不应触发 embedding
