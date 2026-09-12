# -*- coding: utf-8 -*-
"""长期记忆"读取"：对话开始时同步取用户记忆，组装成掩码 prompt 文本注入 LLM。

对外：
  - load_user_memory(user_id, user_msg) -> {"prompt_text": str(掩码), "semantic": list}
  - build_memory_prompt(structured_dict, semantic_list) -> str
  - resolve_masked_id_card(user_id, value) -> str   # 供 _tools_node 在工具执行前解码掩码 id_card

设计：
  - 结构化明文只在本地用于生成掩码文本，用完即弃，不进 graph state
    （避免明文被 MemorySaver checkpointer 跨轮持久化）。
  - 语义检索复用 rag._embed_texts + semantic_store.search，按相似度阈值过滤
    （读取阈值 0.45，独立于存储去重的 0.85：检索是"当前问题 vs 历史事实"，措辞天然差异大）。
  - 全函数 try/except，任何失败返回空结构，绝不影响对话主流程。
"""
from typing import Dict, List, Optional

from src.logger import get_logger
from src.memory import structured_store, semantic_store
from src.memory.mask import (
    is_masked, mask_name, mask_age, mask_phone, mask_id_card, mask_plate,
)

logger = get_logger(__name__)

# 读取侧语义检索：相似度阈值（独立于存储去重的 0.85）/ 召回条数
_SEMANTIC_READ_THRESHOLD = 0.45
_SEMANTIC_TOP_K = 3

_EMPTY_MEMORY: Dict = {"prompt_text": "", "semantic": []}


def load_user_memory(user_id: str, user_msg: str) -> Dict:
    """同步读取用户长期记忆，返回 {"prompt_text"(掩码文本), "semantic"(0~3条事实)}。

    user_id 空（未登录）或任何异常 → 返回空结构，不影响主流程。
    """
    if not user_id:
        return dict(_EMPTY_MEMORY)
    try:
        # 1) 结构化明文（仅本地用于生成掩码文本，不外传、不进 state）
        structured = None
        try:
            structured = structured_store.get(user_id)
        except Exception as e:
            logger.error("[记忆-读取] 结构化读取失败 user_id=%s: %s", user_id, e)

        # 2) 语义检索（独立 try/except，失败不影响结构化）
        semantic: List[str] = []
        try:
            if user_msg and str(user_msg).strip():
                from src.rag import _embed_texts  # 懒 import，复用已加载的 fastembed 模型
                emb = _embed_texts([str(user_msg).strip()])[0]
                results = semantic_store.search(user_id, emb, top_k=_SEMANTIC_TOP_K)
                semantic = [c for c, s in results if s >= _SEMANTIC_READ_THRESHOLD]
        except Exception as e:
            logger.error("[记忆-读取] 语义检索失败 user_id=%s: %s", user_id, e)
            semantic = []

        # 3) 组装注入文本（掩码）
        prompt_text = build_memory_prompt(structured, semantic)
        logger.debug("[记忆-读取] user_id=%s 命中语义%d条 prompt_len=%d",
                     user_id, len(semantic), len(prompt_text))
        return {"prompt_text": prompt_text, "semantic": semantic}
    except Exception as e:
        logger.error("[记忆-读取] 异常，返回空记忆 user_id=%s: %s", user_id, e)
        return dict(_EMPTY_MEMORY)


def build_memory_prompt(structured_dict: Optional[Dict], semantic_list: Optional[List[str]]) -> str:
    """把结构化画像（掩码）+ 语义事实组装成注入 prompt 的文本。两块都空 → ""。

    - 结构化块：姓名/年龄明文，身份证/手机号/车牌掩码；字段有值才显示对应行。
    - 语义块：措辞与结构化块区分（可作背景自然参考）。
    - 只用 5 个业务字段，忽略 get() 返回的 user_id/updated_at/created_at。
    """
    blocks: List[str] = []

    # ---- 结构化块 ----
    if structured_dict:
        lines: List[str] = []
        name = mask_name(structured_dict.get("name"))
        if name:
            lines.append(f"姓名：{name}")
        age = mask_age(structured_dict.get("age"))
        if age:
            lines.append(f"年龄：{age}")
        id_card = mask_id_card(structured_dict.get("id_card"))
        if id_card:
            lines.append(f"身份证：{id_card}")
        phone = mask_phone(structured_dict.get("phone"))
        if phone:
            lines.append(f"手机号：{phone}")
        plate = mask_plate(structured_dict.get("plate"))
        if plate:
            lines.append(f"车牌：{plate}")
        if lines:
            blocks.append(
                "【用户留存信息】（以下证件信息已掩码，仅用于工具调用入参，切勿输出明文。"
                "仅当工具成功返回查询结果（status=success）时，才必须在回复中主动复述本次所使用的掩码证件号并请用户确认，"
                "例如“本次查询使用了您留存的身份证 110***********1234，请确认信息是否无误”；"
                "若工具返回缺参数（missing_params）、报错（error）、查询失败或根本未执行，则绝不能出现任何“已使用您留存的证件号/请确认”类表述——"
                "传入入参不等于查询成功，二者务必区分。此复述确认为强制要求，优先级高于“简洁、不重复询问已提供信息”等风格规则）\n" + "\n".join(lines)
            )

    # ---- 语义块 ----
    if semantic_list:
        facts = [f"- {str(s).strip()}" for s in semantic_list if s and str(s).strip()]
        if facts:
            blocks.append(
                "【用户相关历史】（以下是与用户相关的既往事实。若用户当前问题的答案已明确包含在其中，"
                "应直接据此作答，无需调用任何工具；仅当其中信息不足以回答当前问题时，才按正常流程调用工具查询。"
                "注意：本授权仅适用于这段历史事实；掩码的证件号始终只能作为工具入参，不可直接写进答案）\n" + "\n".join(facts)
            )

    return "\n\n".join(blocks)


def resolve_masked_id_card(user_id: str, value) -> str:
    """把 LLM 传入的掩码 id_card 解码为明文（供 _tools_node 在工具执行前调用）。

    - 不含 '*' → 是明文（用户当场提供），原样返回；
    - 含 '*' → 用 user_id 重查 structured_store 取明文；取不到则原样返回掩码值
      （query_policy 查不到会走 error 分支，error 里是掩码值，安全；LLM 会追问用户，优雅降级）。
    """
    if not is_masked(value):
        return value
    try:
        mem = structured_store.get(user_id) if user_id else None
        plain = mem.get("id_card") if mem else None
        if plain:
            logger.debug("[记忆-解码] id_card 掩码→明文成功 user_id=%s", user_id)
            return plain
    except Exception as e:
        logger.error("[记忆-解码] 失败 user_id=%s: %s", user_id, e)
    logger.debug("[记忆-解码] 无留存明文，保留掩码值 user_id=%s", user_id)
    return value
