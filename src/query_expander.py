"""
查询扩展模块
将用户口语化表达转换为保险条款标准术语，提升 BM25 关键词召回率
"""
import os
import json
import logging

logger = logging.getLogger(__name__)

_SYNONYM_DICT = None
_DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synonym_dict.json")


def load_synonym_dict() -> dict:
    """加载 synonym_dict.json（带缓存）"""
    global _SYNONYM_DICT
    if _SYNONYM_DICT is not None:
        return _SYNONYM_DICT

    if os.path.exists(_DICT_PATH):
        with open(_DICT_PATH, "r", encoding="utf-8") as f:
            _SYNONYM_DICT = json.load(f)
        logger.info("加载同义词词典: %d 条 (%s)", len(_SYNONYM_DICT), _DICT_PATH)
    else:
        logger.warning("同义词词典不存在: %s，使用空词典", _DICT_PATH)
        _SYNONYM_DICT = {}
    return _SYNONYM_DICT


# 险种类术语（硬编码，与业务术语分开计数）
INSURANCE_TYPE_TERMS = {
    "机动车损失保险",
    "机动车第三者责任保险",
    "机动车车上人员责任保险",
    "第三者责任保险",
    "车上人员责任保险",
    "车上人员责任险",
    "车辆损失险",
    "机动车交通事故责任强制保险",
    "交通事故责任强制保险",
    "交强险",
    "商业险",
    "附加险",
    "主险",
}


def expand_query(query: str, insurance_type: str = None) -> str:
    """
    将用户口语化查询扩展为包含标准术语的查询（仅用于 BM25）

    策略：
    - 险种类术语：最多取 1 个（匹配最长的，优先带"机动车"前缀的全称）
    - 业务类术语：最多取 2 个（按匹配长度降序，长的更精确）
    - 增量去重：标准术语已出现在 query 中则跳过

    参数:
        query: 原始用户查询（已经过简称替换）
        insurance_type: 险种识别结果（仅用于日志）

    返回:
        扩展后的查询字符串（原始query + 空格 + 标准术语）

    示例:
        >>> expand_query("机动车第三者责任保险赔多少", "三者险")
        "机动车第三者责任保险赔多少 责任限额 赔偿处理"
    """
    synonym_dict = load_synonym_dict()

    # ---- a. 遍历词典，收集所有命中的术语 ----
    matched_items = []  # [(match_len, standard_term, is_insurance_type)]
    MIN_MATCH_LEN = 3  # 最小匹配长度，防止2字短词子串误匹配

    for standard_term, variants in synonym_dict.items():
        best_variant_len = 0
        # 检查标准术语本身是否在 query 中（术语长度 < 3 跳过）
        if len(standard_term) >= MIN_MATCH_LEN and standard_term in query:
            best_variant_len = max(best_variant_len, len(standard_term))
        # 检查口语变体是否在 query 中（变体长度 < 3 跳过，取最长匹配）
        for variant in variants:
            if len(variant) >= MIN_MATCH_LEN and variant in query:
                best_variant_len = max(best_variant_len, len(variant))
        if best_variant_len > 0:
            is_type = standard_term in INSURANCE_TYPE_TERMS
            matched_items.append((best_variant_len, standard_term, is_type))

    # ---- b. 分类：险种类 vs 业务类 ----
    type_items = [(l, t) for l, t, is_t in matched_items if is_t]
    biz_items = [(l, t) for l, t, is_t in matched_items if not is_t]

    # 险种类：按匹配长度降序，取第 1 个（最精确的全称）
    type_items.sort(key=lambda x: x[0], reverse=True)
    expansion_terms = []
    if type_items:
        expansion_terms.append(type_items[0][1])

    # ---- c. 业务类：增量去重 + 按匹配长度降序，最多取 2 个 ----
    biz_items.sort(key=lambda x: x[0], reverse=True)
    for match_len, term in biz_items:
        # 增量去重：该术语已出现在 query 或已有扩展词中，跳过
        current = query + " " + " ".join(expansion_terms)
        if term in current:
            continue
        expansion_terms.append(term)
        if len(expansion_terms) >= 3:  # 1 险种 + 2 业务 = 最多 3 个
            break

    # ---- d. 构建扩展 query ----
    if expansion_terms:
        expanded = query + " " + " ".join(expansion_terms)
    else:
        expanded = query

    # ---- e. 日志 ----
    matched_variants = []
    for standard_term, variants in synonym_dict.items():
        for variant in variants:
            if variant in query:
                matched_variants.append(f'"{variant}"->"{standard_term}"')
                break

    logger.info(
        "查询扩展: 原始=\"%s\" | 险种=%s | 命中变体=%s | 扩展词=%s | 结果=\"%s\"",
        query,
        insurance_type or "None",
        matched_variants or "无",
        expansion_terms or "无",
        expanded,
    )

    return expanded
