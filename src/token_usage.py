"""
Token 使用量统计和每日限额管理
数据存储在 data/usage_cache.json 中
"""

import json
import os
from datetime import date
from typing import Dict, Any

USAGE_FILE = "data/usage_cache.json"

# 每日 Token 上限（输入 + 输出）
# 基于真实账单数据：日均约 2 万 token，1,000,000 token 约为 50 倍冗余
# 测试时可临时改为 100 验证拦截逻辑
DAILY_TOKEN_LIMIT = int(os.environ.get("DAILY_TOKEN_LIMIT", 1_000_000))



def _load_usage() -> Dict[str, Any]:
    """加载今日使用量"""
    if not os.path.exists(USAGE_FILE):
        return {"date": str(date.today()), "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    with open(USAGE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_usage(usage: Dict[str, Any]):
    """保存使用量"""
    os.makedirs(os.path.dirname(USAGE_FILE), exist_ok=True)
    with open(USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(usage, f, ensure_ascii=False, indent=2)


def get_today_usage() -> Dict[str, Any]:
    """
    获取今日使用量
    如果日期不是今天，自动重置
    """
    usage = _load_usage()
    today = str(date.today())
    if usage.get("date") != today:
        # 新的一天，重置
        usage = {"date": today, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        _save_usage(usage)
    return usage


def add_tokens(input_tokens: int, output_tokens: int) -> Dict[str, Any]:
    """
    累加今日 Token 使用量
    参数：
        input_tokens: 输入 token 数（含缓存命中）
        output_tokens: 输出 token 数
    返回：
        更新后的今日使用量
    """
    usage = get_today_usage()
    usage["input_tokens"] += input_tokens
    usage["output_tokens"] += output_tokens
    usage["total_tokens"] += input_tokens + output_tokens
    _save_usage(usage)
    return usage

def is_budget_exceeded() -> bool:
    """检查今日累计 token 是否超过上限"""
    usage = get_today_usage()
    return usage.get("total_tokens", 0) >= DAILY_TOKEN_LIMIT


def reset_today_usage():
    """手动重置今日使用量（用于测试）"""
    today = str(date.today())
    usage = {"date": today, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    _save_usage(usage)