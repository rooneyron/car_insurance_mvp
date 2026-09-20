# -*- coding: utf-8 -*-
"""pytest 共享配置。

- 把项目根加入 sys.path，使测试可 import src.*（src 为命名空间包，无 __init__.py）。
- 提供 temp_user fixture：种一个独立测试账号的结构化记忆，测完自动删除，
  绝不触碰演示用的真实用户数据（如 xuwei）。
"""
import os
import sys

import pytest

# 项目根（tests/ 的上一级）加入 sys.path，保证 `import src.*` 可用
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.memory import structured_store  # noqa: E402  （sys.path 设置后再导入）

# 独立测试账号 + 画像（id_card 与 data/policies.json 的 POL20260001 / 张三 匹配）
_TEST_USER = "_pytest_mem_read"
_TEST_PROFILE = {
    "name": "张三",
    "age": 37,
    "id_card": "110101199001011234",
    "phone": "13812345678",
    "plate": "京A12345",
}


@pytest.fixture
def temp_user():
    """种临时测试用户的结构化记忆；测试结束后删除，保证不残留、不污染真实数据。"""
    structured_store.upsert(_TEST_USER, _TEST_PROFILE)
    yield _TEST_USER
    structured_store.delete(_TEST_USER)
