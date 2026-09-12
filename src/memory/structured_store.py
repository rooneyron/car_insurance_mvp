# -*- coding: utf-8 -*-
"""结构化记忆存储（sqlite）：用户 5 字段画像（姓名/年龄/身份证/手机/车牌）。

设计要点：
  - 只用标准库 sqlite3，不引新依赖；连接用完即关（MVP 不做连接池）。
  - upsert 只覆盖 memory_dict 中非 None 的字段，其余保留旧值（增量更新）。
  - user_id 为空快速返回，避免脏数据。
  - sqlite 文件基于项目根绝对定位，不受启动 CWD 影响。
  - 写操作加进程内锁 + connect(timeout)，规避后台线程并发写 "database is locked"。
"""
import os
import sqlite3
import threading
from typing import Optional, Dict

from src.logger import get_logger

logger = get_logger(__name__)

# data/user_memory.db 绝对路径：从本文件回溯三级到项目根（src/memory/structured_store.py -> 根）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "user_memory.db")

# 结构化记忆的业务字段白名单（user_id 为主键，不在此列）；upsert 只认这些列，防注入
_FIELDS = ("name", "age", "id_card", "phone", "plate")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_memory (
    user_id    TEXT PRIMARY KEY,
    name       TEXT,
    age        INTEGER,
    id_card    TEXT,
    phone      TEXT,
    plate      TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


class SqliteStructuredStore:
    """sqlite 结构化记忆存储（用户画像 5 字段）。

    每次操作独立开/关连接；写操作串行化，可安全被后台线程调用。
    """

    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        self._write_lock = threading.Lock()  # 串行化写，规避多线程锁冲突
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        # timeout：并发写等锁上限；check_same_thread=False：允许后台线程访问同一 store
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self) -> None:
        """初始化即建表（幂等），并确保 data/ 目录存在。"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = self._connect()
            try:
                conn.execute(_CREATE_TABLE_SQL)
                conn.commit()
                logger.info("[记忆-结构化] sqlite 表就绪: %s", self.db_path)
            finally:
                conn.close()
        except Exception as e:
            # 兜底所有异常（含 os.makedirs 的 OSError/PermissionError）：建表失败绝不能
            # 阻断模块级实例化→import src.memory→app 启动；表没建成时后续 get/upsert 各自降级。
            logger.error("[记忆-结构化] 建表失败（记忆功能降级，不影响启动）: %s", e)

    def get(self, user_id: str) -> Optional[Dict]:
        """按 user_id 查询，返回字段 dict 或 None。"""
        if not user_id:
            return None
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT user_id, name, age, id_card, phone, plate, updated_at, created_at "
                "FROM user_memory WHERE user_id = ?",
                (user_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error("[记忆-结构化] get 失败 user_id=%s: %s", user_id, e)
            return None
        finally:
            conn.close()

    def upsert(self, user_id: str, memory_dict: Dict) -> None:
        """插入或增量更新：只覆盖 memory_dict 中非 None 的合法字段，其余保留旧值。"""
        if not user_id or not memory_dict:
            return
        # 只取白名单字段且值非 None（None 表示"本轮没提取到"，不覆盖旧值）
        updates = {k: v for k, v in memory_dict.items() if k in _FIELDS and v is not None}
        if not updates:
            return
        cols = list(updates.keys())
        vals = [updates[c] for c in cols]
        placeholders = ", ".join("?" * len(cols))
        set_clause = ", ".join(f"{c} = excluded.{c}" for c in cols)
        sql = (
            f"INSERT INTO user_memory (user_id, {', '.join(cols)}, updated_at, created_at) "
            f"VALUES (?, {placeholders}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
            f"ON CONFLICT(user_id) DO UPDATE SET {set_clause}, updated_at = CURRENT_TIMESTAMP"
        )
        with self._write_lock:
            conn = self._connect()
            try:
                conn.execute(sql, [user_id] + vals)
                conn.commit()
                logger.info("[记忆-结构化] upsert user_id=%s 字段=%s", user_id, cols)
            except sqlite3.Error as e:
                logger.error("[记忆-结构化] upsert 失败 user_id=%s: %s", user_id, e)
            finally:
                conn.close()

    def delete(self, user_id: str) -> None:
        """按 user_id 删除整条画像。"""
        if not user_id:
            return
        with self._write_lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM user_memory WHERE user_id = ?", (user_id,))
                conn.commit()
                logger.info("[记忆-结构化] delete user_id=%s", user_id)
            except sqlite3.Error as e:
                logger.error("[记忆-结构化] delete 失败 user_id=%s: %s", user_id, e)
            finally:
                conn.close()
