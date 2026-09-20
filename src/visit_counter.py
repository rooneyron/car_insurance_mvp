# -*- coding: utf-8 -*-
"""
访问计数器模块
- 内存计数 + 异步持久化到 SQLite
- 服务启动时从 DB 加载当天计数
"""

import sqlite3
from datetime import date
from pathlib import Path
from src.logger import get_logger

logger = get_logger(__name__)

# SQLite 数据库路径
DB_PATH = Path("data/visits.db")

# 内存计数：{date_str: count}
_visit_counts: dict[str, int] = {}


def _get_db_connection() -> sqlite3.Connection:
    """获取 SQLite 连接"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_visit_counter():
    """
    初始化访问计数器：
    1. 创建表（如果不存在）
    2. 从 DB 加载当天计数到内存
    """
    global _visit_counts
    
    try:
        conn = _get_db_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                date TEXT PRIMARY KEY,
                count INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()
        
        # 加载当天计数
        today = date.today().isoformat()
        cursor = conn.execute("SELECT count FROM visits WHERE date = ?", (today,))
        row = cursor.fetchone()
        if row:
            _visit_counts[today] = row[0]
            logger.info("[访问计数] 从 DB 加载当天计数: %s = %d", today, row[0])
        else:
            _visit_counts[today] = 0
            logger.info("[访问计数] 当天无记录，初始化为 0")
        
        conn.close()
    except Exception as e:
        logger.error("[访问计数] 初始化失败: %s", e)
        _visit_counts[date.today().isoformat()] = 0


def increment_visit():
    """
    增加访问计数（同步更新内存 + 同步写入 DB）
    访问量很小，直接写 DB 即可，无需异步
    """
    global _visit_counts
    
    today = date.today().isoformat()
    
    # 更新内存计数
    if today not in _visit_counts:
        _visit_counts[today] = 0
    _visit_counts[today] += 1
    
    count = _visit_counts[today]
    logger.info("[访问计数] %s 访问 +1，当前 = %d", today, count)
    
    # 直接写 DB（访问量小，同步即可）
    try:
        _write_visit_to_db(today, count)
    except Exception as e:
        logger.error("[访问计数] 写入 DB 失败: %s", e)


def _write_visit_to_db(date_str: str, count: int):
    """同步写入 DB（在线程池中执行）"""
    conn = _get_db_connection()
    conn.execute("""
        INSERT INTO visits (date, count) VALUES (?, ?)
        ON CONFLICT(date) DO UPDATE SET count = ?
    """, (date_str, count, count))
    conn.commit()
    conn.close()


def get_all_visits() -> dict:
    """
    获取所有访问记录
    返回: {"total": int, "records": [{"date": str, "count": int}, ...]}
    """
    try:
        conn = _get_db_connection()
        cursor = conn.execute("SELECT date, count FROM visits ORDER BY date DESC")
        rows = cursor.fetchall()
        conn.close()
        
        records = [{"date": row[0], "count": row[1]} for row in rows]
        total = sum(r["count"] for r in records)
        
        return {
            "total": total,
            "records": records
        }
    except Exception as e:
        logger.error("[访问计数] 读取 DB 失败: %s", e)
        return {"total": 0, "records": []}
