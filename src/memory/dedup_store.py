# -*- coding: utf-8 -*-
"""飞书消息去重持久化存储（sqlite）：防止进程重启后飞书重投已处理消息导致重复回复。

设计要点（与 structured_store.py 对齐）：
  - 只用标准库 sqlite3，不引新依赖；连接用完即关（MVP 不做连接池）。
  - db 复用 data/user_memory.db（与长期记忆同一 sqlite 文件），路径基于项目根绝对定位，不受启动 CWD 影响。
  - 判重与登记分离：is_duplicate_message 只读 SELECT 判重、不登记；mark_message_processed 在
    「飞书 reply 成功后」才 INSERT OR IGNORE 登记。避免回复失败的消息被误标记为已处理、
    导致飞书重投时被跳过而用户永远收不到回复。
  - 建表幂等、失败降级：任何异常都放行（返回 False），保证去重故障绝不阻断飞书消息处理
    （处理逻辑本身已幂等：RAG/工具只读、长期记忆有相似度去重，偶漏判不产生脏数据）。
  - 写操作加进程内锁 + connect(timeout)，规避飞书 WS 后台线程并发写 "database is locked"。

对外暴露：
  - is_duplicate_message(message_id) -> bool：只读判重，已成功回复过返回 True（跳过），否则 False（不登记）
  - mark_message_processed(message_id) -> None：登记「已成功回复」，仅在飞书 reply 成功后调用
  - cleanup_old_messages(days=7) -> None：清理 N 天前记录，防表无限增长（app.py 启动时调用一次）
"""
import os
import sqlite3
import threading

from src.logger import get_logger

logger = get_logger(__name__)

# data/user_memory.db 绝对路径：从本文件回溯三级到项目根（src/memory/dedup_store.py -> 根），与 structured_store 一致
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "user_memory.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS processed_messages (
    message_id   TEXT PRIMARY KEY,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_write_lock = threading.Lock()   # 串行化写，规避多线程 "database is locked"
_table_ready = False             # 进程内建表只做一次的标志


def _connect() -> sqlite3.Connection:
    # timeout：并发写等锁上限；check_same_thread=False：允许飞书 WS 后台线程访问同一 store
    return sqlite3.connect(_DB_PATH, timeout=30.0, check_same_thread=False)


def _ensure_table() -> None:
    """幂等建表（进程内成功一次后置 _table_ready）。

    CREATE TABLE IF NOT EXISTS 天然幂等，故此处不加写锁——避免与调用方的
    `with _write_lock` 嵌套导致死锁（threading.Lock 不可重入）；并发首次最多重复建表，无害。
    """
    global _table_ready
    if _table_ready:
        return
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = _connect()
    try:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
        _table_ready = True
        logger.info("[去重] processed_messages 表就绪: %s", _DB_PATH)
    finally:
        conn.close()


def is_duplicate_message(message_id: str) -> bool:
    """只读判重：message_id 是否已「成功回复」过。

    已处理 -> True（重复消息，调用方应直接跳过：不回复、不调 chat、不存记忆）；
    未处理 -> False（放行，但此处不登记）。

    只读 SELECT、不写入——登记由 mark_message_processed 在「飞书 reply 成功后」完成，
    避免回复失败的消息被误标记为已处理、导致飞书重投时被跳过而用户永远收不到回复。
    任何异常降级为 False（放行），保证去重故障不阻断正常消息（处理逻辑本身幂等）。
    """
    if not message_id:
        return False
    try:
        _ensure_table()
        conn = _connect()   # 只读查询不加写锁（sqlite 支持并发读，与写操作互斥由 sqlite 自身保证）
        try:
            cur = conn.execute(
                "SELECT 1 FROM processed_messages WHERE message_id = ? LIMIT 1",
                (message_id,),
            )
            return cur.fetchone() is not None   # 有记录=已成功回复过=重复
        finally:
            conn.close()
    except Exception as e:
        logger.error("[去重] is_duplicate_message 查询失败 message_id=%s（降级放行）: %s", message_id, e)
        return False


def mark_message_processed(message_id: str) -> None:
    """登记 message_id 为「已成功回复」。仅在飞书 reply 成功后调用（回复失败不登记，留给重投重试）。

    INSERT OR IGNORE 幂等：重复登记同一 message_id 无害。失败只记日志——大不了飞书重投/重连
    补推时重新处理（chat/reply 本身幂等），不会因登记失败而丢消息或重复回复已成功过的消息。
    """
    if not message_id:
        return
    try:
        _ensure_table()
        with _write_lock:
            conn = _connect()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO processed_messages (message_id) VALUES (?)",
                    (message_id,),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        logger.error("[去重] mark_message_processed 失败 message_id=%s（不影响回复）: %s", message_id, e)


def cleanup_old_messages(days: int = 7) -> None:
    """清理 N 天前的已处理消息记录，防止 processed_messages 无限增长。app.py 启动时调用一次。

    用 sqlite 的 datetime('now','-N days')（UTC）与 processed_at（CURRENT_TIMESTAMP 同为 UTC）比较，
    规避 Python 本地时间与 sqlite UTC 的时区偏差。失败只记日志，不影响启动。
    """
    try:
        _ensure_table()
        with _write_lock:
            conn = _connect()
            try:
                cur = conn.execute(
                    f"DELETE FROM processed_messages WHERE processed_at < datetime('now', '-{int(days)} days')"
                )
                conn.commit()
                logger.info("[去重] 清理 %d 天前记录 %d 条", days, cur.rowcount)
            finally:
                conn.close()
    except Exception as e:
        logger.error("[去重] cleanup_old_messages 失败（不影响启动）: %s", e)
