"""
数据持久化模块 - SQLite 数据库管理。

负责检测记录的增删改查、批量写入和备份。
"""

import json
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from typing import Optional
from contextlib import contextmanager


@dataclass
class InspectionRecord:
    """检测记录数据类"""

    timestamp: str = ""
    image_path: str = ""
    result_path: str = ""
    yolo_result: str = "{}"       # JSON 字符串
    vlm_result: str = "{}"        # JSON 字符串
    final_result: str = "{}"      # JSON 字符串
    defect_types: str = ""
    defect_count: int = 0
    confidence: float = 0.0
    reviewer: str = ""
    review_status: str = "pending"  # pending | confirmed | corrected
    review_time: str = ""
    note: str = ""
    id: Optional[int] = None


class DBManager:
    """SQLite 数据库管理器"""

    def __init__(self, db_path: str = "data/inspection.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库表"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inspection_records (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    image_path      TEXT NOT NULL,
                    result_path     TEXT DEFAULT '',
                    yolo_result     TEXT DEFAULT '{}',
                    vlm_result      TEXT DEFAULT '{}',
                    final_result    TEXT DEFAULT '{}',
                    defect_types    TEXT DEFAULT '',
                    defect_count    INTEGER DEFAULT 0,
                    confidence      REAL DEFAULT 0.0,
                    reviewer        TEXT DEFAULT '',
                    review_status   TEXT DEFAULT 'pending',
                    review_time     DATETIME,
                    note            TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON inspection_records(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_defect_types
                ON inspection_records(defect_types)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_review_status
                ON inspection_records(review_status)
            """)
            conn.commit()

    @contextmanager
    def _get_conn(self):
        """获取线程安全的数据库连接"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise

    # ==================== 写入操作 ====================

    def insert(self, record: InspectionRecord) -> int:
        """插入一条检测记录，返回自增 ID"""
        d = asdict(record)
        d.pop("id", None)
        d.setdefault("timestamp", datetime.now().isoformat())

        columns = ", ".join(d.keys())
        placeholders = ", ".join("?" * len(d))
        values = list(d.values())

        with self._get_conn() as conn:
            cursor = conn.execute(
                f"INSERT INTO inspection_records ({columns}) VALUES ({placeholders})",
                values,
            )
            conn.commit()
            return cursor.lastrowid

    def insert_batch(self, records: list[InspectionRecord]) -> int:
        """批量插入检测记录"""
        if not records:
            return 0

        d0 = asdict(records[0])
        d0.pop("id", None)
        columns = ", ".join(d0.keys())
        placeholders = ", ".join("?" * len(d0))

        values_list = []
        for r in records:
            d = asdict(r)
            d.pop("id", None)
            d.setdefault("timestamp", datetime.now().isoformat())
            values_list.append(list(d.values()))

        with self._get_conn() as conn:
            conn.executemany(
                f"INSERT INTO inspection_records ({columns}) VALUES ({placeholders})",
                values_list,
            )
            conn.commit()

        return len(records)

    def update_review(
        self,
        record_id: int,
        final_result: dict,
        reviewer: str,
        review_status: str = "confirmed",
        note: str = "",
    ) -> bool:
        """更新审核结果"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """UPDATE inspection_records
                   SET final_result = ?,
                       review_status = ?,
                       reviewer = ?,
                       review_time = ?,
                       note = ?
                   WHERE id = ?""",
                (
                    json.dumps(final_result, ensure_ascii=False),
                    review_status,
                    reviewer,
                    datetime.now().isoformat(),
                    note,
                    record_id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    # ==================== 查询操作 ====================

    def get_by_id(self, record_id: int) -> Optional[InspectionRecord]:
        """按 ID 查询"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM inspection_records WHERE id = ?", (record_id,)
            ).fetchone()
            return self._row_to_record(row) if row else None

    def query(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        defect_type: Optional[str] = None,
        review_status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[InspectionRecord]:
        """多条件查询检测记录"""
        conditions = []
        params = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        if defect_type:
            conditions.append("defect_types LIKE ?")
            params.append(f"%{defect_type}%")
        if review_status:
            conditions.append("review_status = ?")
            params.append(review_status)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM inspection_records WHERE {where} "
                f"ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()

        return [self._row_to_record(r) for r in rows]

    def count(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        defect_type: Optional[str] = None,
    ) -> int:
        """统计记录数量"""
        conditions = []
        params = []

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        if defect_type:
            conditions.append("defect_types LIKE ?")
            params.append(f"%{defect_type}%")

        where = " AND ".join(conditions) if conditions else "1=1"

        with self._get_conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM inspection_records WHERE {where}", params
            ).fetchone()
            return row[0] if row else 0

    def get_defect_stats(
        self, start_time: Optional[str] = None, end_time: Optional[str] = None
    ) -> list[dict]:
        """按缺陷类型统计"""
        conditions = []
        params = []
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        where = " AND ".join(conditions) if conditions else "1=1"

        # 这里使用简化统计，复杂统计建议在应用层做
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT defect_types, defect_count FROM inspection_records "
                f"WHERE {where} AND defect_count > 0",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    # ==================== 备份 ====================

    def backup(self, backup_path: Optional[str] = None) -> str:
        """备份数据库"""
        if backup_path is None:
            backup_path = f"data/inspection_backup_{datetime.now():%Y%m%d_%H%M%S}.db"
        shutil.copy2(self.db_path, backup_path)
        return backup_path

    # ==================== 工具方法 ====================

    def _row_to_record(self, row) -> InspectionRecord:
        """将数据库行转换为 InspectionRecord，自动过滤无效字段"""
        row_dict = dict(row)
        # 只保留 InspectionRecord 定义的字段，忽略数据库中多余的列
        valid_fields = {f.name for f in fields(InspectionRecord)}
        filtered = {k: v for k, v in row_dict.items() if k in valid_fields}
        return InspectionRecord(**filtered)

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
