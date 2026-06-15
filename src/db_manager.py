"""
数据持久化模块 - SQLite 数据库管理。

负责检测记录的增删改查、批量写入和备份。
"""

import json
import os
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
from contextlib import contextmanager
from loguru import logger

# 可选 SQLCipher 支持
try:
    import sqlcipher3
    HAS_SQLCIPHER = True
except ImportError:
    HAS_SQLCIPHER = False
    logger.info("sqlcipher3 未安装，使用标准 SQLite3（无加密）")


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
    engine: str = "yolo"          # 检测引擎类型 (yolo/vlm)
    id: Optional[int] = None


class DBManager:
    """SQLite 数据库管理器 (支持 SQLCipher 加密)"""

    def __init__(self, db_path: str = "data/inspection.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._all_conns = set()
        self._conns_lock = threading.Lock()
        
        # 获取加密密钥，必须从环境变量读取
        self.encryption_key = os.getenv("DB_ENCRYPTION_KEY", "")
        if not self.encryption_key:
            if HAS_SQLCIPHER:
                logger.warning("DB_ENCRYPTION_KEY 未设置，数据库将以明文存储；生产环境必须设置此环境变量")
            self.encryption_key = ""
        
        # 自动迁移已有的明文数据库为 SQLCipher 密文数据库
        self._migrate_to_sqlcipher()
        
        self._init_db()

    def _migrate_to_sqlcipher(self) -> None:
        """明文数据库转密文数据库的自动迁移逻辑"""
        if not HAS_SQLCIPHER:
            return
        if not self.encryption_key:
            return
        if not os.path.exists(self.db_path):
            return

        # 检测是否为明文数据库
        is_plaintext = False
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("SELECT count(*) FROM sqlite_master")
            conn.close()
            is_plaintext = True
        except Exception:
            # 无法以明文方式读取，说明已经是密文或不是有效SQLite
            pass

        if is_plaintext:
            logger.info(f"检测到明文数据库 {self.db_path}，正在进行 SQLCipher 加密迁移...")
            encrypted_path = self.db_path + ".encrypted"
            if os.path.exists(encrypted_path):
                os.remove(encrypted_path)

            try:
                # 打开明文库并附加加密新库
                conn = sqlcipher3.connect(self.db_path)
                conn.execute(f"ATTACH DATABASE '{encrypted_path}' AS encrypted KEY '{self.encryption_key}'")
                conn.execute("SELECT sqlcipher_export('encrypted')")
                conn.execute("DETACH DATABASE encrypted")
                conn.close()

                # 文件备份与替换
                backup_path = self.db_path + ".plaintext.bak"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                shutil.move(self.db_path, backup_path)
                shutil.move(encrypted_path, self.db_path)
                logger.info(f"数据库加密成功！明文备份已保存至: {backup_path}")
            except Exception as e:
                logger.error(f"数据库加密迁移失败: {e}")
                if os.path.exists(encrypted_path):
                    os.remove(encrypted_path)

    def _init_db(self) -> None:
        """初始化数据库表并升级列"""
        try:
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
                        note            TEXT DEFAULT '',
                        engine          TEXT DEFAULT 'yolo'
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
                
                # 检测并在线升级已有老数据库结构
                cursor = conn.execute("PRAGMA table_info(inspection_records)")
                columns = [row['name'] for row in cursor.fetchall()]
                if 'engine' not in columns:
                    conn.execute("ALTER TABLE inspection_records ADD COLUMN engine TEXT DEFAULT 'yolo'")
                    logger.info("数据库结构升级：为 'inspection_records' 添加了 'engine' 字段")
                    
                conn.commit()
        except Exception as e:
            logger.warning(f"数据库打开失败 ({e})，备份旧文件并重建...")
            backup_path = self.db_path + ".backup"
            try:
                if os.path.exists(self.db_path):
                    import shutil
                    shutil.copy2(self.db_path, backup_path)
                os.remove(self.db_path)
            except Exception:
                pass
            self._local.conn = None
            self._init_db()

    @contextmanager
    def _get_conn(self):
        """获取线程安全的数据库连接"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            if HAS_SQLCIPHER and self.encryption_key:
                conn = sqlcipher3.connect(self.db_path, timeout=5.0)
                conn.execute(f"PRAGMA key = '{self.encryption_key}'")
                conn.row_factory = sqlcipher3.Row
            else:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                conn.row_factory = sqlite3.Row
            self._local.conn = conn
            with self._conns_lock:
                self._all_conns.add(conn)
            try:
                self._local.conn.execute("PRAGMA journal_mode=WAL")
                self._local.conn.execute("PRAGMA synchronous=NORMAL")
            except Exception as e:
                logger.warning(f"无法设置 WAL 模式: {e}")
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

    def get_audited_dataset(self) -> list[dict]:
        """获取经审核修正或确权后的检测记录数据集"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM inspection_records WHERE review_status IN ('corrected', 'confirmed') ORDER BY timestamp DESC"
            ).fetchall()
            return [dict(r) for r in rows]

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
        return InspectionRecord(**dict(row))

    def close(self) -> None:
        with self._conns_lock:
            for conn in list(self._all_conns):
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_conns.clear()
        if hasattr(self._local, "conn"):
            self._local.conn = None
