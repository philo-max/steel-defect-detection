"""
数据库模块单元测试。
"""

import os
import tempfile

import pytest

from src.db_manager import DBManager, InspectionRecord


@pytest.fixture
def db():
    """创建临时数据库"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DBManager(path)
    yield db
    db.close()
    os.unlink(path)


class TestDBManager:
    def test_insert_and_query(self, db):
        record = InspectionRecord(
            image_path="test.jpg",
            defect_types="crack,scratch",
            defect_count=2,
            confidence=0.95,
        )
        rid = db.insert(record)
        assert rid > 0

        fetched = db.get_by_id(rid)
        assert fetched is not None
        assert fetched.defect_count == 2
        assert "crack" in fetched.defect_types

    def test_insert_batch(self, db):
        records = [
            InspectionRecord(image_path=f"test_{i}.jpg", defect_count=i)
            for i in range(5)
        ]
        count = db.insert_batch(records)
        assert count == 5
        assert db.count() == 5

    def test_update_review(self, db):
        record = InspectionRecord(image_path="test.jpg")
        rid = db.insert(record)

        ok = db.update_review(
            rid,
            final_result={"defects": []},
            reviewer="张三",
            review_status="confirmed",
            note="无缺陷",
        )
        assert ok

        updated = db.get_by_id(rid)
        assert updated.review_status == "confirmed"
        assert updated.reviewer == "张三"

    def test_query_filters(self, db):
        db.insert(InspectionRecord(
            image_path="a.jpg",
            defect_types="crack",
            review_status="pending",
        ))
        db.insert(InspectionRecord(
            image_path="b.jpg",
            defect_types="scratch",
            review_status="confirmed",
        ))

        pending = db.query(review_status="pending")
        assert len(pending) == 1
        assert pending[0].defect_types == "crack"

        scratch = db.query(defect_type="scratch")
        assert len(scratch) == 1

    def test_count(self, db):
        assert db.count() == 0
        db.insert(InspectionRecord(image_path="a.jpg"))
        db.insert(InspectionRecord(image_path="b.jpg"))
        assert db.count() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
